// The page side of the loader Worker: one job per file (or for the demo set),
// tracked in the app state as reading -> parsing -> adding -> done (or error).
// The page only adds the parsed arrays to the store.
//
// Large-file UX (#113): jobs carry progress (the Worker's coarse stage
// fractions), cancel drops the worker job and the pending add (no orphan
// meshes), and heavy loads are decimated on load through the #118 simplify
// op, with a post-load summary on the done job.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../scene/app-state";
import { BIG_FILE_DEFAULTS, parseBigFileParams, summarizeLoad } from "./big-file";
import { isDicom } from "./dicom";
import { PHANTOM_NAMES, type PhantomKind } from "./phantom";
import type { LoadedMesh, LoaderReply, LoaderRequest, VolumePayload } from "./protocol";
import { SAMPLE_NAMES, type SampleKind } from "./vtkjs";

/** How long a finished job stays listed. */
const DONE_LINGER_MS = 2500;

/**
 * Whether a dropped file is DICOM: by extension, else by the DICM prefix
 * (Part-10 files often carry no extension). Reads 132 bytes at most.
 */
async function isDicomFile(file: File): Promise<boolean> {
  const ext = /\.[^./\\]+$/.exec(file.name)?.[0].toLowerCase() ?? "";
  if (ext === ".dcm" || ext === ".dicom") {
    return true;
  }
  if (
    ext === ".stl" ||
    ext === ".obj" ||
    ext === ".ply" ||
    ext === ".3mf" ||
    ext === ".gltf" ||
    ext === ".glb" ||
    ext === ".drc"
  ) {
    return false;
  }
  try {
    return isDicom(await file.slice(0, 132).arrayBuffer());
  } catch {
    return false;
  }
}

/**
 * A drop batch's `.bin` files, when a `.gltf`/`.glb` dropped with them can
 * resolve them as external buffers; otherwise none (a lone `.bin` stays a
 * regular file, whose job reports it as unreadable, as before).
 */
export function siblingBinFiles(files: readonly File[]): File[] {
  if (!files.some((f) => isGltfName(f.name))) {
    return [];
  }
  return files.filter((f) => /\.bin$/i.test(f.name));
}

function isGltfName(name: string): boolean {
  return /\.gl(?:tf|b)$/i.test(name);
}

export class Loader {
  /** The loader Worker; `null` after it failed, until the next load starts a new one. */
  #worker: Worker | null = null;
  readonly #store: SceneStore;
  readonly #app: AppState;
  readonly #waiting = new Map<number, { name: string; resolve: (items: Item[]) => void }>();
  /** Jobs landing (added, decimating, or awaiting the scene): cancellable. */
  readonly #landing = new Map<number, { name: string; resolve: (items: Item[]) => void; cancel: () => void }>();
  /** Jobs the user cancelled: their late worker replies are dropped. */
  readonly #cancelled = new Set<number>();
  /** Load start times, for the post-load summary. */
  readonly #startedAt = new Map<number, number>();
  #nextJob = 0;
  /** Fetched source volumes by series id, for the volume view. */
  readonly #volumes = new Map<string, VolumePayload>();
  readonly #pendingVolumes = new Map<number, { resolve: (v: VolumePayload) => void; reject: (e: Error) => void }>();

  constructor(store: SceneStore, app: AppState) {
    this.#store = store;
    this.#app = app;
    // Evict volumes whose mesh left (removed): both the page copy and the
    // Worker's retained one (a CT stack is tens of megabytes).
    app.subscribe(() => {
      const live = new Set<string>();
      for (const item of app.getSnapshot().items) {
        if (item.volumeId !== null) {
          live.add(item.volumeId);
        }
      }
      for (const id of this.#volumes.keys()) {
        if (!live.has(id)) {
          this.#volumes.delete(id);
          this.#worker?.postMessage({ kind: "dropVolume", volumeId: id });
        }
      }
    });
  }

  /**
   * Fetches a retained source volume (cached after the first fetch).
   * `@throws` when the Worker no longer retains it.
   */
  fetchVolume(volumeId: string): Promise<VolumePayload> {
    const cached = this.#volumes.get(volumeId);
    if (cached !== undefined) {
      return Promise.resolve(cached);
    }
    const id = this.#nextJob++;
    return new Promise<VolumePayload>((resolve, reject) => {
      this.#pendingVolumes.set(id, {
        resolve: (v) => {
          this.#volumes.set(volumeId, v);
          resolve(v);
        },
        reject,
      });
      this.#workerFor().postMessage({ kind: "volume", id, volumeId });
    });
  }

  /**
   * Loads files; resolves with the meshes added (none for a file that
   * failed: its job shows why). DICOM files go as one job: the Worker
   * stacks their series and meshes each isosurface. `.bin` files dropped
   * with a `.gltf`/`.glb` ride on its job as sibling buffers.
   */
  async loadFiles(files: Iterable<File>): Promise<Item[]> {
    const list = [...files];
    const siblings = siblingBinFiles(list);
    const mesh: File[] = [];
    const dicom: File[] = [];
    for (const file of list) {
      if (siblings.includes(file)) {
        continue;
      }
      (await isDicomFile(file) ? dicom : mesh).push(file);
    }
    // Mesh files only: a DICOM series states its own spacing, in mm.
    const { importScale } = this.#app.getSnapshot();
    const jobs = mesh.map((file) =>
      isGltfName(file.name) && siblings.length > 0
        ? this.#start(file.name, { kind: "file", id: 0, file, buffers: siblings, importScale })
        : this.#start(file.name, { kind: "file", id: 0, file, importScale }),
    );
    if (dicom.length > 0) {
      const name = dicom.length === 1 ? dicom[0].name : `DICOM series (${dicom.length} files)`;
      jobs.push(this.#start(name, { kind: "dicom", id: 0, files: dicom, threshold: null }));
    }
    return (await Promise.all(jobs)).flat();
  }

  /** Generates and adds the demo meshes A, B and C. */
  loadDemo(big: boolean | number): Promise<Item[]> {
    return this.#start(big ? "demo meshes (big)" : "demo meshes", { kind: "demo", id: 0, big });
  }

  /** Meshes a procedural DICOM phantom through the extraction pipeline. */
  loadDicomDemo(phantom: PhantomKind): Promise<Item[]> {
    return this.#start(`${PHANTOM_NAMES[phantom]} (DICOM demo)`, { kind: "phantom", id: 0, phantom });
  }

  /** Meshes a bundled sample CT through the extraction pipeline. */
  loadSample(sample: SampleKind): Promise<Item[]> {
    return this.#start(`${SAMPLE_NAMES[sample]} (sample CT)`, { kind: "sample", id: 0, sample });
  }

  #workerFor(): Worker {
    if (this.#worker === null) {
      const worker = new Worker(new URL("./parser.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (e: MessageEvent<LoaderReply>) => this.#receive(e.data);
      // The Worker failed (its script did not load, it crashed) or a reply
      // could not be read: no job in it can finish. Fail them all and drop
      // the Worker; the next load starts a new one.
      worker.onerror = (e) => this.#fail(worker, e.message || "the loader Worker failed");
      worker.onmessageerror = () => this.#fail(worker, "a loader Worker reply could not be read");
      this.#worker = worker;
    }
    return this.#worker;
  }

  #fail(worker: Worker, message: string): void {
    if (this.#worker === worker) {
      this.#worker = null;
    }
    worker.terminate();
    for (const [id, job] of this.#waiting) {
      this.#app.setJob({ id, name: job.name, phase: "error", message });
      job.resolve([]);
    }
    this.#waiting.clear();
    for (const [id, landing] of this.#landing) {
      landing.cancel();
      this.#app.setJob({ id, name: landing.name, phase: "error", message });
    }
    for (const pending of this.#pendingVolumes.values()) {
      pending.reject(new Error(message));
    }
    this.#pendingVolumes.clear();
    this.#volumes.clear();
  }

  #start(
    name: string,
    request: Extract<LoaderRequest, { kind: "file" | "dicom" | "phantom" | "sample" | "demo" }>,
  ): Promise<Item[]> {
    const id = this.#nextJob++;
    this.#startedAt.set(id, Date.now());
    this.#app.setJob({ id, name, phase: "reading", progress: 0 });
    const { decimateTriangles, decimateTarget } = loadThresholds();
    const decimate =
      this.#app.getSnapshot().decimateOnLoad === true
        ? { threshold: decimateTriangles, target: decimateTarget }
        : undefined;
    const payload: LoaderRequest =
      decimate === undefined ? { ...request, id } : { ...request, id, decimate };
    return new Promise((resolve) => {
      this.#waiting.set(id, { name, resolve });
      this.#workerFor().postMessage(payload);
    });
  }

  /**
   * Cancels a load: the worker drops the job at its next stage boundary,
   * and a pending add (meshes already in the store, decimating, or awaiting
   * the scene) is removed again, so no orphan mesh stays behind. Settles the
   * job as `error: "cancelled"` and resolves its `loadFiles` caller with no
   * meshes. No-op for unknown or finished jobs.
   */
  cancelJob(id: number): void {
    const waiting = this.#waiting.get(id);
    const landing = this.#landing.get(id);
    if (waiting === undefined && landing === undefined) {
      return;
    }
    this.#cancelled.add(id);
    try {
      this.#worker?.postMessage({ kind: "cancel", id });
    } catch {
      // The worker is gone: settling below is what matters.
    }
    if (landing !== undefined) {
      landing.cancel();
      return;
    }
    if (waiting !== undefined) {
      this.#waiting.delete(id);
      this.#cancelled.delete(id);
      this.#startedAt.delete(id);
      this.#app.setJob({ id, name: waiting.name, phase: "error", message: "cancelled" });
      waiting.resolve([]);
    }
  }

  #receive(reply: LoaderReply): void {
    if (reply.kind === "volume") {
      const pending = this.#pendingVolumes.get(reply.id);
      if (pending !== undefined) {
        this.#pendingVolumes.delete(reply.id);
        pending.resolve(reply.volume);
      }
      return;
    }
    if (reply.kind === "error") {
      const pending = this.#pendingVolumes.get(reply.id);
      if (pending !== undefined) {
        this.#pendingVolumes.delete(reply.id);
        pending.reject(new Error(reply.message));
        return;
      }
    }
    const job = this.#waiting.get(reply.id);
    if (job === undefined) {
      return;
    }
    const { name } = job;
    if (reply.kind === "progress") {
      if (!this.#cancelled.has(reply.id)) {
        this.#app.setJob({ id: reply.id, name, phase: reply.phase, progress: clamp01(reply.fraction) });
      }
      return;
    }
    if (reply.kind === "parsing") {
      if (!this.#cancelled.has(reply.id)) {
        const cur = this.#app.getSnapshot().jobs.find((j) => j.id === reply.id);
        this.#app.setJob({ id: reply.id, name, phase: "parsing", progress: cur?.progress });
      }
      return;
    }
    this.#waiting.delete(reply.id);
    if (this.#cancelled.has(reply.id)) {
      // Cancelled while parsing: nothing was added page-side (no orphans).
      // (The worker may still post `error: "cancelled"`; it finds no waiter.)
      this.#cancelled.delete(reply.id);
      this.#startedAt.delete(reply.id);
      this.#app.setJob({ id: reply.id, name, phase: "error", message: "cancelled" });
      job.resolve([]);
      return;
    }
    if (reply.kind === "error") {
      this.#startedAt.delete(reply.id);
      this.#app.setJob({ id: reply.id, name, phase: "error", message: reply.message });
      job.resolve([]);
      return;
    }
    void this.#land(reply.id, name, reply.meshes, job.resolve);
  }

  /**
   * Adds the parsed (and optionally worker-decimated) meshes, then awaits
   * the scene. Registers in `#landing` before any await so `cancelJob` drops
   * pending adds cleanly with no orphan meshes.
   */
  async #land(id: number, name: string, meshes: LoadedMesh[], resolve: (items: Item[]) => void): Promise<void> {
    const t0 = this.#startedAt.get(id) ?? Date.now();
    const decimated = summarizeDecimated(meshes);
    this.#app.setJob({ id, name, phase: "adding", progress: decimated !== null ? 0.94 : 0.8 });
    let items: Item[] = [];
    try {
      items = meshes.map((m) => this.#app.addMesh(this.#store, m));
    } catch (error) {
      this.#startedAt.delete(id);
      this.#app.setJob({ id, name, phase: "error", message: String(error) });
      resolve([]);
      return;
    }
    let settled = false;
    const finish = (landed: Item[]): void => {
      if (!settled) {
        settled = true;
        resolve(landed);
      }
    };
    this.#landing.set(id, {
      name,
      resolve: finish,
      cancel: () => {
        if (settled) {
          return;
        }
        settled = true;
        for (const item of items) {
          if (this.#app.getSnapshot().items.includes(item)) {
            try {
              this.#app.removeMesh(this.#store, item);
            } catch {
              // Removed meanwhile: nothing to drop.
            }
          }
        }
        this.#landing.delete(id);
        this.#cancelled.delete(id);
        this.#startedAt.delete(id);
        this.#app.setJob({ id, name, phase: "error", message: "cancelled" });
        resolve([]);
      },
    });
    this.#awaitAdded(id, name, items, t0, decimated, finish);
  }

  /**
   * Keeps the job at "adding" until the scene reports every mesh added (its
   * mirror arrived from a Worker scene; the next frame on a local one). A mesh
   * removed meanwhile no longer counts. When the scene errors first (one of
   * these meshes, or the scene turning to "error": a poisoned Worker, which
   * Restart replays), the job fails instead of waiting. An error the scene
   * already had is someone else's.
   */
  #awaitAdded(id: number, name: string, items: Item[], t0: number, decimated: { from: number; to: number } | null, resolve: (items: Item[]) => void): void {
    const pending = new Set(items.map((i) => i.mesh));
    const erredBefore = this.#store.getSnapshot().status === "error";
    let settled = false;
    const entry = this.#landing.get(id);
    const settle = (error?: string) => {
      settled = true;
      offScene();
      offStore();
      this.#landing.delete(id);
      this.#startedAt.delete(id);
      const triangles = items.reduce((n, i) => n + i.triangles, 0);
      if (error === undefined) {
        const summary = summarizeLoad({ name, triangles, loadMs: Date.now() - t0, decimated });
        this.#app.setJob({ id, name, phase: "done", progress: 1, summary });
        setTimeout(() => this.#app.dropJob(id), DONE_LINGER_MS);
      } else {
        this.#app.setJob({ id, name, phase: "error", message: error });
      }
      resolve(items);
    };
    // A cancel during the wait drops the pending add through the same settle.
    if (entry !== undefined) {
      const outerCancel = entry.cancel;
      entry.cancel = () => {
        if (settled) {
          return;
        }
        settled = true;
        offScene();
        offStore();
        outerCancel();
      };
    }
    const check = () => {
      if (settled) {
        return;
      }
      const snap = this.#store.getSnapshot();
      let error: string | undefined;
      for (const mesh of pending) {
        const state = snap.meshes.get(mesh);
        if (state === undefined) {
          pending.delete(mesh);
        } else if (state.status === "error") {
          error = state.error?.message ?? "unknown error";
        }
      }
      if (error === undefined && !erredBefore && snap.status === "error") {
        error = snap.error?.message ?? "unknown error";
      }
      if (error !== undefined) {
        settle(`the scene failed while adding: ${error}`);
      } else if (pending.size === 0) {
        settle();
      }
    };
    const offScene = this.#store.scene.onChanges((changes) => {
      for (const c of changes) {
        if (c.added || c.removed) {
          pending.delete(c.mesh);
        }
      }
      check();
    });
    const offStore = this.#store.subscribe(check);
    check();
  }
}

/** Combines per-mesh worker decimation tallies for the job summary (`null` when none ran). */
function summarizeDecimated(meshes: readonly LoadedMesh[]): { from: number; to: number } | null {
  let from = 0;
  let to = 0;
  let any = false;
  for (const m of meshes) {
    if (m.decimated !== undefined) {
      any = true;
      from += m.decimated.from;
      to += m.decimated.to;
    }
  }
  return any ? { from, to } : null;
}

/** Clamps a worker fraction into 0..1 (a corrupt reply never widens the bar). */
function clamp01(fraction: number): number {
  return Number.isFinite(fraction) ? Math.min(1, Math.max(0, fraction)) : 0;
}

/** The warn/decimate thresholds, from the URL for testing (`location` is absent in unit tests). */
function loadThresholds(): { warnTriangles: number; decimateTriangles: number; decimateTarget: number } {
  try {
    if (typeof location !== "undefined") {
      const p = parseBigFileParams(location.search);
      return { warnTriangles: p.warnTriangles, decimateTriangles: p.decimateTriangles, decimateTarget: p.decimateTarget };
    }
  } catch {
    // Fall through to the defaults.
  }
  return {
    warnTriangles: BIG_FILE_DEFAULTS.warnTriangles,
    decimateTriangles: BIG_FILE_DEFAULTS.decimateTriangles,
    decimateTarget: BIG_FILE_DEFAULTS.decimateTarget,
  };
}
