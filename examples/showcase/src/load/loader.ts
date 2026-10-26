// The page side of the loader Worker: one job per file (or for the demo set),
// tracked in the app state as reading -> parsing -> adding -> done (or error).
// The page only adds the parsed arrays to the store.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../scene/app-state";
import { isDicom } from "./dicom";
import { PHANTOM_NAMES, type PhantomKind } from "./phantom";
import type { LoaderReply, LoaderRequest } from "./protocol";

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
  if (ext === ".stl" || ext === ".obj" || ext === ".ply") {
    return false;
  }
  try {
    return isDicom(await file.slice(0, 132).arrayBuffer());
  } catch {
    return false;
  }
}

export class Loader {
  /** The loader Worker; `null` after it failed, until the next load starts a new one. */
  #worker: Worker | null = null;
  readonly #store: SceneStore;
  readonly #app: AppState;
  readonly #waiting = new Map<number, { name: string; resolve: (items: Item[]) => void }>();
  #nextJob = 0;

  constructor(store: SceneStore, app: AppState) {
    this.#store = store;
    this.#app = app;
  }

  /**
   * Loads files; resolves with the meshes added (none for a file that
   * failed: its job shows why). DICOM files go as one job: the Worker
   * stacks their series and meshes each isosurface.
   */
  async loadFiles(files: Iterable<File>): Promise<Item[]> {
    const mesh: File[] = [];
    const dicom: File[] = [];
    for (const file of files) {
      (await isDicomFile(file) ? dicom : mesh).push(file);
    }
    const jobs = mesh.map((file) => this.#start(file.name, { kind: "file", id: 0, file }));
    if (dicom.length > 0) {
      const name = dicom.length === 1 ? dicom[0].name : `DICOM series (${dicom.length} files)`;
      jobs.push(this.#start(name, { kind: "dicom", id: 0, files: dicom, threshold: null }));
    }
    return (await Promise.all(jobs)).flat();
  }

  /** Generates and adds the demo meshes A, B and C. */
  loadDemo(big: boolean): Promise<Item[]> {
    return this.#start(big ? "demo meshes (big)" : "demo meshes", { kind: "demo", id: 0, big });
  }

  /** Meshes a procedural DICOM phantom through the extraction pipeline. */
  loadDicomDemo(phantom: PhantomKind): Promise<Item[]> {
    return this.#start(`${PHANTOM_NAMES[phantom]} (DICOM demo)`, { kind: "phantom", id: 0, phantom });
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
  }

  #start(name: string, request: LoaderRequest): Promise<Item[]> {
    const id = this.#nextJob++;
    this.#app.setJob({ id, name, phase: "reading" });
    return new Promise((resolve) => {
      this.#waiting.set(id, { name, resolve });
      this.#workerFor().postMessage({ ...request, id });
    });
  }

  #receive(reply: LoaderReply): void {
    const job = this.#waiting.get(reply.id);
    if (job === undefined) {
      return;
    }
    const { name } = job;
    if (reply.kind === "parsing") {
      this.#app.setJob({ id: reply.id, name, phase: "parsing" });
      return;
    }
    this.#waiting.delete(reply.id);
    if (reply.kind === "error") {
      this.#app.setJob({ id: reply.id, name, phase: "error", message: reply.message });
      job.resolve([]);
      return;
    }
    this.#app.setJob({ id: reply.id, name, phase: "adding" });
    let items: Item[] = [];
    try {
      items = reply.meshes.map((m) => this.#app.addMesh(this.#store, m));
    } catch (error) {
      this.#app.setJob({ id: reply.id, name, phase: "error", message: String(error) });
      job.resolve([]);
      return;
    }
    this.#awaitAdded(reply.id, name, items, job.resolve);
  }

  /**
   * Keeps the job at "adding" until the scene reports every mesh added (its
   * mirror arrived from a Worker scene; the next frame on a local one). A mesh
   * removed meanwhile no longer counts. When the scene errors first (one of
   * these meshes, or the scene turning to "error": a poisoned Worker, which
   * Restart replays), the job fails instead of waiting. An error the scene
   * already had is someone else's.
   */
  #awaitAdded(id: number, name: string, items: Item[], resolve: (items: Item[]) => void): void {
    const pending = new Set(items.map((i) => i.mesh));
    const erredBefore = this.#store.getSnapshot().status === "error";
    let settled = false;
    const settle = (error?: string) => {
      settled = true;
      offScene();
      offStore();
      if (error === undefined) {
        this.#app.setJob({ id, name, phase: "done" });
        setTimeout(() => this.#app.dropJob(id), DONE_LINGER_MS);
      } else {
        this.#app.setJob({ id, name, phase: "error", message: error });
      }
      resolve(items);
    };
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
