// The loader Worker: reads and parses mesh files, generates the demo meshes,
// and decimates oversized loads off the main thread. Results go back as
// transferred typed arrays, so the page only calls `store.addMesh` (which
// transfers them on to the scene Worker).
import { demoMeshes } from "../scene/demo";
import { NO_IMPORT_SCALE, scalePositions } from "../panels/units";
import { decimateStepTargets, decimateTargetFor } from "./big-file";
import { extractSeries, meshSlices } from "./extract";
import { bounds, type MeshData } from "./mesh-data";
import { parseMeshFilesAsync } from "./parse";
import { PHANTOM_FIELD, phantomSlices } from "./phantom";
import type { DecimateOnLoadOptions, LoadedMesh, LoaderReply, LoaderRequest, VolumePayload } from "./protocol";
import type { DicomVolume } from "./series";
import { loadSample } from "./vtkjs";

let wasmReady: Promise<typeof import("fast-mesh-wasm")> | null = null;

function loadWasm(): Promise<typeof import("fast-mesh-wasm")> {
  if (wasmReady === null) {
    wasmReady = (async () => {
      const mod = await import("fast-mesh-wasm");
      await mod.init();
      return mod;
    })().catch((err) => {
      wasmReady = null;
      throw err;
    });
  }
  return wasmReady;
}

function yieldForCancel(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function post(reply: LoaderReply): void {
  const transfer =
    reply.kind === "meshes"
      ? reply.meshes.flatMap((m) => [m.positions.buffer, m.indices.buffer])
      : reply.kind === "volume"
        ? [reply.volume.values.buffer]
        : [];
  self.postMessage(reply, { transfer });
}

function loaded(
  name: string,
  data: MeshData,
  volumeId: string | null = null,
  preset: string | null = null,
): LoadedMesh {
  const { center, radius } = bounds(data.positions);
  return { name, positions: data.positions, indices: data.indices, center, radius, volumeId, preset };
}

/** Source volumes by series id, served to the volume view on demand. */
const volumes = new Map<string, { values: Float32Array; pseudoHu: boolean; meta: Omit<VolumePayload, "values" | "pseudoHu"> }>();

/**
 * Jobs the page cancelled: checked between stages (reads, parses, posts),
 * since a parse itself is one synchronous call (#64). A cancelled job posts
 * `error: "cancelled"` instead of its meshes; the page ignores late replies
 * for jobs it already settled, so either side may win the race.
 */
const cancelled = new Set<number>();
const activeAborts = new Map<number, AbortController>();

function isCancelled(id: number): boolean {
  if (cancelled.has(id)) {
    cancelled.delete(id);
    return true;
  }
  return false;
}

function postCancelled(id: number): void {
  post({ kind: "error", id, message: "cancelled" });
}

function progress(id: number, phase: "reading" | "parsing" | "decimating", fraction: number): void {
  post({ kind: "progress", id, phase, fraction });
}

async function readFileStreaming(
  id: number,
  file: File,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  if (typeof file.stream !== "function") {
    const buf = await file.arrayBuffer();
    progress(id, "reading", 0.15);
    return buf;
  }
  const reader = file.stream().getReader();
  const total = file.size;
  const chunks: Uint8Array[] = [];
  let bytesRead = 0;
  let lastProgressAt = -Infinity;

  const onAbort = () => {
    chunks.length = 0;
    void reader.cancel(signal.reason).catch(() => {});
  };
  signal.addEventListener("abort", onAbort, { once: true });

  try {
    progress(id, "reading", 0.02);
    while (true) {
      if (signal.aborted) {
        throw new DOMException("cancelled", "AbortError");
      }
      const { value, done } = await reader.read();
      if (done) break;
      if (signal.aborted) {
        throw new DOMException("cancelled", "AbortError");
      }
      if (value && value.byteLength > 0) {
        bytesRead += value.byteLength;
        chunks.push(new Uint8Array(value));
        const now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (now - lastProgressAt >= 16) {
          lastProgressAt = now;
          const frac = total > 0 ? Math.min(1, bytesRead / total) : 0.5;
          progress(id, "reading", 0.02 + 0.3 * frac);
        }
      }
    }
    progress(id, "reading", 0.32);
    const out = new Uint8Array(bytesRead);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      out.set(chunks[i], offset);
      offset += chunks[i].byteLength;
    }
    chunks.length = 0;
    return out.buffer;
  } catch (err) {
    chunks.length = 0;
    await reader.cancel(err).catch(() => {});
    throw err;
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}

async function maybeDecimate(
  id: number,
  meshes: LoadedMesh[],
  decimate?: DecimateOnLoadOptions,
): Promise<LoadedMesh[] | null> {
  if (decimate === undefined) {
    return meshes;
  }
  const total = meshes.reduce((n, m) => n + Math.round(m.indices.length / 3), 0);
  const plan = meshes
    .map((mesh) => {
      const from = Math.round(mesh.indices.length / 3);
      return decimateTargetFor(from, decimate.target) === null
        ? null
        : { mesh, from, steps: decimateStepTargets(from, decimate.target, 5) };
    })
    .filter((entry): entry is { mesh: LoadedMesh; from: number; steps: number[] } => entry !== null);
  if (total < decimate.threshold || plan.length === 0) {
    return meshes;
  }
  const totalSteps = plan.reduce((n, entry) => n + entry.steps.length, 0);
  if (totalSteps === 0) {
    return meshes;
  }

  progress(id, "decimating", 0.72);
  await yieldForCancel();
  if (isCancelled(id)) {
    return null;
  }
  const { Scene } = await loadWasm();
  if (isCancelled(id)) {
    return null;
  }

  let completedSteps = 0;
  for (const { mesh, from, steps } of plan) {
    const scene = new Scene();
    try {
      const h = scene.addMesh(mesh.positions, mesh.indices, { mode: "manual", normals: false });
      try {
        for (const stepTarget of steps) {
          await yieldForCancel();
          if (isCancelled(id)) {
            return null;
          }
          scene.repair(h, { op: "simplify", targetTriangles: stepTarget });
          completedSteps += 1;
          progress(id, "decimating", 0.72 + (0.2 * completedSteps) / totalSteps);
        }
        scene.compact(h);
        const views = scene.views(h);
        const posLocal = views.positions();
        const idx = views.indices().slice();
        const [ox, oy, oz] = scene.origin(h);
        const positions = new Float64Array(posLocal.length);
        for (let i = 0; i < posLocal.length; i += 3) {
          positions[i] = posLocal[i] + ox;
          positions[i + 1] = posLocal[i + 1] + oy;
          positions[i + 2] = posLocal[i + 2] + oz;
        }
        const nextBounds = bounds(positions);
        mesh.positions = positions;
        mesh.indices = idx;
        mesh.center = nextBounds.center;
        mesh.radius = nextBounds.radius;
        mesh.decimated = { from, to: Math.round(idx.length / 3) };
      } catch {
        // Leave original mesh intact if simplify rejects the geometry.
      }
    } finally {
      scene.free();
    }
  }
  return meshes;
}

async function postMeshes(id: number, meshes: LoadedMesh[], decimate?: DecimateOnLoadOptions): Promise<void> {
  const ready = await maybeDecimate(id, meshes, decimate);
  if (ready === null) {
    postCancelled(id);
    return;
  }
  post({ kind: "meshes", id, meshes: ready });
}

self.onmessage = async (event: MessageEvent<LoaderRequest>) => {
  const req = event.data;
  if (req.kind === "dropVolume") {
    volumes.delete(req.volumeId);
    return;
  }
  if (req.kind === "cancel") {
    cancelled.add(req.id);
    activeAborts.get(req.id)?.abort();
    return;
  }
  const abortCtrl = new AbortController();
  activeAborts.set(req.id, abortCtrl);
  try {
    if (req.kind === "demo") {
      progress(req.id, "parsing", 0.2);
      if (isCancelled(req.id)) {
        postCancelled(req.id);
        return;
      }
      const meshes = demoMeshes(req.big).map((m) => loaded(m.name, m.data));
      if (isCancelled(req.id)) {
        postCancelled(req.id);
        return;
      }
      await postMeshes(req.id, meshes, req.decimate);
      return;
    }
    if (req.kind === "volume") {
      const v = volumes.get(req.volumeId);
      if (v === undefined) {
        throw new Error(`volume ${req.volumeId} is no longer retained`);
      }
      // A copy: transferring would neuter the retained original.
      post({ kind: "volume", id: req.id, volume: { ...v.meta, values: v.values.slice(), pseudoHu: v.pseudoHu } });
      return;
    }
    if (req.kind === "dicom") {
      const parsed: { name: string; buffer: ArrayBuffer }[] = [];
      for (let i = 0; i < req.files.length; i++) {
        const file = req.files[i];
        parsed.push({ name: file.name, buffer: await file.arrayBuffer() });
        progress(req.id, "reading", 0.05 + (0.25 * (i + 1)) / req.files.length);
        if (isCancelled(req.id)) {
          postCancelled(req.id);
          return;
        }
      }
      post({ kind: "parsing", id: req.id });
      progress(req.id, "parsing", 0.45);
      const extracted = extractSeries(parsed, req.threshold);
      if (isCancelled(req.id)) {
        postCancelled(req.id);
        return;
      }
      progress(req.id, "parsing", 0.7);
      const meshes = extracted.map((m) => {
        retain(m.volume, false);
        return loaded(m.name, m.data, m.seriesUid);
      });
      await postMeshes(req.id, meshes, req.decimate);
      return;
    }
    if (req.kind === "phantom") {
      post({ kind: "parsing", id: req.id });
      progress(req.id, "parsing", 0.3);
      const meshed = meshSlices(phantomSlices(req.phantom), 0);
      if (isCancelled(req.id)) {
        postCancelled(req.id);
        return;
      }
      progress(req.id, "parsing", 0.7);
      const meshes = meshed.map((m) => {
        retain(m.volume, PHANTOM_FIELD[req.phantom] === "sdf");
        return loaded(m.name, m.data, m.seriesUid);
      });
      await postMeshes(req.id, meshes, req.decimate);
      return;
    }
    if (req.kind === "sample") {
      post({ kind: "parsing", id: req.id });
      const { slices, iso, preset } = await loadSample(req.sample, import.meta.env.BASE_URL);
      if (isCancelled(req.id)) {
        postCancelled(req.id);
        return;
      }
      progress(req.id, "parsing", 0.4);
      const meshed = meshSlices(slices, iso);
      if (isCancelled(req.id)) {
        postCancelled(req.id);
        return;
      }
      progress(req.id, "parsing", 0.7);
      const meshes = meshed.map((m) => {
        retain(m.volume, false);
        return loaded(m.name, m.data, m.seriesUid, preset);
      });
      await postMeshes(req.id, meshes, req.decimate);
      return;
    }
    const importScale = req.importScale ?? NO_IMPORT_SCALE;
    const buffer = await readFileStreaming(req.id, req.file, abortCtrl.signal);
    if (isCancelled(req.id)) {
      postCancelled(req.id);
      return;
    }
    // A `.gltf`/`.glb` job's sibling `.bin` files, for external buffers.
    let buffers: Map<string, ArrayBuffer> | undefined;
    if (req.buffers !== undefined && req.buffers.length > 0) {
      buffers = new Map();
      for (const sibling of req.buffers) {
        buffers.set(sibling.name, await sibling.arrayBuffer());
        if (isCancelled(req.id)) {
          postCancelled(req.id);
          return;
        }
      }
    }
    post({ kind: "parsing", id: req.id });
    const parsed = await parseMeshFilesAsync(req.file.name, buffer, buffers === undefined ? undefined : { buffers });
    if (isCancelled(req.id)) {
      postCancelled(req.id);
      return;
    }
    progress(req.id, "parsing", 0.7);
    const meshes = parsed.map((m) => {
      // Before `loaded` takes the bounds: centre and radius are in mm too.
      scalePositions(m.data.positions, importScale.factor);
      return { ...loaded(m.name, m.data), importScale };
    });
    await postMeshes(req.id, meshes, req.decimate);
  } catch (error) {
    if (isCancelled(req.id) || abortCtrl.signal.aborted) {
      postCancelled(req.id);
    } else {
      post({ kind: "error", id: req.id, message: error instanceof Error ? error.message : String(error) });
    }
  } finally {
    activeAborts.delete(req.id);
  }
};

function retain(volume: DicomVolume, pseudoHu: boolean): void {
  volumes.set(volume.seriesUid, {
    values: volume.values,
    pseudoHu,
    meta: {
      volumeId: volume.seriesUid,
      columns: volume.columns,
      rows: volume.rows,
      slices: volume.slices,
      spacing: volume.spacing,
      origin: volume.origin,
    },
  });
}
