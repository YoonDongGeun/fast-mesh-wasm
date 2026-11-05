// The loader Worker: reads and parses mesh files, and generates the demo
// meshes, off the main thread. Results go back as transferred typed arrays,
// so the page only calls `store.addMesh` (which transfers them on to the
// scene Worker).
import { demoMeshes } from "../scene/demo";
import { extractSeries, meshSlices } from "./extract";
import { bounds, type MeshData } from "./mesh-data";
import { NO_IMPORT_SCALE, scalePositions } from "../panels/units";
import { parseMeshFilesAsync } from "./parse";
import { PHANTOM_FIELD, phantomSlices } from "./phantom";
import type { LoadedMesh, LoaderReply, LoaderRequest, VolumePayload } from "./protocol";
import type { DicomVolume } from "./series";
import { loadSample } from "./vtkjs";

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

function progress(id: number, phase: "reading" | "parsing", fraction: number): void {
  post({ kind: "progress", id, phase, fraction });
}

self.onmessage = async (event: MessageEvent<LoaderRequest>) => {
  const req = event.data;
  if (req.kind === "dropVolume") {
    volumes.delete(req.volumeId);
    return;
  }
  if (req.kind === "cancel") {
    cancelled.add(req.id);
    return;
  }
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
      post({ kind: "meshes", id: req.id, meshes });
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
      post({
        kind: "meshes",
        id: req.id,
        meshes: extracted.map((m) => {
          retain(m.volume, false);
          return loaded(m.name, m.data, m.seriesUid);
        }),
      });
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
      post({
        kind: "meshes",
        id: req.id,
        meshes: meshed.map((m) => {
          retain(m.volume, PHANTOM_FIELD[req.phantom] === "sdf");
          return loaded(m.name, m.data, m.seriesUid);
        }),
      });
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
      post({
        kind: "meshes",
        id: req.id,
        meshes: meshed.map((m) => {
          retain(m.volume, false);
          return loaded(m.name, m.data, m.seriesUid, preset);
        }),
      });
      return;
    }
    const importScale = req.importScale ?? NO_IMPORT_SCALE;
    const buffer = await req.file.arrayBuffer();
    progress(req.id, "reading", 0.15);
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
    post({
      kind: "meshes",
      id: req.id,
      meshes: parsed.map((m) => {
        // Before `loaded` takes the bounds: centre and radius are in mm too.
        scalePositions(m.data.positions, importScale.factor);
        return { ...loaded(m.name, m.data), importScale };
      }),
    });
  } catch (error) {
    post({ kind: "error", id: req.id, message: error instanceof Error ? error.message : String(error) });
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
