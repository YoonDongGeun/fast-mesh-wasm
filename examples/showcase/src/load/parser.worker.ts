// The loader Worker: reads and parses mesh files, and generates the demo
// meshes, off the main thread. Results go back as transferred typed arrays,
// so the page only calls `store.addMesh` (which transfers them on to the
// scene Worker).
import { demoMeshes } from "../scene/demo";
import { extractSeries, meshSlices } from "./extract";
import { bounds, type MeshData } from "./mesh-data";
import { parseMeshFile } from "./parse";
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

self.onmessage = async (event: MessageEvent<LoaderRequest>) => {
  const req = event.data;
  if (req.kind === "dropVolume") {
    volumes.delete(req.volumeId);
    return;
  }
  try {
    if (req.kind === "demo") {
      post({ kind: "meshes", id: req.id, meshes: demoMeshes(req.big).map((m) => loaded(m.name, m.data)) });
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
      for (const file of req.files) {
        parsed.push({ name: file.name, buffer: await file.arrayBuffer() });
      }
      post({ kind: "parsing", id: req.id });
      post({
        kind: "meshes",
        id: req.id,
        meshes: extractSeries(parsed, req.threshold).map((m) => {
          retain(m.volume, false);
          return loaded(m.name, m.data, m.seriesUid);
        }),
      });
      return;
    }
    if (req.kind === "phantom") {
      post({ kind: "parsing", id: req.id });
      post({
        kind: "meshes",
        id: req.id,
        meshes: meshSlices(phantomSlices(req.phantom), 0).map((m) => {
          retain(m.volume, PHANTOM_FIELD[req.phantom] === "sdf");
          return loaded(m.name, m.data, m.seriesUid);
        }),
      });
      return;
    }
    if (req.kind === "sample") {
      post({ kind: "parsing", id: req.id });
      const { slices, iso, preset } = await loadSample(req.sample, import.meta.env.BASE_URL);
      post({
        kind: "meshes",
        id: req.id,
        meshes: meshSlices(slices, iso).map((m) => {
          retain(m.volume, false);
          return loaded(m.name, m.data, m.seriesUid, preset);
        }),
      });
      return;
    }
    const buffer = await req.file.arrayBuffer();
    post({ kind: "parsing", id: req.id });
    post({ kind: "meshes", id: req.id, meshes: [loaded(req.file.name, parseMeshFile(req.file.name, buffer))] });
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
