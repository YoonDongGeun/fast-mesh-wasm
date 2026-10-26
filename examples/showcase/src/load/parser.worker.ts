// The loader Worker: reads and parses mesh files, and generates the demo
// meshes, off the main thread. Results go back as transferred typed arrays,
// so the page only calls `store.addMesh` (which transfers them on to the
// scene Worker).
import { demoMeshes } from "../scene/demo";
import { extractSeries } from "./extract";
import { bounds, type MeshData } from "./mesh-data";
import { parseMeshFile } from "./parse";
import type { LoadedMesh, LoaderReply, LoaderRequest } from "./protocol";

function post(reply: LoaderReply): void {
  const transfer = reply.kind === "meshes" ? reply.meshes.flatMap((m) => [m.positions.buffer, m.indices.buffer]) : [];
  self.postMessage(reply, { transfer });
}

function loaded(name: string, data: MeshData): LoadedMesh {
  const { center, radius } = bounds(data.positions);
  return { name, positions: data.positions, indices: data.indices, center, radius };
}

self.onmessage = async (event: MessageEvent<LoaderRequest>) => {
  const req = event.data;
  try {
    if (req.kind === "demo") {
      post({ kind: "meshes", id: req.id, meshes: demoMeshes(req.big).map((m) => loaded(m.name, m.data)) });
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
        meshes: extractSeries(parsed, req.threshold).map((m) => loaded(m.name, m.data)),
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
