// The export path: a scene mesh out to a downloadable file. The page reads
// the mesh through the store's scene (a `SceneLike` for local and Worker
// scenes alike) and the `fast-mesh-wasm/io` encoders do the rest, so meshes
// edited, repaired or colored in the session export as they are.
import type { MeshHandle, SceneLike } from "fast-mesh-wasm";
import { export3MFSync, exportGLTF, exportOBJ, exportPLY, exportSTL } from "fast-mesh-wasm/io";

/** The formats the showcase offers for download. */
export type MeshExportFormat = "stl" | "obj" | "ply" | "3mf" | "gltf" | "glb";

/** The file extension (and download MIME type) of each format. */
export function exportFileInfo(format: MeshExportFormat): { extension: string; mime: string } {
  switch (format) {
    case "stl":
      return { extension: "stl", mime: "model/stl" };
    case "obj":
      return { extension: "obj", mime: "text/plain" };
    case "ply":
      return { extension: "ply", mime: "text/plain" };
    case "3mf":
      return { extension: "3mf", mime: "model/3mf" };
    case "gltf":
      return { extension: "gltf", mime: "model/gltf+json" };
    case "glb":
      return { extension: "glb", mime: "model/gltf-binary" };
  }
}

/**
 * Encodes one mesh. `name` seeds the format's internal name (`solid`, `o`,
 * glTF node, 3MF object); the bytes always hold world coordinates. Throws what
 * the encoders throw (unknown mesh, missing live triangles are left to them).
 */
export function exportMeshBytes(
  scene: SceneLike,
  mesh: MeshHandle,
  format: MeshExportFormat,
  name: string,
): Uint8Array {
  switch (format) {
    case "stl":
      return exportSTL(scene, mesh, { name });
    case "obj":
      return exportOBJ(scene, mesh, { name });
    case "ply":
      return exportPLY(scene, mesh);
    case "3mf":
      return export3MFSync(scene, mesh, { name });
    case "gltf":
      return exportGLTF(scene, mesh, { binary: false, name });
    case "glb":
      return exportGLTF(scene, mesh, { name });
  }
}

/** A download file name: the item name with the format's extension. */
export function exportFileName(name: string, format: MeshExportFormat): string {
  const stem = name.replace(/[\\/:*?"<>|]/g, "_").trim() || "mesh";
  return `${stem}.${exportFileInfo(format).extension}`;
}

/**
 * Encodes one mesh and triggers a browser download (an object URL on a
 * temporary anchor). A UI helper only: it touches `document` and `URL`.
 */
export function downloadMesh(scene: SceneLike, mesh: MeshHandle, name: string, format: MeshExportFormat): void {
  const bytes = exportMeshBytes(scene, mesh, format, name);
  const { mime } = exportFileInfo(format);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const url = URL.createObjectURL(new Blob([copy.buffer as ArrayBuffer], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = exportFileName(name, format);
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
