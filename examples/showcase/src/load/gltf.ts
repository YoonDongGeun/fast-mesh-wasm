// glTF 2.0 and GLB, uncompressed buffers only: nodes/meshes/primitives become
// one mesh each, node placements baked to world coordinates. Vertex colors ride
// along in the file but stay out of `MeshData`: the showcase shades meshes
// with its own palette materials (like the OBJ loader ignoring `v r g b`).
// Animations, skins and materials beyond vertex colors are out of scope.
// Draco-compressed primitives (`KHR_draco_mesh_compression`) decode through
// `parseGltfAsync` (draco3d in the loader Worker); the sync entry throws on
// them, for files known to be uncompressed.
import { importGLTF, importGLTFAsync, type ImportedMesh } from "fast-mesh-wasm/io";
import { decoderModule } from "./draco";
import type { MeshData } from "./mesh-data";

export interface NamedMeshData {
  name: string;
  data: MeshData;
}

/**
 * Parses a glTF asset into its meshes. A single-mesh file keeps the file
 * name (like every other format); a multi-mesh one names each after the file
 * stem and its node (or mesh) name.
 *
 * @throws When the asset is malformed, Draco-compressed, externally buffered
 * or holds no triangle meshes.
 */
export function parseGltf(name: string, buffer: ArrayBuffer): NamedMeshData[] {
  return nameParts(name, importGLTF(buffer));
}

/**
 * Parses a glTF asset like {@link parseGltf}, decoding `KHR_draco_mesh_compression`
 * primitives through the draco3d decoder. Uncompressed assets never touch the
 * decoder: they parse on the sync path.
 *
 * @throws When the asset is malformed, Draco-undecodable, externally
 * buffered or holds no triangle meshes.
 */
export async function parseGltfAsync(name: string, buffer: ArrayBuffer): Promise<NamedMeshData[]> {
  try {
    return parseGltf(name, buffer);
  } catch (error) {
    if (!(error instanceof Error) || !error.message.includes("KHR_draco_mesh_compression")) {
      throw error;
    }
  }
  return nameParts(name, await importGLTFAsync(buffer, { decoderModule: await decoderModule() }));
}

/** Names each part after the file: alone the file name, else stem plus node (or mesh) name. */
function nameParts(name: string, parts: ImportedMesh[]): NamedMeshData[] {
  if (parts.length === 1) {
    const only = parts[0];
    return [{ name, data: { positions: only.positions, indices: only.indices } }];
  }
  const stem = name.replace(/\.[^./\\]+$/, "") || name;
  return parts.map((part) => ({
    name: `${stem} · ${part.name}`,
    data: { positions: part.positions, indices: part.indices },
  }));
}
