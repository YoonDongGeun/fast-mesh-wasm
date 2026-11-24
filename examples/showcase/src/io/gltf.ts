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

/** Options of {@link parseGltfAsync}. */
export interface GltfParseOptions {
  /**
   * Sibling buffers by file name, for `.gltf` assets with external `.bin`
   * buffers (a drop batch's `.bin` files, read in the loader Worker).
   */
  buffers?: Map<string, ArrayBuffer>;
}

/**
 * Parses a glTF asset into its meshes. A single-mesh file keeps the file
 * name (like every other format); a multi-mesh one names each after the file
 * stem and its node (or mesh) name.
 *
 * @throws When the asset is malformed, Draco-compressed, externally buffered
 * (the async entry resolves sibling `.bin` files) or holds no triangle
 * meshes.
 */
export function parseGltf(name: string, buffer: ArrayBuffer): NamedMeshData[] {
  return nameParts(name, importGLTF(buffer));
}

/**
 * Parses a glTF asset like {@link parseGltf}, decoding `KHR_draco_mesh_compression`
 * primitives through the draco3d decoder and resolving external `.bin`
 * buffers from `options.buffers`. Self-contained uncompressed assets never
 * touch the decoder: they parse on the sync path.
 *
 * @throws When the asset is malformed, Draco-undecodable, externally
 * buffered without its sibling, or holds no triangle meshes.
 */
export async function parseGltfAsync(
  name: string,
  buffer: ArrayBuffer,
  options?: GltfParseOptions,
): Promise<NamedMeshData[]> {
  try {
    return parseGltf(name, buffer);
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    const draco = message.includes("KHR_draco_mesh_compression");
    const external =
      (options?.buffers?.size ?? 0) > 0 && message.includes("external");
    if (!draco && !external) {
      throw error;
    }
  }
  // The async entry always carries the decoder module: an external-buffer
  // asset can still hold Draco primitives, and telling that apart upfront
  // would mean a second parse. One cached module either way.
  return nameParts(
    name,
    await importGLTFAsync(buffer, {
      decoderModule: await decoderModule(),
      resolveBuffer: resolverFor(options?.buffers),
    }),
  );
}

/**
 * A `resolveBuffer` over the dropped siblings: the declared URI verbatim,
 * else its file name (a drop batch is flat, so `buffers/model.bin` matches
 * `model.bin`), each also URI-decoded (`%20` and friends). Anything else is
 * a missing sibling, not a fetch: the importer never leaves these bytes.
 */
function resolverFor(buffers?: Map<string, ArrayBuffer>): ((uri: string) => ArrayBuffer) | undefined {
  if (buffers === undefined || buffers.size === 0) {
    return undefined;
  }
  return (uri: string) => {
    const decoded = safeDecode(uri);
    for (const candidate of [uri, decoded]) {
      const hit = buffers.get(candidate) ?? buffers.get(candidate.slice(candidate.lastIndexOf("/") + 1));
      if (hit !== undefined) {
        return hit;
      }
    }
    throw new Error(`unknown external buffer "${uri}" (drop its .bin file alongside the .gltf)`);
  };
}

function safeDecode(uri: string): string {
  try {
    return decodeURIComponent(uri);
  } catch {
    return uri;
  }
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
