// Picks the parser by file extension. glTF/GLB assets can hold several meshes;
// `parseMeshFiles` names each, while `parseMeshFile` serves the single-mesh formats.
import { decodeDraco } from "fast-mesh-wasm/io";
import { decoderModule } from "./draco";
import type { MeshData } from "./mesh-data";
import { parseGltf, parseGltfAsync, type GltfParseOptions, type NamedMeshData } from "./gltf";
import { parseObj } from "./obj";
import { parsePly } from "./ply";
import { parseStl } from "./stl";
import { parse3mf } from "./threemf";
import { checkMesh } from "./mesh-data";

/** The extensions the app reads, for the file picker's `accept`. */
export const MESH_EXTENSIONS = [".stl", ".obj", ".ply", ".3mf", ".gltf", ".glb", ".drc", ".dcm", ".dicom"];

/** @throws for an unknown extension or a file its parser rejects. */
export function parseMeshFile(name: string, buffer: ArrayBuffer): MeshData {
  const ext = /\.[^./\\]+$/.exec(name)?.[0].toLowerCase() ?? "";
  switch (ext) {
    case ".stl":
      return parseStl(buffer);
    case ".obj":
      return parseObj(buffer);
    case ".ply":
      return parsePly(buffer);
    case ".3mf":
      throw new Error(`${name}: 3MF archives parse asynchronously, parse it with parseMeshFilesAsync`);
    case ".gltf":
    case ".glb":
      // A glTF asset can hold several meshes: it needs the plural entry.
      throw new Error(`${name}: glTF holds one mesh per node primitive, parse it with parseMeshFiles`);
    case ".drc":
      // Draco decodes through WASM: it needs the async entry.
      throw new Error(`${name}: Draco decodes asynchronously, parse it with parseMeshFilesAsync`);
    case ".dcm":
    case ".dicom":
      // The loader stacks series before parsing; a lone file here missed it.
      throw new Error(`${name}: DICOM loads as a series (drop the whole series, one file per slice)`);
    default:
      throw new Error(
        `cannot read ${ext === "" ? "files without an extension" : `${ext} files`} (STL, OBJ, PLY, 3MF, glTF/GLB, DRC or DICOM)`,
      );
  }
}

/**
 * Parses a mesh file into its meshes: one for STL/OBJ/PLY/DRC, one per glTF node
 * primitive for `.gltf`/`.glb`. The loader Worker maps each onto a `LoadedMesh`.
 *
 * @throws for an unknown extension or a file its parser rejects.
 */
export function parseMeshFiles(name: string, buffer: ArrayBuffer): NamedMeshData[] {
  const ext = /\.[^./\\]+$/.exec(name)?.[0].toLowerCase() ?? "";
  if (ext === ".gltf" || ext === ".glb") {
    return parseGltf(name, buffer);
  }
  return [{ name, data: parseMeshFile(name, buffer) }];
}

/**
 * Parses a mesh file like {@link parseMeshFiles}, and additionally decodes
 * `.3mf` containers, `.drc` files, and Draco-compressed (`KHR_draco_mesh_compression`)
 * glTF assets through the draco3d decoder, resolving `.gltf`/`.glb` external
 * `.bin` buffers from `options.buffers`.
 *
 * @throws for an unknown extension or a file its parser rejects.
 */
export async function parseMeshFilesAsync(
  name: string,
  buffer: ArrayBuffer,
  options?: GltfParseOptions,
): Promise<NamedMeshData[]> {
  const ext = /\.[^./\\]+$/.exec(name)?.[0].toLowerCase() ?? "";
  if (ext === ".3mf") {
    return parse3mf(name, buffer);
  }
  if (ext === ".drc") {
    return [{ name, data: await parseDrc(buffer) }];
  }
  if (ext === ".gltf" || ext === ".glb") {
    return parseGltfAsync(name, buffer, options);
  }
  return [{ name, data: parseMeshFile(name, buffer) }];
}

/**
 * Decodes a standalone Draco file (`.drc`) into world positions and faces.
 *
 * @throws When the bytes are not a Draco triangle mesh, or it holds no triangles.
 */
export async function parseDrc(buffer: ArrayBuffer): Promise<MeshData> {
  const { positions, indices } = await decodeDraco(buffer, { decoderModule: await decoderModule() });
  return checkMesh({ positions, indices }, "DRC");
}
