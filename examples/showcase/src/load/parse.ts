// Picks the parser by file extension.
import type { MeshData } from "./mesh-data";
import { parseObj } from "./obj";
import { parsePly } from "./ply";
import { parseStl } from "./stl";

/** The extensions the app reads, for the file picker's `accept`. */
export const MESH_EXTENSIONS = [".stl", ".obj", ".ply", ".dcm", ".dicom"];

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
    case ".dcm":
    case ".dicom":
      // The loader stacks series before parsing; a lone file here missed it.
      throw new Error(`${name}: DICOM loads as a series (drop the whole series, one file per slice)`);
    default:
      throw new Error(
        `cannot read ${ext === "" ? "files without an extension" : `${ext} files`} (STL, OBJ, PLY or DICOM)`,
      );
  }
}
