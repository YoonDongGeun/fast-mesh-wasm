// Messages between the page and the loader Worker (parser.worker.ts).
import type { Vec3 } from "./mesh-data";

export type LoaderRequest =
  /** Parse a file (read in the Worker: the page never touches its bytes). */
  | { kind: "file"; id: number; file: File }
  /** Stack DICOM files into their series and mesh each isosurface (`threshold` null takes Otsu's). */
  | { kind: "dicom"; id: number; files: File[]; threshold: number | null }
  /** Generate the demo meshes (spec §6.8); `big` makes A and B about a million triangles each. */
  | { kind: "demo"; id: number; big: boolean };

/** A mesh ready for `store.addMesh`; its arrays are transferred to the page. */
export interface LoadedMesh {
  name: string;
  positions: Float64Array;
  indices: Uint32Array;
  /** Its bounding box centre (the mesh origin, so the gizmo sits on it) and half diagonal. */
  center: Vec3;
  radius: number;
}

export type LoaderReply =
  | { kind: "parsing"; id: number }
  | { kind: "meshes"; id: number; meshes: LoadedMesh[] }
  | { kind: "error"; id: number; message: string };
