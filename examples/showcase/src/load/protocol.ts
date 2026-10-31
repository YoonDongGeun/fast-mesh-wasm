// Messages between the page and the loader Worker (parser.worker.ts).
import type { Vec3 } from "./mesh-data";
import type { PhantomKind } from "./phantom";
import type { SampleKind } from "./vtkjs";

export type LoaderRequest =
  /** Parse a file (read in the Worker: the page never touches its bytes). */
  | { kind: "file"; id: number; file: File }
  /** Stack DICOM files into their series and mesh each isosurface (`threshold` null takes Otsu's). */
  | { kind: "dicom"; id: number; files: File[]; threshold: number | null }
  /** Mesh a procedural DICOM phantom through the same extraction pipeline. */
  | { kind: "phantom"; id: number; phantom: PhantomKind }
  /** Mesh a bundled sample CT through the same extraction pipeline. */
  | { kind: "sample"; id: number; sample: SampleKind }
  /** Fetch a retained source volume (page cache miss). */
  | { kind: "volume"; id: number; volumeId: string }
  /** Drop a retained source volume (fire-and-forget: no reply). */
  | { kind: "dropVolume"; volumeId: string }
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
  /** The retained source volume's id (DICOM + phantoms), for the volume view. */
  volumeId: string | null;
  /** The suggested volume preset (a sample's descriptor hint), if any. */
  preset: string | null;
}

/** A retained source volume, fetched for the volume view (values transferred). */
export interface VolumePayload {
  volumeId: string;
  columns: number;
  rows: number;
  slices: number;
  spacing: [number, number, number];
  origin: [number, number, number];
  values: Float32Array;
  /** Procedural phantom SDF: map to pseudo-HU for the CT presets. */
  pseudoHu: boolean;
}

export type LoaderReply =
  | { kind: "parsing"; id: number }
  | { kind: "meshes"; id: number; meshes: LoadedMesh[] }
  | { kind: "volume"; id: number; volume: VolumePayload }
  | { kind: "error"; id: number; message: string };
