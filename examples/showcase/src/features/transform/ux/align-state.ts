// The align tools' part of the app state (#253), and the commands that press
// the align panel's buttons, apart from the math in align-tools.ts: the app
// state and the command palette need them before the transform chunk is here.
import type { MeshHandle, Vec3 } from "fast-mesh-wasm";

/** Which mesh of a pair a landmark is on. */
export type LandmarkSide = "moving" | "fixed";

/** The landmark pairs picked so far, each point in its mesh's local render frame. */
export interface LandmarkSet {
  moving: MeshHandle;
  fixed: MeshHandle;
  movingPoints: readonly Vec3[];
  fixedPoints: readonly Vec3[];
  /** Whether canvas clicks add points (Esc stops it and keeps the points). */
  picking: boolean;
}

/** A detected symmetry plane, kept in its mesh's local render frame. */
export interface SymmetryPlaneShown {
  mesh: MeshHandle;
  /** A point on the plane (local). */
  point: Vec3;
  /** The plane's unit normal (local). */
  normal: Vec3;
  /** The RMS distance from the reflected surface to the mesh (mm) when it was detected. */
  errorRms: number;
  /** 1 for an exact mirror image, toward 0 for none. */
  score: number;
}

/** The align tools' app state: the landmark pairs and the symmetry plane on show. */
export interface AlignToolState {
  landmarks: LandmarkSet | null;
  symmetry: SymmetryPlaneShown | null;
}

export const DEFAULT_ALIGN_TOOL: AlignToolState = { landmarks: null, symmetry: null };

/** The align commands the command palette offers, by the panel button they press. */
export const ALIGN_COMMANDS: readonly { action: string; label: string; keywords: string; meshes: 1 | 2 }[] = [
  {
    action: "landmarks-pick",
    label: "Align: Pick Landmark Pairs",
    keywords: "align landmark pairs points pick correspond register",
    meshes: 2,
  },
  {
    action: "multistage-run",
    label: "Align: Coarse + ICP",
    keywords: "align register coarse icp multistage bounding box principal axes",
    meshes: 2,
  },
  {
    action: "symmetry-detect",
    label: "Align: Detect Symmetry Plane",
    keywords: "align symmetry plane mirror reflect detect",
    meshes: 1,
  },
];
