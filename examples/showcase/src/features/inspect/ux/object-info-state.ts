// The object info slice of the app state (#257): the info box's frame and
// whether it is open, the bounding box and normals overlays, and the angle
// measure's picked points. Display state only: none of it is undoable or
// persisted, like the ruler.
import type { Vec3 } from "../../../io/mesh-data";
import type { InfoSpace } from "./object-info";

/** What the normals overlay draws: nothing, one line per face, or one per vertex. */
export type NormalsMode = "off" | "face" | "vertex";

export interface ObjectInfoState {
  /** Whether the info box at the bottom of the left sidebar is expanded. */
  open: boolean;
  /** The frame the info box's bounding box numbers and the bounding box overlay use. */
  space: InfoSpace;
  /** Whether the bounding box overlay draws on the selected mesh. */
  showBox: boolean;
  /** The normals overlay on the selected mesh. */
  normals: NormalsMode;
  /** The normals' length as a fraction of the mesh's radius (half its box diagonal). */
  normalsScale: number;
  /** The angle measure's world points in click order: first arm, the vertex, second arm. */
  angle: readonly Vec3[];
}

export const DEFAULT_OBJECT_INFO: ObjectInfoState = {
  open: true,
  space: "world",
  showBox: false,
  normals: "off",
  normalsScale: 0.05,
  angle: [],
};

/** The normals length slider's range, as fractions of the mesh's radius. */
export const NORMALS_SCALE_RANGE = { min: 0.005, max: 0.3, step: 0.005 } as const;
