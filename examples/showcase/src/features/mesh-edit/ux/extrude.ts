// Extrude's pure parts (#262): the directions its fields offer, the engine
// options they become, and the report lines. Lengths are in millimetres (the
// engine's world unit); text follows the unit setting.
import type {
  ExtrudeBaseOptions,
  ExtrudeBaseResult,
  ExtrudeFacesOptions,
  ExtrudeFacesResult,
} from "fast-mesh-wasm";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import type { SignedAxis } from "../../transform/ux/placement-math";

/** Which way faces extrude: along region or vertex normals, or a world axis. */
export type FaceDirection = "region" | "vertex" | "x" | "y" | "z";

export const FACE_DIRECTIONS: readonly (readonly [FaceDirection, string, string])[] = [
  ["region", "Normal", "Each region moves as one piece along its average normal"],
  ["vertex", "Per vertex", "Each vertex moves along its own average normal"],
  ["x", "X", "Along the world X axis (negative distance: the other way)"],
  ["y", "Y", "Along the world Y axis (negative distance: the other way)"],
  ["z", "Z", "Along the world Z axis (negative distance: the other way)"],
];

/** The directions a base extends in, and where the app starts: down with Z up. */
export const BASE_DIRECTIONS: readonly SignedAxis[] = ["-x", "+x", "-y", "+y", "-z", "+z"];
export const DEFAULT_BASE_DIRECTION: SignedAxis = "-z";

/** Where the base plane is: past the mesh's furthest point, or at a coordinate. */
export type BasePlaneMode = "below" | "at";

/** The unit vector of a signed axis. */
export function signedAxisVector(axis: SignedAxis): [number, number, number] {
  const v: [number, number, number] = [0, 0, 0];
  v["xyz".indexOf(axis[1])] = axis[0] === "-" ? -1 : 1;
  return v;
}

/** Whether Extrude Faces cannot run with these fields. */
export function extrudeFacesBlocked(selected: number, distance: number): boolean {
  return selected === 0 || !Number.isFinite(distance) || distance === 0;
}

/** The engine's options for the Extrude Faces fields. */
export function extrudeFacesOptions(fields: { distance: number; direction: FaceDirection }): ExtrudeFacesOptions {
  const { distance, direction } = fields;
  if (direction === "region" || direction === "vertex") {
    return { distance, direction };
  }
  return { distance, direction: signedAxisVector(`+${direction}`) };
}

/**
 * The engine's options for the Extrude Base fields. `at` is a world
 * coordinate along the axis; the engine measures a height along the opposite
 * of the direction.
 */
export function extrudeBaseOptions(fields: {
  direction: SignedAxis;
  plane: BasePlaneMode;
  offset: number;
  at: number;
  allLoops: boolean;
  refine: boolean;
}): ExtrudeBaseOptions {
  const { direction, plane, offset, at, allLoops, refine } = fields;
  return {
    direction: signedAxisVector(direction),
    ...(plane === "below" ? { offset } : { height: direction[0] === "-" ? at : -at + 0 }),
    loops: allLoops ? "all" : "largest",
    refine,
  };
}

const n = (count: number) => count.toLocaleString("en");
const plural = (count: number, one: string, many = `${one}s`) => `${n(count)} ${count === 1 ? one : many}`;

/** What to say about the crossing faces a result reports; nothing when there are none. */
export function crossingNote(crossing: Uint32Array | null): string {
  return crossing === null || crossing.length === 0
    ? ""
    : ` Warning: ${plural(crossing.length, "face")} touch or cross each other.`;
}

/** The report line of a finished Extrude Faces. */
export function describeExtrudeFaces(r: ExtrudeFacesResult, distance: number, units: LengthUnit): string {
  return (
    `${plural(r.capTriangles, "face")} in ${plural(r.regions, "region")} by ${formatLength(distance, units)}, ` +
    `${plural(r.wallTriangles.length, "wall triangle")}.${crossingNote(r.selfIntersections)}`
  );
}

/** The report line of a finished Extrude Base. */
export function describeExtrudeBase(r: ExtrudeBaseResult, direction: SignedAxis, units: LengthUnit): string {
  const at = direction[0] === "-" ? r.height : -r.height + 0;
  return (
    `${plural(r.loops.length, "loop")} to ${direction[1].toUpperCase()} = ${formatLength(at, units)}, ` +
    `${plural(r.wallTriangles, "wall")} and ${plural(r.capTriangles, "cap triangle")}.` +
    (r.flipped ? " The surface was turned to face outward." : "") +
    crossingNote(r.selfIntersections)
  );
}
