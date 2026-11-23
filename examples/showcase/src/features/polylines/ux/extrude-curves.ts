// Extrude (#268): planar closed curves into a solid. The curves that lie in
// one plane are extruded together, so a curve inside another is a hole in it
// (and one inside that hole an island), as the letters of the text tool: the
// same nesting, cap triangulation and walls (`text-mesh.ts`) build it, in
// the plane's own coordinates.
import { cleanContour } from "../../mesh-edit/ux/text-font";
import { extrudeLayout, isClosed } from "../../mesh-edit/ux/text-mesh";
import { crosses2, planarFit, type Plane, planarTolerance, toPlane } from "./planar";
import { curvesExtent, type PolylineCurve } from "./polyline";
import { simplifyCurve } from "./simplify";
import { centredSolid, type SolidMesh } from "./solid";

/** Where an extrusion goes from the curve's plane: one way (a negative depth: the other), or half each way. */
export type ExtrudeSide = "one" | "both";

/** The most points the outline of one extrusion may have: its caps are triangulated on the page. */
export const MAX_EXTRUDE_POINTS = 50_000;

/** Why curves cannot be extruded (a bent one's deviation in millimetres). */
export type ExtrudeProblem =
  | { reason: "open" | "straight" | "crossing" | "caps" }
  | { reason: "bent"; deviation: number }
  | { reason: "points"; points: number };

/** Curves that lie in one plane, and the plane. */
export interface PlanarGroup {
  plane: Plane;
  curves: PolylineCurve[];
}

/**
 * The closed curves grouped by the plane they lie in (parallel and at the
 * same height, within the planar tolerance), in order; or why one of them
 * cannot be extruded.
 */
export function planarGroups(curves: readonly PolylineCurve[]): PlanarGroup[] | ExtrudeProblem {
  const tolerance = planarTolerance(curves);
  const groups: PlanarGroup[] = [];
  for (const curve of curves) {
    if (!curve.closed) {
      return { reason: "open" };
    }
    const fit = planarFit([curve]);
    if ("problem" in fit) {
      return fit.problem === "bent" ? { reason: "bent", deviation: fit.deviation } : { reason: "straight" };
    }
    const group = groups.find(({ plane: { origin: o, normal: n } }) => {
      for (let i = 0; i + 2 < curve.points.length; i += 3) {
        const p = curve.points;
        if (Math.abs((p[i] - o[0]) * n[0] + (p[i + 1] - o[1]) * n[1] + (p[i + 2] - o[2]) * n[2]) > tolerance) {
          return false;
        }
      }
      return true;
    });
    if (group === undefined) {
      groups.push({ plane: fit.plane, curves: [curve] });
    } else {
      group.curves.push(curve);
    }
  }
  return groups;
}

/** Points a straight run passes through within this share of the outline's size are dropped from it. */
const STRAIGHT = 1e-6;

/**
 * The solid between the group's curves and the same curves `depth` along
 * their plane's normal (against it when negative), or about the plane with
 * `"both"`: closed, welded and wound outward; or why there is none. A hole
 * is a curve inside another.
 */
export function extrudeGroup(group: PlanarGroup, depth: number, side: ExtrudeSide): SolidMesh | ExtrudeProblem {
  const { plane } = group;
  const slack = STRAIGHT * curvesExtent(group.curves);
  const contours: number[][] = [];
  let points = 0;
  for (const curve of group.curves) {
    const xy = toPlane(simplifyCurve(curve, slack), plane);
    // As the mesh stores them, so that the caps are triangulated over the very points it has.
    const contour = cleanContour(Array.from(xy, (v) => Math.fround(v)));
    if (contour.length < 6) {
      return { reason: "straight" };
    }
    points += contour.length / 2;
    contours.push(contour);
  }
  if (points > MAX_EXTRUDE_POINTS) {
    return { reason: "points", points };
  }
  if (crosses2(contours.map((xy) => ({ xy, closed: true })))) {
    return { reason: "crossing" };
  }
  const height = Math.abs(depth);
  const z0 = side === "both" ? -height / 2 : Math.min(0, depth);
  const solid = extrudeLayout({ glyphs: [{ char: "", contours }], missing: [], box: [0, 0, 0, 0] }, height, z0);
  if (solid.indices.length === 0 || !isClosed(solid.indices)) {
    return { reason: "caps" };
  }
  const { origin: o, u, v, normal: n } = plane;
  const local = solid.positions;
  const world = new Float64Array(local.length);
  for (let i = 0; i < local.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      world[i + a] = o[a] + local[i] * u[a] + local[i + 1] * v[a] + local[i + 2] * n[a];
    }
  }
  return centredSolid(world, solid.indices);
}
