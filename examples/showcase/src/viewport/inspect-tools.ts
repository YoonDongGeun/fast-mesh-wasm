// Inspection math: the ruler, vertex snapping, section dragging and the
// clip box. Pure (no three, no DOM), so vitest covers it; the viewport
// layers and the inspect panel wire it up.
import type { Vec3 } from "../load/mesh-data";

/** The distance between two world points, for the ruler. */
export function rulerDistance(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

/**
 * The hit triangle's corner nearest to the hit: the largest barycentric
 * weight (ties go to the lowest index, like the hover tip). Snaps the
 * ruler and the query to vertices.
 */
export function snapSlot(barycentric: readonly [number, number, number]): 0 | 1 | 2 {
  const [u, v, w] = barycentric;
  return u >= v && u >= w ? 0 : v >= w ? 1 : 2;
}

/** The `slot`th vertex of the local positions. */
export function localVertex(positions: ArrayLike<number>, slot: number): Vec3 {
  return [positions[3 * slot], positions[3 * slot + 1], positions[3 * slot + 2]];
}

/**
 * The local point in world coordinates through the mesh's column-major 4x4
 * (`scene.matrix4`): translation, rotation and scale (#103) all apply, so
 * the ruler and the query follow placements. The ruler stores world points,
 * so undoing a placement move leaves its endpoints where they were.
 */
export function worldFromMatrix(m: ArrayLike<number>, p: readonly [number, number, number]): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}

/** The pointer's travel from the grab point along the (unit) section normal, in world units. */
export function normalTravel(
  grab: readonly [number, number, number],
  now: readonly [number, number, number],
  normal: readonly [number, number, number],
): number {
  return (now[0] - grab[0]) * normal[0] + (now[1] - grab[1]) * normal[1] + (now[2] - grab[2]) * normal[2];
}

/** One clip-box face: keeps the points with `normal . p + constant >= 0`. */
export interface BoxFace {
  normal: Vec3;
  constant: number;
}

/**
 * The six faces of the axis-aligned clip box: each face keeps its inside
 * half-space, so together they keep the box's interior. Three.js clips
 * away a plane's negative side, which is what these are.
 */
export function boxFaces(min: Vec3, max: Vec3): BoxFace[] {
  return [
    { normal: [1, 0, 0], constant: -min[0] },
    { normal: [-1, 0, 0], constant: max[0] },
    { normal: [0, 1, 0], constant: -min[1] },
    { normal: [0, -1, 0], constant: max[1] },
    { normal: [0, 0, 1], constant: -min[2] },
    { normal: [0, 0, -1], constant: max[2] },
  ];
}

/** Whether the world point is inside the box (on its faces counts as in). */
export function pointInBox(p: readonly [number, number, number], min: Vec3, max: Vec3): boolean {
  return (
    p[0] >= min[0] && p[0] <= max[0] && p[1] >= min[1] && p[1] <= max[1] && p[2] >= min[2] && p[2] <= max[2]
  );
}

/** A world coordinate with three decimals, for the ruler and query readouts. */
export function formatCoord(v: number): string {
  return Number(v.toPrecision(4)).toString();
}
