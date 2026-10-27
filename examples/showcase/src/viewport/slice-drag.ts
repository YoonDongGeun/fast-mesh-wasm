// Dragging an MPR slice quad through the main scene: the pointer ray
// meets a drag plane (through the grab point, facing the camera while it
// contains the quad normal), and the hit's travel along the normal
// becomes slice steps.
import type { Vec3 } from "../load/mesh-data";

const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const norm = (v: readonly number[]): Vec3 => {
  const len = Math.sqrt(dot(v, v));
  return len === 0 ? [0, 0, 0] : [(v[0] / len) as number, (v[1] / len) as number, (v[2] / len) as number];
};

/**
 * The drag plane's normal: the view direction with the quad normal taken
 * out, so the plane faces the camera while it contains the normal. Null
 * looking straight down the normal (a drag there cannot move the slice).
 */
export function dragPlaneNormal(quadNormal: Vec3, viewDir: Vec3): Vec3 | null {
  const n = norm(quadNormal);
  const v = norm(viewDir);
  const along = dot(v, n);
  const m: Vec3 = [v[0] - along * n[0], v[1] - along * n[1], v[2] - along * n[2]];
  return dot(m, m) < 1e-12 ? null : norm(m);
}

/** Where the ray meets the plane, or null when parallel or behind. */
export function planeHit(
  origin: Vec3,
  dir: Vec3,
  planePoint: Vec3,
  planeNormal: Vec3,
): Vec3 | null {
  const denom = dot(dir, planeNormal);
  if (Math.abs(denom) < 1e-9) {
    return null;
  }
  const t = dot([planePoint[0] - origin[0], planePoint[1] - origin[1], planePoint[2] - origin[2]], planeNormal) / denom;
  if (t < 0) {
    return null;
  }
  return [origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]];
}

/** The hit's travel from the grab point along the normal, in spacings. */
export function sliceSteps(grab: Vec3, now: Vec3, quadNormal: Vec3, spacing: number): number {
  if (!(spacing > 0)) {
    return 0;
  }
  const n = norm(quadNormal);
  return dot([now[0] - grab[0], now[1] - grab[1], now[2] - grab[2]], n) / spacing;
}

/** The float index as a slice of a `count` stack. */
export function clampSlice(value: number, count: number): number {
  if (!(count > 0)) {
    return 0;
  }
  return Math.min(count - 1, Math.max(0, Math.round(value)));
}
