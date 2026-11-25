// The angle measure (#257): three clicks on vertices, the middle one the
// angle's vertex, measured in world coordinates like the ruler (so the points
// stay where they were picked through placement undo). The arc and the label
// anchor the viewport layer draws come from here too. Pure, so vitest covers it.
import type { Vec3 } from "../../../io/mesh-data";

/** The points after a click: appended, and a click after a finished angle starts a new one. */
export function nextAnglePoints(points: readonly Vec3[], picked: Vec3): Vec3[] {
  return points.length >= 3 ? [picked] : [...points, picked];
}

function sub(a: readonly number[], b: readonly number[]): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm(a: readonly number[]): number {
  return Math.hypot(a[0], a[1], a[2]);
}

/**
 * The angle at `vertex` between the arms to `a` and `c`, in degrees (0 to
 * 180), `null` when an arm has no length. `atan2` of the cross and dot
 * products stays exact near 0° and 180°, where `acos` loses digits.
 */
export function angleDegrees(a: readonly number[], vertex: readonly number[], c: readonly number[]): number | null {
  const u = sub(a, vertex);
  const v = sub(c, vertex);
  if (norm(u) === 0 || norm(v) === 0) {
    return null;
  }
  const cross: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  return (Math.atan2(norm(cross), dot) * 180) / Math.PI;
}

/** An angle readout: `42.5°`, one decimal. */
export function formatAngle(deg: number): string {
  return `${deg.toFixed(1)}°`;
}

/**
 * The arc between the arms, `segments + 1` points `radius` from the vertex
 * (spherical interpolation between the arm directions), flat `x, y, z`.
 * Empty when an arm has no length or the arms are opposite (no single arc).
 */
export function angleArc(
  a: readonly number[],
  vertex: readonly number[],
  c: readonly number[],
  radius: number,
  segments = 24,
): Float32Array {
  const u = sub(a, vertex);
  const v = sub(c, vertex);
  const lu = norm(u);
  const lv = norm(v);
  if (lu === 0 || lv === 0) {
    return new Float32Array(0);
  }
  const p: Vec3 = [u[0] / lu, u[1] / lu, u[2] / lu];
  const q: Vec3 = [v[0] / lv, v[1] / lv, v[2] / lv];
  const theta = Math.acos(Math.min(1, Math.max(-1, p[0] * q[0] + p[1] * q[1] + p[2] * q[2])));
  const s = Math.sin(theta);
  if (s < 1e-9 && theta > 1) {
    return new Float32Array(0);
  }
  const out = new Float32Array(3 * (segments + 1));
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    // Collinear arms (theta ~ 0): the arc is the shared direction.
    const wa = s < 1e-9 ? 1 - t : Math.sin((1 - t) * theta) / s;
    const wb = s < 1e-9 ? t : Math.sin(t * theta) / s;
    for (let k = 0; k < 3; k++) {
      out[3 * i + k] = vertex[k] + radius * (wa * p[k] + wb * q[k]);
    }
  }
  return out;
}

/** The arc's radius: a third of the shorter arm. */
export function arcRadius(a: readonly number[], vertex: readonly number[], c: readonly number[]): number {
  return Math.min(norm(sub(a, vertex)), norm(sub(c, vertex))) / 3;
}

/**
 * Where the angle's label sits: out along the bisector of the arms, past
 * the arc. On the first arm when the arms are opposite (no bisector).
 */
export function angleLabelAnchor(
  a: readonly number[],
  vertex: readonly number[],
  c: readonly number[],
  radius: number,
): Vec3 {
  const u = sub(a, vertex);
  const v = sub(c, vertex);
  const lu = norm(u) || 1;
  const lv = norm(v) || 1;
  let b: Vec3 = [u[0] / lu + v[0] / lv, u[1] / lu + v[1] / lv, u[2] / lu + v[2] / lv];
  const lb = norm(b);
  b = lb < 1e-9 ? [u[0] / lu, u[1] / lu, u[2] / lu] : [b[0] / lb, b[1] / lb, b[2] / lb];
  const d = radius * 1.6;
  return [vertex[0] + b[0] * d, vertex[1] + b[1] * d, vertex[2] + b[2] * d];
}
