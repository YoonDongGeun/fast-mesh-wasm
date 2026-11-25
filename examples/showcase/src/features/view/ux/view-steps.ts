// The navigation cube's arrow steps as exact math: the orbit arrows turn the
// view about the screen's up or right axis, and the roll arrows turn it about
// the viewing direction. A quarter turn uses exact cosines and sines, and
// every result snaps values within 1e-9 of -1, 0 or 1, so the step from an
// axis view lands exactly on the next axis view, however many presses.
import * as THREE from "three";

export type Vec3 = readonly [number, number, number];

/** An orbit arrow: the way the camera moves around the target, as seen on screen. */
export type OrbitDirection = "up" | "down" | "left" | "right";

/** A roll arrow: the way the picture turns on screen. */
export type RollDirection = "cw" | "ccw";

/**
 * The orbit arrows' step: a quarter turn, so a press from an axis view lands
 * on the next axis view and repeated presses stay on axis views.
 */
export const ORBIT_STEP_DEG = 90;

/** The orbit arrows' step with Shift held: six presses make one quarter turn. */
export const ORBIT_FINE_STEP_DEG = 15;

/** The roll arrows' step. */
export const ROLL_STEP_DEG = 90;

/**
 * A view's orientation: `dir` points from the orbit target to the camera
 * and `up` is the screen's up. Both are unit length and at right angles.
 */
export interface ViewFrame {
  dir: Vec3;
  up: Vec3;
}

/** How near a value must be to -1, 0 or 1 to be taken as exactly that. */
const SNAP_EPS = 1e-9;

function snapValue(v: number): number {
  const r = Math.round(v);
  return r >= -1 && r <= 1 && Math.abs(v - r) < SNAP_EPS ? r + 0 : v; // `+ 0` turns -0 into 0
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): [number, number, number] {
  // `+ 0` turns -0 into 0.
  return [a[1] * b[2] - a[2] * b[1] + 0, a[2] * b[0] - a[0] * b[2] + 0, a[0] * b[1] - a[1] * b[0] + 0];
}

function combine(a: Vec3, ka: number, b: Vec3, kb: number): [number, number, number] {
  return [a[0] * ka + b[0] * kb, a[1] * ka + b[1] * kb, a[2] * ka + b[2] * kb];
}

/**
 * `v` at unit length, with each component within 1e-9 of -1, 0 or 1 set to
 * it (then scaled back to unit length if anything else is left).
 */
export function snapUnit(v: Vec3): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  const s: [number, number, number] = [snapValue(v[0] / len), snapValue(v[1] / len), snapValue(v[2] / len)];
  const sLen = Math.hypot(s[0], s[1], s[2]);
  if (sLen === 1 || sLen === 0) {
    return s;
  }
  return [snapValue(s[0] / sLen), snapValue(s[1] / sLen), snapValue(s[2] / sLen)];
}

/** Cosine and sine of `deg`, exact at whole quarter turns. */
export function cosSinDeg(deg: number): [number, number] {
  const r = ((deg % 360) + 360) % 360;
  if (r === 0) return [1, 0];
  if (r === 90) return [0, 1];
  if (r === 180) return [-1, 0];
  if (r === 270) return [0, -1];
  const a = THREE.MathUtils.degToRad(deg);
  return [Math.cos(a), Math.sin(a)];
}

/** `frame` cleaned up: both axes snapped to unit length, `up` made square to `dir`. */
export function snapFrame(frame: ViewFrame): ViewFrame {
  const dir = snapUnit(frame.dir);
  const rest = combine(frame.up, 1, dir, -dot(frame.up, dir));
  if (Math.hypot(rest[0], rest[1], rest[2]) > 1e-6) {
    return { dir, up: snapUnit(rest) };
  }
  // `up` along `dir`: any square axis will do. World +Y, or for a view
  // straight down (up) the one the top (bottom) view uses.
  const ref: Vec3 = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [0, 0, dir[1] > 0 ? -1 : 1];
  return { dir, up: snapUnit(combine(ref, 1, dir, -dot(ref, dir))) };
}

/** The screen's right in world space: `up` x `dir`. */
export function screenRight(frame: ViewFrame): [number, number, number] {
  return cross(frame.up, frame.dir);
}

/**
 * Orbits the camera `deg` degrees about the target, the way an arrow
 * points: `right` brings the face on the right of the screen to the front,
 * `up` the face above. From the front view (+Z, up +Y), `right` lands on
 * the right view (+X, up +Y) and `up` on the top view (+Y, up -Z).
 */
export function orbitStep(frame: ViewFrame, direction: OrbitDirection, deg = ORBIT_STEP_DEG): ViewFrame {
  const f = snapFrame(frame);
  const [c, s0] = cosSinDeg(deg);
  const s = direction === "left" || direction === "down" ? -s0 : s0;
  if (direction === "left" || direction === "right") {
    const right = screenRight(f);
    return snapFrame({ dir: combine(f.dir, c, right, s), up: f.up });
  }
  return snapFrame({ dir: combine(f.dir, c, f.up, s), up: combine(f.up, c, f.dir, -s) });
}

/**
 * Rolls the view `deg` degrees about the viewing direction: `cw` turns the
 * picture clockwise on screen (what was at the top moves to the right).
 */
export function rollStep(frame: ViewFrame, direction: RollDirection, deg = ROLL_STEP_DEG): ViewFrame {
  const f = snapFrame(frame);
  const [c, s0] = cosSinDeg(deg);
  const s = direction === "cw" ? -s0 : s0;
  return snapFrame({ dir: f.dir, up: combine(f.up, c, screenRight(f), s) });
}

/** The frame a camera with `quaternion` looks along (its local +Z and +Y), snapped. */
export function frameFromQuaternion(quaternion: THREE.Quaternion): ViewFrame {
  const z = new THREE.Vector3(0, 0, 1).applyQuaternion(quaternion);
  const y = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion);
  return snapFrame({ dir: [z.x, z.y, z.z], up: [y.x, y.y, y.z] });
}

/** The camera orientation that looks along `frame` (local +Z = `dir`, +Y = `up`). */
export function quaternionFromFrame(frame: ViewFrame, out = new THREE.Quaternion()): THREE.Quaternion {
  const f = snapFrame(frame);
  const x = screenRight(f);
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(x[0], x[1], x[2]),
    new THREE.Vector3(f.up[0], f.up[1], f.up[2]),
    new THREE.Vector3(f.dir[0], f.dir[1], f.dir[2]),
  );
  return out.setFromRotationMatrix(m);
}
