// Short camera moves for the navigation cube (#255): a preset, a cube
// direction or an arrow step eases from the camera's pose to its goal (the
// orientation turned along the shortest arc, the target moved, the distance
// and the orthographic height scaled), and its last frame lands on the goal
// exactly. Pure math on three.js objects: the viewport's frame loop drives it.
import * as THREE from "three";
import type { CameraPresetPose } from "./camera";
import { frameFromQuaternion, quaternionFromFrame, type ViewFrame } from "./view-steps";

/** How long a view change takes, in milliseconds. */
export const VIEW_TWEEN_MS = 240;

/** The move's length here: none when the system asks for reduced motion. */
export function viewTweenMs(): number {
  const reduced =
    typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  return reduced ? 0 : VIEW_TWEEN_MS;
}

/** One end of a move. */
export interface TweenEnd {
  target: THREE.Vector3;
  /** The camera's orientation (local +Z away from the target, +Y the screen's up). */
  quaternion: THREE.Quaternion;
  /** From the target to the camera, in world units. */
  distance: number;
  /** Half the view's height at the target, in world units (an orthographic camera's frustum, at zoom 1). */
  halfHeight: number;
}

export interface CameraTween {
  from: TweenEnd;
  to: TweenEnd;
  /** When it started, on the `performance.now()` clock. */
  start: number;
  /** How long it runs, in milliseconds. */
  duration: number;
}

/** Eases in and out (a cubic): 0 at 0, 1 at 1, flat at both ends. */
export function easeInOut(t: number): number {
  const c = Math.min(Math.max(t, 0), 1);
  return c < 0.5 ? 4 * c * c * c : 1 - Math.pow(-2 * c + 2, 3) / 2;
}

/** `a` to `b` at `t` on a log scale, so a long dolly is as smooth near as far. */
function lerpScale(a: number, b: number, t: number): number {
  return a > 0 && b > 0 ? a * Math.pow(b / a, t) : a + (b - a) * t;
}

/** Where `tween` is at `now`: eased, and `done` once its time has run. */
export function sampleTween(tween: CameraTween, now: number): TweenEnd & { done: boolean } {
  const raw = tween.duration > 0 ? (now - tween.start) / tween.duration : 1;
  if (raw >= 1) {
    return { ...tween.to, done: true };
  }
  const t = easeInOut(raw);
  return {
    target: tween.from.target.clone().lerp(tween.to.target, t),
    quaternion: tween.from.quaternion.clone().slerp(tween.to.quaternion, t),
    distance: lerpScale(tween.from.distance, tween.to.distance, t),
    halfHeight: lerpScale(tween.from.halfHeight, tween.to.halfHeight, t),
    done: false,
  };
}

/** Half the view's height at `distance`: the orthographic frustum's, else the perspective cone's. */
function viewHalfHeight(camera: THREE.Camera, distance: number): number {
  if (camera instanceof THREE.OrthographicCamera) {
    return Math.abs(camera.top - camera.bottom) / (2 * Math.max(camera.zoom, 1e-4));
  }
  const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 45;
  return Math.tan(THREE.MathUtils.degToRad(fov / 2)) * distance;
}

/** The camera's pose about `target`, as a move's start. */
export function tweenEndFromCamera(camera: THREE.Camera, target: THREE.Vector3): TweenEnd {
  const distance = camera.position.distanceTo(target);
  return {
    target: target.clone(),
    quaternion: camera.quaternion.clone(),
    distance,
    halfHeight: viewHalfHeight(camera, distance),
  };
}

/** A goal pose looking along `frame`, as a move's end. */
export function tweenEndFromPose(pose: CameraPresetPose, frame: ViewFrame): TweenEnd {
  return {
    target: new THREE.Vector3(pose.target[0], pose.target[1], pose.target[2]),
    quaternion: quaternionFromFrame(frame),
    distance: pose.distance,
    halfHeight: pose.halfHeight,
  };
}

/**
 * The camera's view about `target` as a pose and frame (snapped: within
 * 1e-9 of an axis is on it), for a step to start from. Its clip planes and
 * height follow the distance as a preset's do; the frame's, too.
 */
export function viewFromCamera(
  camera: THREE.Camera,
  target: THREE.Vector3,
): { pose: CameraPresetPose; frame: ViewFrame } {
  const frame = frameFromQuaternion(camera.quaternion);
  const distance = Math.max(camera.position.distanceTo(target), 1e-6);
  const t: [number, number, number] = [target.x, target.y, target.z];
  return {
    frame,
    pose: {
      position: [t[0] + frame.dir[0] * distance, t[1] + frame.dir[1] * distance, t[2] + frame.dir[2] * distance],
      target: t,
      up: [...frame.up],
      distance,
      near: Math.max(distance / 1000, 1e-4),
      far: Math.max(distance * 100, 100),
      halfHeight: viewHalfHeight(camera, distance),
    },
  };
}

/** `base` turned to look along `frame`: the same target, distance and framing. */
export function poseAlong(base: CameraPresetPose, frame: ViewFrame): CameraPresetPose {
  const [tx, ty, tz] = base.target;
  const d = base.distance;
  return {
    ...base,
    position: [tx + frame.dir[0] * d, ty + frame.dir[1] * d, tz + frame.dir[2] * d],
    target: [tx, ty, tz],
    up: [...frame.up],
  };
}

/**
 * Opens the clip planes wide enough for every frame of a move to `goal`
 * (its own land when it ends): nothing is clipped on the way.
 */
export function widenClipRange(camera: THREE.Camera, goal: CameraPresetPose): void {
  if (camera instanceof THREE.PerspectiveCamera) {
    camera.near = Math.min(camera.near, goal.near);
    camera.far = Math.max(camera.far, goal.far);
    camera.updateProjectionMatrix();
  } else if (camera instanceof THREE.OrthographicCamera) {
    const span = Math.max(Math.abs(camera.near), Math.abs(camera.far), goal.distance * 50, 500);
    camera.near = -span;
    camera.far = span;
    camera.updateProjectionMatrix();
  }
}

/** Puts `camera` at `at`: position, up, orientation, and an orthographic camera's height (zoom 1). */
export function applyTweenEnd(camera: THREE.Camera, at: TweenEnd, aspect: number): void {
  const back = new THREE.Vector3(0, 0, 1).applyQuaternion(at.quaternion);
  camera.position.copy(at.target).addScaledVector(back, at.distance);
  camera.up.set(0, 1, 0).applyQuaternion(at.quaternion);
  camera.lookAt(at.target);
  if (camera instanceof THREE.OrthographicCamera) {
    const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
    camera.top = at.halfHeight;
    camera.bottom = -at.halfHeight;
    camera.left = -at.halfHeight * safeAspect;
    camera.right = at.halfHeight * safeAspect;
    camera.zoom = 1;
    camera.updateProjectionMatrix();
  }
}
