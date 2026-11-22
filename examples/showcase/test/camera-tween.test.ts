// The navigation cube's camera moves (#255): eased in between, exact at the
// end, and an arrow step taken from a real camera's pose lands it on axis
// values (the path the viewport's camera controller takes).
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import type { CameraPreset } from "../src/app/state/app-state";
import { applyCameraPose, presetFrame, presetPose } from "../src/features/view/ux/camera";
import {
  applyTweenEnd,
  type CameraTween,
  easeInOut,
  poseAlong,
  sampleTween,
  tweenEndFromCamera,
  tweenEndFromPose,
  VIEW_TWEEN_MS,
  viewFromCamera,
  widenClipRange,
} from "../src/features/view/ux/camera-tween";
import { orbitStep, rollStep, type Vec3, type ViewFrame } from "../src/features/view/ux/view-steps";

const BOUNDS = { center: [0.35, -1.2, 2.75] as [number, number, number], radius: 3.1 };

function isAxis(v: Vec3): boolean {
  const nonZero = v.filter((c) => c !== 0);
  return nonZero.length === 1 && Math.abs(nonZero[0]) === 1;
}

/** A camera at a preset, as the controller lands it (the pose, then the trackball's lookAt). */
function cameraAt(preset: CameraPreset, camera: THREE.Camera = new THREE.PerspectiveCamera(45, 1.5, 0.01, 5000)) {
  const pose = presetPose(preset, BOUNDS);
  applyCameraPose(camera, pose, 1.5, camera instanceof THREE.OrthographicCamera ? "orthographic" : "perspective");
  const target = new THREE.Vector3(...pose.target);
  camera.lookAt(target);
  return { camera, target };
}

/** The view direction (target to camera) and up the camera ends on. */
function landed(camera: THREE.Camera, target: THREE.Vector3): ViewFrame {
  const d = camera.position.clone().sub(target).normalize();
  return { dir: [d.x, d.y, d.z], up: [camera.up.x, camera.up.y, camera.up.z] };
}

describe("easeInOut", () => {
  it("runs_from_0_to_1_and_never_back", () => {
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
    expect(easeInOut(0.5)).toBeCloseTo(0.5, 12);
    let last = 0;
    for (let i = 1; i <= 20; i++) {
      const v = easeInOut(i / 20);
      expect(v).toBeGreaterThanOrEqual(last);
      last = v;
    }
    expect(easeInOut(-1)).toBe(0);
    expect(easeInOut(2)).toBe(1);
  });
});

describe("sampleTween", () => {
  const { camera, target } = cameraAt("front");
  const goal = poseAlong(viewFromCamera(camera, target).pose, presetFrame("right"));
  const tween: CameraTween = {
    from: tweenEndFromCamera(camera, target),
    to: tweenEndFromPose({ ...goal, distance: goal.distance * 4 }, presetFrame("right")),
    start: 1000,
    duration: VIEW_TWEEN_MS,
  };

  it("starts_where_the_camera_is_and_ends_on_the_goal", () => {
    const first = sampleTween(tween, 1000);
    expect(first.done).toBe(false);
    expect(first.quaternion.angleTo(camera.quaternion)).toBeLessThan(1e-9);
    expect(first.distance).toBeCloseTo(tween.from.distance, 9);
    const last = sampleTween(tween, 1000 + VIEW_TWEEN_MS);
    expect(last.done).toBe(true);
    expect(last.distance).toBe(tween.to.distance);
    expect(last.quaternion.equals(tween.to.quaternion)).toBe(true);
  });

  it("turns_half_way_and_dollies_on_a_log_scale_at_the_middle", () => {
    const mid = sampleTween(tween, 1000 + VIEW_TWEEN_MS / 2);
    expect(THREE.MathUtils.radToDeg(mid.quaternion.angleTo(tween.from.quaternion))).toBeCloseTo(45, 6);
    expect(mid.distance).toBeCloseTo(Math.sqrt(tween.from.distance * tween.to.distance), 9);
  });

  it("is_done_at_once_without_a_duration", () => {
    expect(sampleTween({ ...tween, duration: 0 }, 0).done).toBe(true);
  });
});

describe("arrow steps on a camera", () => {
  it("land_the_camera_up_and_direction_on_axis_values_press_after_press", () => {
    for (const make of [
      () => new THREE.PerspectiveCamera(45, 1.5, 0.01, 5000),
      () => new THREE.OrthographicCamera(-3, 3, 2, -2, -500, 500),
    ]) {
      const { camera, target } = cameraAt("front", make());
      const presses: ((f: ViewFrame) => ViewFrame)[] = [
        (f) => orbitStep(f, "right"),
        (f) => orbitStep(f, "up"),
        (f) => rollStep(f, "cw"),
        (f) => orbitStep(f, "left"),
        (f) => orbitStep(f, "down"),
        (f) => rollStep(f, "ccw"),
        (f) => orbitStep(f, "up"),
      ];
      const distance = camera.position.distanceTo(target);
      for (const press of presses) {
        const base = viewFromCamera(camera, target);
        const frame = press(base.frame);
        const goal = poseAlong(base.pose, frame);
        applyCameraPose(camera, goal, 1.5, camera instanceof THREE.OrthographicCamera ? "orthographic" : "perspective");
        camera.lookAt(target);
        const at = landed(camera, target);
        expect(isAxis(at.dir), `dir ${at.dir}`).toBe(true);
        expect(isAxis(at.up), `up ${at.up}`).toBe(true);
        expect(at).toEqual(frame);
        expect(camera.position.distanceTo(target)).toBeCloseTo(distance, 9);
      }
    }
  });

  it("keep_an_orthographic_view_s_framing", () => {
    const { camera, target } = cameraAt("front", new THREE.OrthographicCamera(-3, 3, 2, -2, -500, 500));
    const ortho = camera as THREE.OrthographicCamera;
    ortho.zoom = 2;
    ortho.updateProjectionMatrix();
    const base = viewFromCamera(ortho, target);
    applyCameraPose(ortho, poseAlong(base.pose, orbitStep(base.frame, "right")), 1.5, "orthographic");
    // Same visible height, now at zoom 1.
    expect(ortho.zoom).toBe(1);
    expect(ortho.top - ortho.bottom).toBeCloseTo(presetPose("front", BOUNDS).halfHeight, 9);
  });
});

describe("tween frames on a camera", () => {
  it("place_the_camera_on_the_sample_and_open_the_clip_planes_for_the_way", () => {
    const { camera, target } = cameraAt("front");
    const persp = camera as THREE.PerspectiveCamera;
    const goal = { ...presetPose("top", BOUNDS), near: 1e-3, far: 9e4 };
    widenClipRange(persp, goal);
    expect(persp.near).toBe(Math.min(presetPose("front", BOUNDS).near, 1e-3));
    expect(persp.far).toBe(9e4);
    const end = tweenEndFromPose(goal, presetFrame("top"));
    applyTweenEnd(persp, end, 1.5);
    const at = landed(persp, target);
    expect(at.dir[1]).toBeCloseTo(1, 12);
    expect(at.up[2]).toBeCloseTo(-1, 12);

    const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1);
    ortho.zoom = 3;
    applyTweenEnd(ortho, { ...end, halfHeight: 2.5 }, 2);
    expect([ortho.top, ortho.bottom, ortho.left, ortho.right, ortho.zoom]).toEqual([2.5, -2.5, -5, 5, 1]);
  });
});
