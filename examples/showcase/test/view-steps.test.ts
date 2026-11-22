// The navigation cube's arrow steps (#255): exact quarter turns about the
// screen's axes and the view direction, so the view lands on axis views.
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import type { CameraPreset } from "../src/app/state/app-state";
import { presetFrame } from "../src/features/view/ux/camera";
import { VIEW_CUBE_CORNERS } from "../src/features/view/ux/navigation-hud";
import {
  frameFromQuaternion,
  ORBIT_FINE_STEP_DEG,
  ORBIT_STEP_DEG,
  type OrbitDirection,
  orbitStep,
  quaternionFromFrame,
  rollStep,
  screenRight,
  snapUnit,
  type Vec3,
  type ViewFrame,
} from "../src/features/view/ux/view-steps";

const PRESETS: CameraPreset[] = ["front", "back", "left", "right", "top", "bottom"];
const DIRECTIONS: OrbitDirection[] = ["up", "down", "left", "right"];

/** An axis vector: exactly one component is -1 or 1, the rest exactly 0 (never -0). */
function isAxis(v: Vec3): boolean {
  const nonZero = v.filter((c) => c !== 0);
  return nonZero.length === 1 && Math.abs(nonZero[0]) === 1 && v.every((c) => !Object.is(c, -0));
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

describe("orbitStep", () => {
  it("uses a quarter turn, and 15 degrees as the fine step", () => {
    expect(ORBIT_STEP_DEG).toBe(90);
    expect(ORBIT_FINE_STEP_DEG).toBe(15);
  });

  it("lands_on_an_axis_view_from_every_axis_view_and_direction", () => {
    for (const preset of PRESETS) {
      for (const direction of DIRECTIONS) {
        const next = orbitStep(presetFrame(preset), direction);
        expect(isAxis(next.dir), `${preset} ${direction} dir ${next.dir}`).toBe(true);
        expect(isAxis(next.up), `${preset} ${direction} up ${next.up}`).toBe(true);
        expect(dot(next.dir, next.up)).toBe(0);
      }
    }
  });

  it("brings_the_neighbouring_face_to_the_front_the_way_the_arrow_points", () => {
    const front = presetFrame("front");
    expect(orbitStep(front, "right")).toEqual(presetFrame("right"));
    expect(orbitStep(front, "left")).toEqual(presetFrame("left"));
    expect(orbitStep(front, "up")).toEqual(presetFrame("top"));
    expect(orbitStep(front, "down")).toEqual(presetFrame("bottom"));
    // From the top view (looking down, up -Z), "down" comes back to the front.
    expect(orbitStep(presetFrame("top"), "down")).toEqual(front);
  });

  it("returns_exactly_after_four_presses", () => {
    for (const preset of PRESETS) {
      for (const direction of DIRECTIONS) {
        let f: ViewFrame = presetFrame(preset);
        for (let i = 0; i < 4; i++) {
          f = orbitStep(f, direction);
          expect(isAxis(f.dir) && isAxis(f.up)).toBe(true);
        }
        expect(f).toEqual(presetFrame(preset));
      }
    }
  });

  it("six_fine_presses_make_one_quarter_turn_exactly", () => {
    for (const direction of DIRECTIONS) {
      let f: ViewFrame = presetFrame("front");
      for (let i = 0; i < 6; i++) {
        f = orbitStep(f, direction, ORBIT_FINE_STEP_DEG);
      }
      expect(f).toEqual(orbitStep(presetFrame("front"), direction));
    }
  });

  it("keeps_a_corner_view_square_and_returns_to_it_after_four_presses", () => {
    const corner = VIEW_CUBE_CORNERS[0];
    const start: ViewFrame = { dir: corner.dir, up: corner.up };
    let f = start;
    for (let i = 0; i < 4; i++) {
      f = orbitStep(f, "right");
      expect(Math.hypot(...f.dir)).toBeCloseTo(1, 12);
      expect(Math.abs(dot(f.dir, f.up))).toBeLessThan(1e-12);
    }
    for (let i = 0; i < 3; i++) {
      expect(f.dir[i]).toBeCloseTo(start.dir[i], 12);
      expect(f.up[i]).toBeCloseTo(start.up[i], 12);
    }
  });
});

describe("rollStep", () => {
  it("turns_the_picture_clockwise_a_quarter_about_the_view_direction", () => {
    const front = presetFrame("front");
    const cw = rollStep(front, "cw");
    expect(cw).toEqual({ dir: [0, 0, 1], up: [-1, 0, 0] });
    // What was at the top of the screen (world +Y) is now on its right.
    expect(screenRight(cw)).toEqual(front.up);
    expect(rollStep(front, "ccw")).toEqual({ dir: [0, 0, 1], up: [1, 0, 0] });
  });

  it("lands_on_axis_values_and_undoes_itself", () => {
    for (const preset of PRESETS) {
      const start = presetFrame(preset);
      let f: ViewFrame = start;
      for (let i = 0; i < 4; i++) {
        f = rollStep(f, "cw");
        expect(f.dir).toEqual(start.dir);
        expect(isAxis(f.up)).toBe(true);
      }
      expect(f).toEqual(start);
      expect(rollStep(rollStep(start, "cw"), "ccw")).toEqual(start);
    }
  });
});

describe("frames and camera orientations", () => {
  it("reads_a_looked_at_camera_as_its_exact_axis_frame", () => {
    for (const preset of PRESETS) {
      const { dir, up } = presetFrame(preset);
      const cam = new THREE.PerspectiveCamera();
      cam.position.set(dir[0] * 12.5 + 0.3, dir[1] * 12.5 - 0.7, dir[2] * 12.5 + 2.1);
      cam.up.set(up[0], up[1], up[2]);
      cam.lookAt(0.3, -0.7, 2.1);
      expect(frameFromQuaternion(cam.quaternion)).toEqual({ dir, up });
    }
  });

  it("round_trips_a_frame_through_a_quaternion", () => {
    for (const preset of PRESETS) {
      const f = presetFrame(preset);
      expect(frameFromQuaternion(quaternionFromFrame(f))).toEqual(f);
    }
    const corner = VIEW_CUBE_CORNERS[3];
    const back = frameFromQuaternion(quaternionFromFrame({ dir: corner.dir, up: corner.up }));
    for (let i = 0; i < 3; i++) {
      expect(back.dir[i]).toBeCloseTo(corner.dir[i], 12);
      expect(back.up[i]).toBeCloseTo(corner.up[i], 12);
    }
  });

  it("snaps_near_axis_values_and_never_returns_negative_zero", () => {
    expect(snapUnit([1e-12, -1 + 1e-13, -0])).toEqual([0, -1, 0]);
    expect(snapUnit([-0, 0, -2]).every((c) => !Object.is(c, -0))).toBe(true);
    expect(snapUnit([0.6, 0.8, 0])).toEqual([0.6, 0.8, 0]);
  });
});
