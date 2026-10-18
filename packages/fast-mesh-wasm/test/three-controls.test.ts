// attachTransformControls in Node: a mocked TransformControls (an
// EventDispatcher with `object` and `dragging`) drives scene drags.
import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FastMeshError, type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import {
  attachTransformControls,
  type BoundMesh,
  bindMesh,
  type TransformControlsLike,
} from "../src/three/index.js";

const QUARTER_Z = [0, -1, 0, 1, 0, 0, 0, 0, 1];

/** A box of 8 vertices spanning `min` to `min + size`, so its origin (the box centre) is not zero. */
function box(min: Vec3, size: Vec3): { positions: Float64Array; indices: Uint32Array } {
  const positions: number[] = [];
  for (let k = 0; k < 8; k++) {
    positions.push(
      min[0] + (k & 1) * size[0],
      min[1] + ((k >> 1) & 1) * size[1],
      min[2] + ((k >> 2) & 1) * size[2],
    );
  }
  const indices = [
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
    2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
  ];
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

class MockControls extends THREE.EventDispatcher<any> implements TransformControlsLike {
  object?: THREE.Object3D;
  dragging = false;
  setDragging(value: boolean) {
    this.dragging = value;
    this.dispatchEvent({ type: "dragging-changed", value });
  }
  /** What TransformControls does on a pointer move. */
  moveObject(position: Vec3, quaternion: THREE.Quaternion = new THREE.Quaternion()) {
    this.object!.position.set(...position);
    this.object!.quaternion.copy(quaternion);
    this.dispatchEvent({ type: "objectChange" });
  }
}

function codeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected a throw");
}

let scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes) {
    s.free();
  }
  scenes = [];
  vi.restoreAllMocks();
});

/**
 * A scene with a box mesh (origin away from zero) at a rotated placement and
 * no history, bound and attached to mock controls with a hand-run scheduler.
 */
function setup() {
  const s = new Scene();
  scenes.push(s);
  const g = box([2, -1, 0.5], [1, 3, 2]);
  const m = s.addMesh(g.positions, g.indices);
  s.setPlacement(m, { rotation: QUARTER_Z, translation: [1, 2, 3] });
  s.clearHistory();
  const bound = bindMesh(s, m);
  s.commitFrame();
  const controls = new MockControls();
  controls.object = bound.object;
  const frames: (() => void)[] = [];
  const detach = attachTransformControls(controls, s, bound, { schedule: (f) => frames.push(f) });
  /** Runs the scheduled frame callbacks. */
  const runFrames = () => {
    for (const f of frames.splice(0)) {
      f();
    }
  };
  return { s, m, bound, controls, frames, runFrames, detach };
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-9) {
  expect(actual.length).toBe(expected.length);
  for (let k = 0; k < expected.length; k++) {
    expect(Math.abs(actual[k] - expected[k])).toBeLessThanOrEqual(tolerance);
  }
}

/** The object's pose, composed, equals the scene's matrix. */
function expectPoseMatchesScene(s: Scene, m: MeshHandle, b: BoundMesh) {
  const composed = new THREE.Matrix4().compose(b.object.position, b.object.quaternion, b.object.scale);
  expectClose(composed.elements, s.matrix4(m));
  expectClose(b.object.matrix.elements, s.matrix4(m));
}

const quat = (axis: Vec3, angle: number) =>
  new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(...axis).normalize(), angle);

describe("attachTransformControls", () => {
  it("attach_syncs_object_pose", () => {
    const { s, m, bound } = setup();
    // TransformControls reads position/quaternion when a drag starts.
    expectPoseMatchesScene(s, m, bound);
    expect(bound.object.position.length()).toBeGreaterThan(0);
  });

  it("drag_maps_to_one_undo_step", () => {
    const { s, m, bound, controls, runFrames } = setup();
    const start = s.placement(m);
    controls.setDragging(true);
    controls.moveObject([4, 5, 6], quat([0, 0, 1], 0.3));
    runFrames();
    s.commitFrame();
    controls.moveObject([5, 5, 6], quat([1, 1, 0], 0.7));
    runFrames();
    s.commitFrame();
    const finalQ = quat([0, 1, 1], -1.1);
    controls.moveObject([6, 7, 8], finalQ);
    controls.setDragging(false);
    // The pose is the placement's matrix, right away and after the frame.
    expectClose(bound.object.position.toArray(), [6, 7, 8]);
    expectPoseMatchesScene(s, m, bound);
    s.commitFrame();
    expectPoseMatchesScene(s, m, bound);
    expectClose(bound.object.quaternion.toArray(), finalQ.toArray());
    // One undo step back to the start.
    expect(s.canUndo()).toBe(true);
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expectClose(s.placement(m).rotation, start.rotation);
    expectClose(s.placement(m).translation, start.translation);
    // Undo moves the object through the change stream, pose included.
    s.commitFrame();
    expectPoseMatchesScene(s, m, bound);
    expect(s.redo()).toBe(true);
    s.commitFrame();
    expectClose(bound.object.position.toArray(), [6, 7, 8]);
  });

  it("moves_are_coalesced_per_frame", () => {
    const { s, bound, controls, frames, runFrames } = setup();
    const move = vi.spyOn(s, "setPlacement");
    controls.setDragging(true);
    for (let k = 1; k <= 10; k++) {
      controls.moveObject([k, 0, 0]);
      // The object shows the gizmo's pose at once, before the scene has it.
      expect(bound.object.matrix.elements[12]).toBe(k);
    }
    expect(move).not.toHaveBeenCalled();
    expect(frames.length).toBe(1);
    runFrames();
    expect(move).toHaveBeenCalledTimes(1);
    expect(frames.length).toBe(0);

    // A frame commit while the scene lags the gizmo does not snap it back.
    controls.moveObject([20, 0, 0]);
    s.commitFrame();
    expect(bound.object.matrix.elements[12]).toBe(20);
    expect(bound.object.position.x).toBe(20);
    // Letting go flushes the pending pose; the stale frame callback is a no-op.
    controls.setDragging(false);
    expect(move).toHaveBeenCalledTimes(2);
    runFrames();
    expect(move).toHaveBeenCalledTimes(2);
    expect(bound.object.matrix.elements[12]).toBeCloseTo(20, 9);
  });

  it("undo_mid_transform_drag", () => {
    const { s, m, bound, controls, runFrames } = setup();
    const start = s.matrix4(m);
    controls.setDragging(true);
    controls.moveObject([4, 5, 6]);
    runFrames();
    s.commitFrame();
    controls.moveObject([7, 5, 6]);
    // Ctrl+Z while dragging: the drag ends first, then is undone whole.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    // The next gizmo event already shows the restored placement.
    controls.moveObject([8, 5, 6]);
    expectClose(bound.object.matrix.elements, start);
    runFrames(); // the pending pose is dropped
    s.commitFrame();
    // The object follows the restored placement through the change stream.
    expectClose(bound.object.matrix.elements, start);
    expectPoseMatchesScene(s, m, bound);
    // The rest of the gesture is ignored.
    const move = vi.spyOn(s, "setPlacement");
    controls.moveObject([9, 9, 9]);
    runFrames();
    expect(move).not.toHaveBeenCalled();
    expectClose(bound.object.matrix.elements, start);
    controls.setDragging(false);
    expectPoseMatchesScene(s, m, bound);
    expect(s.canUndo()).toBe(false);
    expect(s.canRedo()).toBe(true);
    // A new drag works.
    controls.setDragging(true);
    controls.moveObject([1, 1, 1]);
    controls.setDragging(false);
    expect(s.canUndo()).toBe(true);
    expectClose(bound.object.position.toArray(), [1, 1, 1]);
  });

  it("rotated_mesh_with_origin_round_trips", () => {
    const { s, m, bound, controls } = setup();
    controls.setDragging(true);
    const q = quat([0.3, -0.5, 0.8], 2.2);
    controls.moveObject([-3.25, 1.5, 7.125], q);
    controls.setDragging(false);
    // Object pose -> placement -> matrix4 reproduces the object matrix.
    const at = new THREE.Vector3(-3.25, 1.5, 7.125);
    const pose = new THREE.Matrix4().compose(at, q, new THREE.Vector3(1, 1, 1));
    expectClose(s.matrix4(m), pose.elements);
    // The placement's rotation is the quaternion's.
    const r = new THREE.Matrix4().makeRotationFromQuaternion(q).elements;
    expectClose(s.placement(m).rotation, [r[0], r[4], r[8], r[1], r[5], r[9], r[2], r[6], r[10]]);
    expectPoseMatchesScene(s, m, bound);
  });

  it("scale_mode_is_ignored_and_other_objects_are_left_alone", () => {
    const { s, m, bound, controls } = setup();
    controls.setDragging(true);
    bound.object.scale.set(2, 2, 2);
    controls.moveObject([1, 2, 3]);
    controls.setDragging(false);
    expect(bound.object.scale.toArray()).toEqual([1, 1, 1]);
    expectPoseMatchesScene(s, m, bound);

    // Controls on another object: no drag.
    s.clearHistory();
    const drag = vi.spyOn(s, "drag");
    controls.object = new THREE.Object3D();
    controls.setDragging(true);
    controls.moveObject([5, 5, 5]);
    controls.setDragging(false);
    expect(drag).not.toHaveBeenCalled();
    expect(s.canUndo()).toBe(false);
  });

  it("detach_ends_a_drag_and_stops_listening", () => {
    const { s, m, bound, controls, detach, runFrames } = setup();
    controls.setDragging(true);
    controls.moveObject([3, 0, 0]);
    detach();
    detach(); // no-op
    // The drag was ended with the latest pose: one undo step.
    expectClose([s.matrix4(m)[12]], [3]);
    expectPoseMatchesScene(s, m, bound);
    expect(s.canUndo()).toBe(true);
    runFrames();
    const drag = vi.spyOn(s, "drag");
    controls.setDragging(true);
    controls.moveObject([9, 0, 0]);
    controls.setDragging(false);
    expect(drag).not.toHaveBeenCalled();
    expect(s.canUndo()).toBe(true);
    s.undo();
    expect(s.canUndo()).toBe(false);
  });

  it("remove_mesh_mid_drag", () => {
    const { s, m, controls, runFrames, detach } = setup();
    const g = box([0, 0, 0], [1, 1, 1]);
    const other = s.addMesh(g.positions, g.indices);
    controls.setDragging(true);
    controls.moveObject([4, 5, 6]);
    runFrames();
    controls.moveObject([5, 5, 6]);
    s.removeMesh(m);
    // The pending pose is dropped without a throw, before and after the frame.
    expect(() => runFrames()).not.toThrow();
    s.commitFrame();
    controls.moveObject([6, 5, 6]);
    expect(() => runFrames()).not.toThrow();
    expect(() => controls.setDragging(false)).not.toThrow();
    // No drag or step is left open: another mesh can be dragged.
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");
    const next = s.drag(other);
    next.move({ rotation: [0, 0, 0, 1], translation: [1, 0, 0] });
    next.end();
    // Later gestures are ignored, and detach does not throw.
    controls.setDragging(true);
    controls.moveObject([7, 7, 7]);
    controls.setDragging(false);
    expect(() => detach()).not.toThrow();
  });

  it("remove_mesh_mid_drag_before_the_frame_commit", () => {
    // Removal seen first by a failing move (no commitFrame in between).
    const { s, m, controls, runFrames, detach } = setup();
    controls.setDragging(true);
    controls.moveObject([4, 5, 6]);
    // Simulate a removal the scene did not report to the drag.
    vi.spyOn(s, "setPlacement").mockImplementation(() => {
      throw new FastMeshError("UNKNOWN_MESH", "gone");
    });
    expect(() => runFrames()).not.toThrow();
    vi.restoreAllMocks();
    expect(() => controls.setDragging(false)).not.toThrow();
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");
    expect(() => detach()).not.toThrow();
    void m;
  });

  it("gesture_that_cannot_start_a_drag_is_ignored", () => {
    const { s, m, bound, controls, detach } = setup();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const g = box([0, 0, 0], [1, 1, 1]);
    const other = s.addMesh(g.positions, g.indices);
    s.clearHistory();
    const busy = s.drag(other);
    // Another drag is active: the gesture is ignored, not thrown.
    expect(() => controls.setDragging(true)).not.toThrow();
    controls.moveObject([9, 9, 9]);
    controls.setDragging(false);
    // The gizmo's pose is put back where the scene has the mesh.
    expectPoseMatchesScene(s, m, bound);
    controls.setDragging(true);
    controls.setDragging(false);
    expect(warn).toHaveBeenCalledTimes(1);
    busy.end();
    expect(s.canUndo()).toBe(false);
    detach();
  });

  it("detach_removes_listeners_even_when_finish_throws", () => {
    const { s, controls, detach } = setup();
    const off = vi.spyOn(controls, "removeEventListener");
    controls.setDragging(true);
    controls.moveObject([1, 0, 0]);
    // Make the final move fail with an unexpected error.
    vi.spyOn(s, "setPlacement").mockImplementation(() => {
      throw new Error("boom");
    });
    expect(() => detach()).toThrow("boom");
    expect(off).toHaveBeenCalledTimes(2);
    // The drag was still ended.
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");
  });

  it("real_transform_controls_fire_dragging_changed", () => {
    // Three's TransformControls reports `dragging` through the same event.
    const { s, bound } = setup();
    const tc = new TransformControls(new THREE.PerspectiveCamera());
    tc.attach(bound.object);
    const detach = attachTransformControls(tc, s, bound, { schedule: (f) => f() });
    tc.dragging = true;
    bound.object.position.set(1, 1, 1);
    tc.dispatchEvent({ type: "objectChange" });
    tc.dragging = false;
    expectClose(bound.object.position.toArray(), [1, 1, 1]);
    expect(s.canUndo()).toBe(true);
    detach();
  });
});
