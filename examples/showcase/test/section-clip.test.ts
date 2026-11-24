// The section plane's and clip box's panel side: the anchor through the
// placement, the axis and offset readout, the clip box fit and its slider
// bounds and edits.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import type { Item } from "../src/app/state/app-state";
import {
  clipBoxWithEnd,
  clipSliderBounds,
  fitWorldBox,
  sectionCenter,
  sectionReadout,
} from "../src/features/inspect/ux/section-clip";

// Column-major 4x4, like `scene.matrix4`.
const translate = (x: number, y: number, z: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
const sceneWith = (m: number[]) => ({ matrix4: () => m }) as unknown as Scene;
const removed = { matrix4: () => { throw new Error("gone"); } } as unknown as Scene;
const item = { mesh: 1 as unknown as MeshHandle, center: [1, 2, 3], radius: 4 } as unknown as Item;
const sectionAt = (origin: [number, number, number], normal: [number, number, number]) => ({
  handle: 7,
  plane: { origin, normal },
});

describe("sectionCenter", () => {
  it("maps_the_local_center_through_the_placement", () => {
    expect(sectionCenter(sceneWith(translate(10, 0, -1)), null, item)).toEqual([11, 2, 2]);
  });

  it("is_null_without_a_mesh_or_once_removed", () => {
    expect(sectionCenter(sceneWith(translate(0, 0, 0)), null, null)).toBeNull();
    expect(sectionCenter(removed, null, item)).toBeNull();
  });
});

describe("sectionReadout", () => {
  it("defaults_to_plus_z_with_no_offset", () => {
    const r = sectionReadout(sceneWith(translate(0, 0, 0)), null, item);
    expect(r.normal).toEqual([0, 0, 1]);
    expect(r.axis).toBe("z");
    expect(r.offset).toBe(0);
    expect(r.range).toBe(4);
  });

  it("reads_the_plane_offset_along_its_normal", () => {
    const r = sectionReadout(sceneWith(translate(0, 0, 0)), sectionAt([1, 5, 3], [0, 1, 0]), item);
    expect(r.axis).toBe("y");
    expect(r.offset).toBe(3);
  });

  it("picks_the_largest_normal_component_ties_to_x_then_y", () => {
    const scene = sceneWith(translate(0, 0, 0));
    expect(sectionReadout(scene, sectionAt([0, 0, 0], [-0.9, 0.1, 0.1]), item).axis).toBe("x");
    expect(sectionReadout(scene, sectionAt([0, 0, 0], [0.5, 0.5, 0]), item).axis).toBe("x");
    expect(sectionReadout(scene, sectionAt([0, 0, 0], [0, 0.5, 0.5]), item).axis).toBe("y");
  });

  it("keeps_a_positive_range_without_a_mesh", () => {
    expect(sectionReadout(sceneWith(translate(0, 0, 0)), null, null).range).toBe(1);
    const tiny = { ...item, radius: 0 } as Item;
    expect(sectionReadout(sceneWith(translate(0, 0, 0)), null, tiny).range).toBe(1e-6);
  });
});

describe("fitWorldBox", () => {
  it("keeps_the_middle_half_around_the_world_center", () => {
    expect(fitWorldBox(sceneWith(translate(10, 0, 0)), item)).toEqual({ min: [9, 0, 1], max: [13, 4, 5] });
  });

  it("stretches_with_the_placement_scale", () => {
    const scaled = [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    expect(fitWorldBox(sceneWith(scaled), item)).toEqual({ min: [-2, 0, 1], max: [6, 4, 5] });
  });

  it("is_null_once_removed", () => {
    expect(fitWorldBox(removed, item)).toBeNull();
  });
});

describe("clipSliderBounds", () => {
  it("grows_the_box_by_its_span", () => {
    expect(clipSliderBounds({ min: [0, 0, 0], max: [1, 2, 3] })).toEqual({ lo: [-1, -2, -3], hi: [2, 4, 6] });
  });

  it("keeps_a_flat_box_draggable", () => {
    const { lo, hi } = clipSliderBounds({ min: [0, 0, 0], max: [0, 1, 1] });
    expect(lo[0]).toBe(-1e-6);
    expect(hi[0]).toBe(1e-6);
  });
});

describe("clipBoxWithEnd", () => {
  const box = { min: [0, 0, 0] as [number, number, number], max: [2, 2, 2] as [number, number, number] };

  it("moves_one_end_and_leaves_the_input", () => {
    expect(clipBoxWithEnd(box, 1, 0, 1)).toEqual({ min: [0, 1, 0], max: [2, 2, 2] });
    expect(clipBoxWithEnd(box, 2, 1, 5)).toEqual({ min: [0, 0, 0], max: [2, 2, 5] });
    expect(box).toEqual({ min: [0, 0, 0], max: [2, 2, 2] });
  });

  it("never_lets_the_ends_cross", () => {
    expect(clipBoxWithEnd(box, 0, 0, 3)).toEqual({ min: [2, 0, 0], max: [2, 2, 2] });
    expect(clipBoxWithEnd(box, 0, 1, -1)).toEqual({ min: [0, 0, 0], max: [0, 2, 2] });
  });
});
