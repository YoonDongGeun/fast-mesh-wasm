// Inspection math: the ruler, vertex snapping, placement-aware world
// mapping, section travel and the clip box faces.
import { describe, expect, it } from "vitest";
import {
  boxFaces,
  formatCoord,
  localVertex,
  normalTravel,
  pointInBox,
  readMeasure,
  rulerDistance,
  snapSlot,
  worldFromMatrix,
} from "../src/viewport/inspect-tools";

describe("rulerDistance", () => {
  it("measures_world_points", () => {
    expect(rulerDistance([0, 0, 0], [1, 0, 0])).toBeCloseTo(1, 12);
    expect(rulerDistance([1, 2, 3], [4, 6, 3])).toBeCloseTo(5, 12);
  });

  it("is_zero_for_one_point", () => {
    expect(rulerDistance([1, 2, 3], [1, 2, 3])).toBe(0);
  });
});

describe("snapSlot", () => {
  it("takes_the_largest_weight", () => {
    expect(snapSlot([0.7, 0.2, 0.1])).toBe(0);
    expect(snapSlot([0.1, 0.7, 0.2])).toBe(1);
    expect(snapSlot([0.1, 0.2, 0.7])).toBe(2);
  });

  it("breaks_ties_to_the_lowest_index_like_the_hover_tip", () => {
    expect(snapSlot([1 / 3, 1 / 3, 1 / 3])).toBe(0);
    expect(snapSlot([0.25, 0.5, 0.25])).toBe(1);
  });
});

describe("localVertex", () => {
  it("reads_the_slot", () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect([...localVertex(positions, 1)]).toEqual([4, 5, 6]);
  });
});

describe("worldFromMatrix", () => {
  // Column-major 4x4, like `scene.matrix4`.
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  it("leaves_identity_points_alone", () => {
    expect([...worldFromMatrix(identity, [1, 2, 3])]).toEqual([1, 2, 3]);
  });

  it("applies_translation", () => {
    const m = [...identity];
    m[12] = 5;
    m[13] = -2;
    m[14] = 7;
    expect([...worldFromMatrix(m, [1, 1, 1])]).toEqual([6, -1, 8]);
  });

  it("applies_non_uniform_scale", () => {
    // A #103 placement: scale [2, 3, 4] plus a translation.
    const m = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 10, 20, 30, 1];
    expect([...worldFromMatrix(m, [1, 1, 1])]).toEqual([12, 23, 34]);
  });
});

describe("normalTravel", () => {
  it("counts_travel_along_the_normal", () => {
    expect(normalTravel([0, 0, 0], [0, 0, 5], [0, 0, 1])).toBeCloseTo(5, 12);
    expect(normalTravel([0, 0, 0], [0, 0, -5], [0, 0, 1])).toBeCloseTo(-5, 12);
    expect(normalTravel([1, 2, 3], [9, 2, 3], [0, 0, 1])).toBeCloseTo(0, 12);
  });
});

describe("boxFaces", () => {
  it("keeps_the_inside_and_clips_each_side", () => {
    const min: [number, number, number] = [0, 0, 0];
    const max: [number, number, number] = [2, 4, 6];
    const faces = boxFaces(min, max);
    expect(faces.length).toBe(6);
    const inside = (p: readonly [number, number, number]) =>
      faces.every((f) => f.normal[0] * p[0] + f.normal[1] * p[1] + f.normal[2] * p[2] + f.constant >= -1e-12);
    expect(inside([1, 2, 3])).toBe(true);
    expect(inside([0, 0, 0])).toBe(true);
    expect(inside([2, 4, 6])).toBe(true);
    expect(inside([-0.1, 2, 3])).toBe(false);
    expect(inside([2.1, 2, 3])).toBe(false);
    expect(inside([1, -0.1, 3])).toBe(false);
    expect(inside([1, 4.1, 3])).toBe(false);
    expect(inside([1, 2, -0.1])).toBe(false);
    expect(inside([1, 2, 6.1])).toBe(false);
  });
});

describe("pointInBox", () => {
  it("agrees_with_the_faces", () => {
    expect(pointInBox([1, 1, 1], [0, 0, 0], [2, 2, 2])).toBe(true);
    expect(pointInBox([3, 1, 1], [0, 0, 0], [2, 2, 2])).toBe(false);
  });
});

describe("formatCoord", () => {
  it("shortens_coordinates", () => {
    expect(formatCoord(1.23456789)).toBe("1.235");
    expect(formatCoord(0)).toBe("0");
  });
});

describe("readMeasure", () => {
  const reading = { area: 6, volume: 1, closed: true };

  it("takes a local Scene's answer", async () => {
    expect(await readMeasure({ measure: () => reading }, 1)).toEqual({ area: 6, volume: 1 });
  });

  it("takes a WorkerScene's promise", async () => {
    expect(await readMeasure({ measure: () => Promise.resolve(reading) }, 1)).toEqual({ area: 6, volume: 1 });
  });

  it("has no reading without measure, or when it fails", async () => {
    expect(await readMeasure({}, 1)).toBeNull();
    expect(await readMeasure({ measure: () => Promise.reject(new Error("gone")) }, 1)).toBeNull();
    expect(
      await readMeasure(
        {
          measure: () => {
            throw new Error("gone");
          },
        },
        1,
      ),
    ).toBeNull();
  });
});
