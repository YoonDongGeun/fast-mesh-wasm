// Face selections as triangle index sets: live slots (a `[0, 0, 0]`
// triangle is dead) and sanitizing a selection against the index buffer.
import { describe, expect, it } from "vitest";
import { isLiveTriangleSlot, sanitizeSelectedTriangles } from "../src/features/select/ux/face-sets";

// Three triangles, the middle one dead.
const indices = new Uint32Array([0, 1, 2, 0, 0, 0, 2, 3, 0]);

describe("isLiveTriangleSlot", () => {
  it("rejects_dead_and_out_of_range_slots", () => {
    expect(isLiveTriangleSlot(indices, 0)).toBe(true);
    expect(isLiveTriangleSlot(indices, 1)).toBe(false);
    expect(isLiveTriangleSlot(indices, 2)).toBe(true);
    expect(isLiveTriangleSlot(indices, 3)).toBe(false);
    expect(isLiveTriangleSlot(indices, -1)).toBe(false);
  });
});

describe("sanitizeSelectedTriangles", () => {
  it("sorts_dedupes_and_drops_dead_or_invalid_slots", () => {
    expect([...sanitizeSelectedTriangles(indices, [2, 1, 0, 2, 7, 1.5, -1])]).toEqual([0, 2]);
  });

  it("is_empty_for_an_empty_selection", () => {
    expect(sanitizeSelectedTriangles(indices, []).length).toBe(0);
    expect(sanitizeSelectedTriangles(indices, [1]).length).toBe(0);
  });
});
