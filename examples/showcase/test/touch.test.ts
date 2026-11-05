// The rules the canvas's pointer listeners share on touch.
import { describe, expect, it } from "vitest";
import { isExtraFinger, tapSlop } from "../src/viewport/touch";

describe("isExtraFinger", () => {
  it("is_only_a_non_primary_touch", () => {
    expect(isExtraFinger({ pointerType: "touch", isPrimary: false })).toBe(true);
    expect(isExtraFinger({ pointerType: "touch", isPrimary: true })).toBe(false);
    expect(isExtraFinger({ pointerType: "mouse", isPrimary: true })).toBe(false);
    expect(isExtraFinger({ pointerType: "mouse", isPrimary: false })).toBe(false);
    expect(isExtraFinger({ pointerType: "pen", isPrimary: false })).toBe(false);
  });
});

describe("tapSlop", () => {
  it("keeps_the_mouse_slop_and_widens_for_a_finger", () => {
    expect(tapSlop("mouse", 2)).toBe(2);
    expect(tapSlop("pen", 6)).toBe(6);
    expect(tapSlop("touch", 2)).toBeGreaterThanOrEqual(10);
    expect(tapSlop("touch", 6)).toBe(tapSlop("touch", 2));
  });
});
