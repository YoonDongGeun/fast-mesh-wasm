// The rules the canvas's pointer listeners share on touch.
import { describe, expect, it } from "vitest";
import { FingerTracker, isExtraFinger, tapSlop } from "../src/viewport/touch";

describe("FingerTracker", () => {
  it("counts_only_canvas_fingers_as_a_gesture", () => {
    const f = new FingerTracker();
    f.down(1, false, true, 0, 0); // a thumb on a panel
    f.down(2, true, false, 10, 10);
    expect(f.isExtra(2)).toBe(false);
    f.down(3, true, false, 20, 10);
    expect(f.isExtra(3)).toBe(true);
    expect(f.has(1)).toBe(false);
    expect(f.has(2)).toBe(true);
  });

  it("keeps_a_finger_extra_until_the_canvas_empties", () => {
    const f = new FingerTracker();
    f.down(1, true, true, 0, 0);
    f.down(2, true, false, 5, 0);
    f.up(1, false, 0, 0);
    expect(f.isExtra(2)).toBe(true);
    f.down(3, true, false, 9, 0);
    expect(f.isExtra(3)).toBe(true);
    f.up(2, false, 5, 0);
    f.up(3, false, 9, 0);
    f.down(4, true, true, 0, 0);
    expect(f.isExtra(4)).toBe(false);
  });

  it("reports_a_tap_only_when_the_browser_will_not_click", () => {
    const f = new FingerTracker();
    // Alone on the glass: the browser clicks.
    f.down(1, true, true, 0, 0);
    expect(f.up(1, false, 3, 0)).toBe(false);
    // A thumb was resting when it landed.
    f.down(2, false, true, 0, 0);
    f.down(3, true, false, 0, 0);
    expect(f.up(3, false, 3, 0)).toBe(true);
    // The thumb stays; this press wandered too far.
    f.down(4, true, false, 0, 0);
    expect(f.up(4, false, 40, 0)).toBe(false);
    // Nor a cancel.
    f.down(5, true, false, 0, 0);
    expect(f.up(5, true, 0, 0)).toBe(false);
    // Nor a pinch.
    f.down(6, true, false, 0, 0);
    f.down(7, true, false, 9, 0);
    expect(f.up(6, false, 0, 0)).toBe(false);
    expect(f.up(7, false, 9, 0)).toBe(false);
  });

  it("reports_a_tap_when_the_other_finger_lands_during_the_press", () => {
    const f = new FingerTracker();
    f.down(1, true, true, 0, 0);
    f.down(2, false, false, 0, 0);
    f.up(2, false, 0, 0);
    expect(f.up(1, false, 0, 0)).toBe(true);
  });

  it("recovers_from_a_lost_lift_at_the_next_primary_press", () => {
    const f = new FingerTracker();
    f.down(1, true, true, 0, 0);
    f.down(2, true, true, 0, 0);
    expect(f.isExtra(2)).toBe(false);
  });
});

describe("isExtraFinger", () => {
  it("is_never_a_mouse_or_pen", () => {
    expect(isExtraFinger({ pointerType: "mouse", pointerId: 7 })).toBe(false);
    expect(isExtraFinger({ pointerType: "pen", pointerId: 7 })).toBe(false);
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
