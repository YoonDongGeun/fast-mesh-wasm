// The deform panel's input parsing (#193): integer inputs clamp into their
// range and fall back when not a number; real inputs fall back when not
// finite. The panel's actions run against the real engine in
// `wave6-deform-193.test.tsx`.
import { describe, expect, it } from "vitest";
import { clampInt, parseFinite } from "../src/features/sculpt/ux/use-deform-panel";

describe("deform panel inputs", () => {
  it("clamps_integers_into_range", () => {
    expect(clampInt("3.7", 0, 10, 5)).toBe(3);
    expect(clampInt("-4.2", 0, 10, 5)).toBe(0);
    expect(clampInt("99", 2, 16, 3)).toBe(16);
    expect(clampInt("", 1, 10, 5)).toBe(1);
    expect(clampInt("abc", 0, 10, 5)).toBe(5);
  });

  it("parses_finite_numbers_or_falls_back", () => {
    expect(parseFinite("0.25", 1)).toBe(0.25);
    expect(parseFinite("-3", 1)).toBe(-3);
    expect(parseFinite("Infinity", 1)).toBe(1);
    expect(parseFinite("x", 2)).toBe(2);
  });
});
