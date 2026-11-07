// When the heatmap's engine interaction runs (panels/interaction.ts).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { InteractionTimer, SETTLE_MS } from "../src/panels/interaction";

describe("interaction timer", () => {
  let seen: boolean[];
  let timer: InteractionTimer;
  beforeEach(() => {
    vi.useFakeTimers();
    seen = [];
    timer = new InteractionTimer((active) => seen.push(active));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts_on_press_and_ends_on_release", () => {
    timer.press();
    expect(seen).toEqual([true]);
    expect(timer.pressed).toBe(true);
    timer.release();
    expect(seen).toEqual([true, false]);
    expect(timer.pressed).toBe(false);
  });

  it("ends_when_the_pointer_stops_with_the_button_held", () => {
    timer.press();
    vi.advanceTimersByTime(SETTLE_MS - 1);
    timer.touch();
    vi.advanceTimersByTime(SETTLE_MS - 1);
    expect(seen).toEqual([true]);
    vi.advanceTimersByTime(1);
    expect(seen).toEqual([true, false]);
    // The pointer moves on: it runs again, and a release ends it once.
    timer.touch();
    expect(seen).toEqual([true, false, true]);
    timer.release();
    vi.advanceTimersByTime(10 * SETTLE_MS);
    expect(seen).toEqual([true, false, true, false]);
  });

  it("ignores_moves_without_a_press", () => {
    timer.touch();
    vi.advanceTimersByTime(10 * SETTLE_MS);
    expect(seen).toEqual([]);
    timer.release();
    expect(seen).toEqual([]);
  });
});
