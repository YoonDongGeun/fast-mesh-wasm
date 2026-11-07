// When undo and redo run around a sculpt stroke (see `afterStroke`).
import { afterEach, describe, expect, it } from "vitest";
import { afterStroke, view } from "../src/viewport/view";

afterEach(() => {
  view.stroking = false;
  view.strokeLanding = null;
});

describe("afterStroke", () => {
  it("runs_at_once_with_no_stroke", () => {
    let runs = 0;
    afterStroke(() => runs++);
    expect(runs).toBe(1);
  });

  it("is_ignored_while_the_button_is_down", async () => {
    let runs = 0;
    view.stroking = true;
    afterStroke(() => runs++);
    view.stroking = false;
    await Promise.resolve();
    expect(runs).toBe(0);
  });

  it("waits_for_a_released_stroke_to_land_and_keeps_the_order", async () => {
    const asked: string[] = [];
    let land = () => {};
    const landing: Promise<void> = new Promise<void>((resolve) => {
      land = resolve;
    }).then(() => {
      view.strokeLanding = null;
    });
    view.strokeLanding = landing;
    afterStroke(() => asked.push("first"));
    afterStroke(() => asked.push("second"));
    expect(asked).toEqual([]);
    land();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(asked).toEqual(["first", "second"]);
  });

  it("waits_again_if_a_new_stroke_is_held_when_the_old_one_lands", async () => {
    let runs = 0;
    let land = () => {};
    view.strokeLanding = new Promise<void>((resolve) => {
      land = resolve;
    }).then(() => {
      view.strokeLanding = null;
    });
    afterStroke(() => runs++);
    view.stroking = true;
    land();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(runs).toBe(0);
  });
});
