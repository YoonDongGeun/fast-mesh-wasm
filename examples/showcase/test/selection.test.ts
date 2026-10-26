// Multi-selection: plain clicks reset (the primary doubles as the store's
// single selection), shift-clicks toggle while the primary stays first, and
// the Join source seeds from the secondaries in order.
import type { MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { clickMulti, nextJoinSource } from "../src/scene/selection";

const [a, b, c] = [1, 2, 3].map((n) => n as unknown as MeshHandle);

describe("clickMulti", () => {
  it("plain_click_resets", () => {
    expect(clickMulti([a, b], a, c, false)).toEqual({ multi: [c], primary: c });
    expect(clickMulti([a, b], a, null, false)).toEqual({ multi: [], primary: null });
  });

  it("shift_click_adds_a_secondary", () => {
    expect(clickMulti([a], a, b, true)).toEqual({ multi: [a, b], primary: a });
    expect(clickMulti([], null, b, true)).toEqual({ multi: [b], primary: b });
  });

  it("shift_click_toggles_off", () => {
    expect(clickMulti([a, b, c], a, b, true)).toEqual({ multi: [a, c], primary: a });
  });

  it("removing_the_primary_promotes_the_next", () => {
    expect(clickMulti([a, b, c], a, a, true)).toEqual({ multi: [b, c], primary: b });
    expect(clickMulti([a], a, a, true)).toEqual({ multi: [], primary: null });
  });

  it("ignores_shift_clicks_on_empty_space", () => {
    expect(clickMulti([a, b], a, null, true)).toEqual({ multi: [a, b], primary: a });
  });

  it("rebases_a_stale_list_onto_the_primary", () => {
    expect(clickMulti([b, c], a, c, true)).toEqual({ multi: [a, c], primary: a });
    expect(clickMulti([a, b], null, c, true)).toEqual({ multi: [c], primary: c });
  });
});

describe("nextJoinSource", () => {
  it("keeps_a_manual_pick", () => {
    expect(nextJoinSource([2, 3], 3)).toBe(3);
  });

  it("seeds_the_first_secondary", () => {
    expect(nextJoinSource([2, 3], "")).toBe(2);
  });

  it("replaces_a_stale_pick", () => {
    expect(nextJoinSource([3], 2)).toBe(3);
    expect(nextJoinSource([], 2)).toBe("");
  });
});
