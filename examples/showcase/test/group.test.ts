// The pure parts of the collision panel's Group mode: the pairs a group
// measures, editing them, each pair's deepest penetration from the members'
// values and nearest partners, and the partner behind one vertex.
import { describe, expect, it } from "vitest";
import { groupPairs, nearestPartner, pairDepths, togglePair } from "../src/panels/group-math";

const NONE = 0xffffffff;

describe("group math", () => {
  it("group_pairs_all", () => {
    expect(groupPairs(["a", "b", "c"], "all")).toEqual([
      ["a", "b"],
      ["a", "c"],
      ["b", "c"],
    ]);
    expect(groupPairs(["a"], "all")).toEqual([]);
    expect(groupPairs([], "all")).toEqual([]);
  });

  it("group_pairs_listed", () => {
    // Member order, each pair once, either way round; non-members and self pairs dropped.
    const pairs: [string, string][] = [
      ["c", "a"],
      ["a", "c"],
      ["b", "x"],
      ["b", "b"],
      ["a", "b"],
    ];
    expect(groupPairs(["a", "b", "c"], pairs)).toEqual([
      ["a", "b"],
      ["a", "c"],
    ]);
    expect(groupPairs(["b", "c"], pairs)).toEqual([]);
  });

  it("toggle_pair", () => {
    const start: [string, string][] = [["a", "b"]];
    expect(togglePair(start, "b", "c")).toEqual([
      ["a", "b"],
      ["b", "c"],
    ]);
    expect(togglePair(start, "b", "a")).toEqual([]);
    expect(start).toEqual([["a", "b"]]);
  });

  it("pair_depths", () => {
    // a's vertices: inside b (-0.3), inside c (-0.1), gap to b; b's: inside a (-0.2); c without values.
    const entries = [
      { mesh: "a", values: new Float32Array([-0.3, -0.1, 0.4]), nearest: new Uint32Array([0, 1, 0]), partners: ["b", "c"] },
      { mesh: "b", values: new Float32Array([-0.2, Infinity]), nearest: new Uint32Array([0, NONE]), partners: ["a"] },
      { mesh: "c", values: null, nearest: null, partners: ["a"] },
    ];
    const pairs: [string, string][] = [
      ["a", "b"],
      ["a", "c"],
      ["b", "c"],
    ];
    const d = pairDepths(entries, pairs);
    expect(d[0]).toBeCloseTo(0.3, 6);
    expect(d[1]).toBeCloseTo(0.1, 6);
    // No member of the pair has values: unknown.
    expect(pairDepths([entries[2]], [["b", "c"]])).toEqual([null]);
    // b has values but none nearest to c (not a partner): no penetration found.
    expect(d[2]).toBe(0);
  });

  it("pair_depths_ignore_stale_indices", () => {
    // A nearest index past the partner list (a target removed since) counts for nobody.
    const entries = [{ mesh: "a", values: new Float32Array([-0.5, -0.1]), nearest: new Uint32Array([3, 0]), partners: ["b"] }];
    expect(pairDepths(entries, [["a", "b"]])[0]).toBeCloseTo(0.1, 6);
  });

  it("nearest_partner", () => {
    const values = new Float32Array([-0.25, 0.5, Infinity, Number.NaN]);
    const nearest = new Uint32Array([1, 0, NONE, NONE]);
    expect(nearestPartner(values, nearest, ["b", "c"], 0)).toEqual({ partner: "c", distance: values[0] });
    expect(nearestPartner(values, nearest, ["b", "c"], 1)).toEqual({ partner: "b", distance: 0.5 });
    expect(nearestPartner(values, nearest, ["b", "c"], 2)).toBeNull();
    expect(nearestPartner(values, nearest, ["b", "c"], 3)).toBeNull();
    expect(nearestPartner(values, nearest, ["b", "c"], 9)).toBeNull();
    expect(nearestPartner(values, new Uint32Array([5]), ["b"], 0)).toBeNull();
  });
});
