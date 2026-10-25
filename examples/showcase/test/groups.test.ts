// The pure parts of named mesh groups: ticking, creating, colouring and
// dropping members on removal.
import { describe, expect, it } from "vitest";
import { createGroup, dropMember, removeGroup, setColored, toggleCheck } from "../src/panels/groups";

describe("named groups", () => {
  it("toggle_check_keeps_list_order", () => {
    expect(toggleCheck([], ["a", "b", "c"], "b", true)).toEqual(["b"]);
    expect(toggleCheck(["b"], ["a", "b", "c"], "a", true)).toEqual(["a", "b"]);
    expect(toggleCheck(["a", "b"], ["a", "b", "c"], "a", false)).toEqual(["b"]);
    expect(toggleCheck(["b"], ["a", "b", "c"], "b", true)).toEqual(["b"]);
  });

  it("create_group_names_and_dedupes", () => {
    const g = createGroup([], 0, "  ", "Group 1", ["b", "a", "b"]);
    expect(g).toEqual([{ id: 0, name: "Group 1", members: ["b", "a"], colored: false }]);
    const h = createGroup(g, 1, " pair ", "Group 2", ["a"]);
    expect(h.map((x) => x.name)).toEqual(["Group 1", "pair"]);
  });

  it("remove_group_and_color", () => {
    const g = createGroup(createGroup([], 0, "a", "G1", ["x"]), 1, "b", "G2", ["y"]);
    expect(setColored(g, 1, true).map((x) => x.colored)).toEqual([false, true]);
    expect(removeGroup(g, 0).map((x) => x.name)).toEqual(["b"]);
    expect(removeGroup(g, 9)).toEqual(g);
  });

  it("drop_member_on_removal", () => {
    const g = createGroup(createGroup([], 0, "a", "G1", ["x", "y"]), 1, "b", "G2", ["y"]);
    expect(dropMember(g, "y").map((x) => x.members)).toEqual([["x"], []]);
  });
});
