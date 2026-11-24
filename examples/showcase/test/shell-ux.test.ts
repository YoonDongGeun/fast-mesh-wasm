// The shell's pure behaviour, apart from its components: the viewport hint
// (an armed tool speaks first, touch words on a coarse pointer), the boot
// landing frame for a volume, the command palette's ranking and cap, and
// the sculpt inspector's dyntopo detail readout. Expected values are
// written out by hand (1 in = 25.4 mm).
import { describe, expect, it } from "vitest";
import { volumeLandingBox } from "../src/app/ux/use-scene-loading";
import { type HintState, viewportHint } from "../src/app/ux/viewport-hint";
import type { CommandPaletteItem } from "../src/shell/ux/command-registry";
import { filterCommands } from "../src/shell/ux/command-search";
import { dyntopoDetailLabel } from "../src/shell/ux/use-sculpt-inspector";

const IDLE: HintState = {
  sculpting: false,
  maskPaint: false,
  selectTool: null,
  deformTool: null,
  printTool: null,
  picking: false,
  selected: false,
  coarse: false,
};

describe("viewport hint", () => {
  it("speaks_of_selecting_then_moving", () => {
    expect(viewportHint(IDLE)).toBe(
      "Click a mesh to select it · drag to rotate · scroll to zoom · Ctrl+K search · drop files to load",
    );
    expect(viewportHint({ ...IDLE, selected: true })).toBe(
      "Drag the gizmo to move · T/R/S move/rotate/scale · F focus · Ctrl+K search · Esc deselect",
    );
  });

  it("uses_touch_words_on_a_coarse_pointer", () => {
    expect(viewportHint({ ...IDLE, coarse: true })).toBe(
      "Tap a mesh to select it · drag to rotate · pinch to zoom · two fingers to pan",
    );
    expect(viewportHint({ ...IDLE, coarse: true, selected: true })).toBe(
      "Drag the gizmo to move · tap empty space to deselect",
    );
  });

  it("puts_the_armed_tool_first", () => {
    const all: HintState = {
      sculpting: true,
      maskPaint: true,
      selectTool: "lasso",
      deformTool: "ffdCage",
      printTool: "drainHole",
      picking: true,
      selected: true,
      coarse: true,
    };
    expect(viewportHint(all)).toBe("Hover any visible mesh to see brush · drag to sculpt · Esc disarms");
    expect(viewportHint({ ...all, sculpting: false })).toBe("Hover any visible mesh · drag to paint mask · Esc disarms");
    expect(viewportHint({ ...all, sculpting: false, maskPaint: false })).toBe("Use lasso selection on mesh · Esc disarms");
    const deform = { ...all, sculpting: false, maskPaint: false, selectTool: null };
    expect(viewportHint(deform)).toBe("Drag deformation handle in viewport · Esc disarms");
    const print = { ...deform, deformTool: null };
    expect(viewportHint(print)).toBe("Click mesh surface to punch drain hole · Esc disarms");
    expect(viewportHint({ ...print, printTool: "placeStamp" })).toBe(
      "Click mesh surface to align and place stamp · Esc disarms",
    );
    expect(viewportHint({ ...print, printTool: null })).toBe("Click the mesh to set the brush center · Esc cancels");
  });
});

describe("volume landing frame", () => {
  // 10 x 20 x 30 voxels at 1 x 1 x 2 mm: a 10 x 20 x 60 mm box.
  const payload: Parameters<typeof volumeLandingBox>[0] = {
    columns: 10,
    rows: 20,
    slices: 30,
    spacing: [1, 1, 2],
    origin: [0, 0, 0],
  };
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  it("frames_the_box_middle_with_half_its_diagonal", () => {
    const box = volumeLandingBox(payload, [0, 0, 0], identity);
    expect(box.center).toEqual([5, 10, 30]);
    // sqrt(10² + 20² + 60²) / 2 = sqrt(4100) / 2.
    expect(box.radius).toBeCloseTo(32.0156, 4);
  });

  it("takes_the_middle_relative_to_the_mesh_center_then_places_it", () => {
    // The box middle sits on the mesh centre, which the matrix moves to (1, 2, 3).
    const moved = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1];
    expect(volumeLandingBox(payload, [5, 10, 30], moved).center).toEqual([1, 2, 3]);
    // A quarter turn about z (column-major): x goes to y, y to -x.
    const turned = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    expect(volumeLandingBox(payload, [0, 0, 0], turned).center).toEqual([-10, 5, 30]);
  });
});

describe("command ranking", () => {
  const command = (id: string, title: string, keywords = ""): CommandPaletteItem => ({
    id,
    title,
    category: "Tool",
    keywords,
    run: () => {},
  });

  it("keeps_registry_order_for_an_empty_query_and_caps_at_24", () => {
    const many = Array.from({ length: 30 }, (_, i) => command(`c${i}`, `Command ${i}`));
    const shown = filterCommands(many, "  ");
    expect(shown.map((c) => c.id)).toEqual(many.slice(0, 24).map((c) => c.id));
  });

  it("ranks_substrings_over_subsequences_and_drops_misses", () => {
    const list = [
      command("sub", "Reset Upload Lines"), // r…u…l, a subsequence only
      command("miss", "Camera Front"),
      command("hit", "Inspect: Ruler"),
    ];
    expect(filterCommands(list, "rul").map((c) => c.id)).toEqual(["hit", "sub"]);
  });

  it("searches_keywords_and_category_too", () => {
    const list = [command("a", "Alpha", "lattice cage"), command("b", "Beta")];
    expect(filterCommands(list, "lattice").map((c) => c.id)).toEqual(["a"]);
    // The match nearer the start ranks first: "Beta Tool" before "Alpha Tool".
    expect(filterCommands(list, "tool").map((c) => c.id)).toEqual(["b", "a"]);
  });
});

describe("dyntopo detail readout", () => {
  it("is_a_percentage_when_brush_relative", () => {
    expect(dyntopoDetailLabel("brush-relative", 0.25, 100, "mm")).toBe("25%");
    expect(dyntopoDetailLabel("brush-relative", 0.044, 100, "in")).toBe("4%");
  });

  it("is_a_length_on_the_mesh_otherwise", () => {
    // 0.5 of a 50.8 mm radius: 25.4 mm, or 1 in.
    expect(dyntopoDetailLabel("constant-detail", 0.5, 50.8, "mm")).toBe("25.4 mm");
    expect(dyntopoDetailLabel("split-only", 0.5, 50.8, "in")).toBe("1 in");
  });
});
