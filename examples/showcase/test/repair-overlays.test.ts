// The repair-issue overlay geometry: boundary polylines, non-manifold
// segments, degenerate triangle edges and small-component fills built from
// `analyze`, and the three.js group hanging them under the bound mesh.
import type { Analysis } from "fast-mesh-wasm";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  buildRepairOverlayGroup,
  clearRepairOverlays,
  countRepairOverlays,
  disposeRepairOverlayGroup,
  hasRepairOverlays,
  holeSegments,
  nonManifoldSegments,
  REPAIR_OVERLAY_COLORS,
  smallComponentTris,
  triangleEdgeSegments,
  triangleSoup,
} from "../src/viewport/repair-overlays";

/** An analysis with nothing to report, then the given fields. */
function analysis(patch: Partial<Analysis>): Analysis {
  return {
    loops: [],
    coincidentGroups: 0,
    coincident: [],
    degenerate: 0,
    degenerateList: new Uint32Array(),
    duplicate: 0,
    duplicateList: new Uint32Array(),
    nonManifoldEdges: 0,
    nonManifoldEdgeList: [],
    nonManifoldVertices: new Uint32Array(),
    pinchVertices: new Uint32Array(),
    components: [{ triangles: new Uint32Array(12), area: 6, closed: true }],
    orientationConflicts: 0,
    selfIntersects: null,
    ...patch,
  };
}

const loop = (vertices: number[], perimeter: number, closed = true) => ({
  id: { minVertex: 0, len: vertices.length, hash: 0n },
  vertices: Uint32Array.from(vertices),
  perimeter,
  area: 0,
  closed,
  clean: true,
});

const component = (triangles: number[], closed = true) => ({
  triangles: Uint32Array.from(triangles),
  area: triangles.length,
  closed,
});

/** A unit square in z=0: corners 0..3. */
const SQUARE = Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);

describe("holeSegments", () => {
  it("wraps_closed_loops", () => {
    const segs = holeSegments([loop([0, 1, 2, 3], 4)], SQUARE);
    expect(segs.length).toBe(4 * 6);
    // First edge: corner 0 to corner 1.
    expect(Array.from(segs.slice(0, 6))).toEqual([0, 0, 0, 1, 0, 0]);
    // Last edge wraps: corner 3 to corner 0.
    expect(Array.from(segs.slice(18, 24))).toEqual([0, 1, 0, 0, 0, 0]);
  });

  it("leaves_open_chains_open", () => {
    const segs = holeSegments([loop([0, 1, 2], 2, false)], SQUARE);
    expect(segs.length).toBe(2 * 6);
  });

  it("skips_tiny_loops", () => {
    expect(holeSegments([loop([0], 0), loop([], 0)], SQUARE).length).toBe(0);
  });

  it("breaks_at_unknown_vertices", () => {
    // Vertex 9 does not exist: both its edges drop, the other two stay.
    const segs = holeSegments([loop([0, 1, 9, 3], 4)], SQUARE);
    expect(segs.length).toBe(2 * 6);
  });
});

describe("nonManifoldSegments", () => {
  it("joins_the_pairs", () => {
    const segs = nonManifoldSegments(
      [
        [0, 1],
        [2, 3],
      ],
      SQUARE,
    );
    expect(segs.length).toBe(2 * 6);
    expect(Array.from(segs.slice(0, 6))).toEqual([0, 0, 0, 1, 0, 0]);
  });

  it("skips_unknown_vertices", () => {
    expect(nonManifoldSegments([[0, 9]], SQUARE).length).toBe(0);
    expect(nonManifoldSegments([], SQUARE).length).toBe(0);
  });
});

describe("triangleEdgeSegments", () => {
  const indices = Uint32Array.from([0, 1, 2, 0, 0, 1]);

  it("draws_three_edges_per_triangle", () => {
    expect(triangleEdgeSegments(Uint32Array.from([0]), indices, SQUARE).length).toBe(3 * 6);
  });

  it("drops_point_edges", () => {
    // Corners 0, 0, 1: the 0-0 edge is a point, the other two draw.
    expect(triangleEdgeSegments(Uint32Array.from([1]), indices, SQUARE).length).toBe(2 * 6);
  });

  it("skips_unknown_slots", () => {
    expect(triangleEdgeSegments(Uint32Array.from([7]), indices, SQUARE).length).toBe(0);
    expect(triangleEdgeSegments(Uint32Array.from([]), indices, SQUARE).length).toBe(0);
  });
});

describe("smallComponentTris", () => {
  it("keeps_everything_but_the_largest", () => {
    const kept = smallComponentTris([
      component([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      component([10, 11, 12]),
      component([13, 14, 15, 16, 17]),
    ]);
    expect(Array.from(kept)).toEqual([10, 11, 12, 13, 14, 15, 16, 17]);
  });

  it("ties_keep_the_first_largest", () => {
    expect(smallComponentTris([component([0]), component([1])]).length).toBe(1);
  });

  it("one_component_highlights_nothing", () => {
    expect(smallComponentTris([component([0, 1])]).length).toBe(0);
    expect(smallComponentTris([]).length).toBe(0);
  });
});

describe("triangleSoup", () => {
  it("expands_slots_to_positions", () => {
    const soup = triangleSoup(Uint32Array.from([0]), Uint32Array.from([0, 1, 2]), SQUARE);
    expect(Array.from(soup)).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0]);
  });

  it("skips_unknown_slots", () => {
    expect(triangleSoup(Uint32Array.from([5]), Uint32Array.from([0, 1, 2]), SQUARE).length).toBe(0);
  });
});

describe("countRepairOverlays", () => {
  it("counts_primitives", () => {
    const counts = countRepairOverlays(
      analysis({
        loops: [loop([0, 1, 2, 3], 4), loop([0, 1], 1, false)],
        nonManifoldEdges: 2,
        nonManifoldEdgeList: [
          [0, 1],
          [2, 3],
        ],
        degenerate: 1,
        degenerateList: Uint32Array.from([4]),
        duplicate: 1,
        duplicateList: Uint32Array.from([5]),
        components: [component([0, 1, 2, 3]), component([4])],
      }),
    );
    // 4 wraparound + 1 open edge, 2 pairs, 2 triangles, 1 small triangle.
    expect(counts).toEqual({ holes: 5, nonManifold: 2, degenerate: 2, components: 1 });
  });
});

describe("buildRepairOverlayGroup", () => {
  const kinds = { holes: true, nonManifold: true, degenerate: true, components: true };
  const indices = Uint32Array.from([0, 1, 2, 0, 2, 3]);

  const defective = () =>
    analysis({
      loops: [loop([0, 1, 2, 3], 4)],
      nonManifoldEdges: 1,
      nonManifoldEdgeList: [[0, 2]],
      degenerate: 1,
      degenerateList: Uint32Array.from([0]),
      components: [component([0, 1]), component([1])],
    });

  it("returns_null_when_nothing_is_on_or_found", () => {
    expect(
      buildRepairOverlayGroup(defective(), SQUARE, indices, {
        holes: false,
        nonManifold: false,
        degenerate: false,
        components: false,
      }),
    ).toBeNull();
    expect(buildRepairOverlayGroup(analysis({}), SQUARE, indices, kinds)).toBeNull();
  });

  it("draws_one_line_set_per_kind", () => {
    const group = buildRepairOverlayGroup(defective(), SQUARE, indices, kinds);
    expect(group).not.toBeNull();
    const lines = group!.children[0].children.filter((c) => c instanceof THREE.LineSegments);
    // Holes, non-manifold, degenerate lines plus the component fill.
    expect(lines.length).toBe(3);
    expect(group!.children[0].children.some((c) => c instanceof THREE.Mesh)).toBe(true);
    for (const line of lines) {
      expect((line as THREE.LineSegments).renderOrder).toBe(999);
      expect((line as THREE.LineSegments).material).toBeDefined();
    }
  });

  it("colours_holes_amber", () => {
    const group = buildRepairOverlayGroup(analysis({ loops: [loop([0, 1, 2, 3], 4)] }), SQUARE, indices, kinds);
    const line = group!.children[0].children[0] as THREE.LineSegments;
    expect((line.material as THREE.LineBasicMaterial).color.getHex()).toBe(REPAIR_OVERLAY_COLORS.holes);
  });

  it("respects_disabled_kinds", () => {
    const group = buildRepairOverlayGroup(defective(), SQUARE, indices, { ...kinds, holes: false });
    const colours = new Set<number>();
    group!.children[0].traverse((c) => {
      const mat = (c as THREE.Mesh).material as THREE.Material | undefined;
      if (mat !== undefined && "color" in mat) {
        colours.add(((mat as THREE.MeshBasicMaterial).color as THREE.Color).getHex());
      }
    });
    expect(colours.has(REPAIR_OVERLAY_COLORS.holes)).toBe(false);
    expect(colours.has(REPAIR_OVERLAY_COLORS.nonManifold)).toBe(true);
  });

  it("hangs_and_clears", () => {
    const parent = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
    expect(hasRepairOverlays(parent)).toBe(false);
    const group = buildRepairOverlayGroup(defective(), SQUARE, indices, kinds);
    parent.add(group!);
    expect(hasRepairOverlays(parent)).toBe(true);
    clearRepairOverlays(parent);
    expect(hasRepairOverlays(parent)).toBe(false);
    expect(parent.children.length).toBe(0);
  });

  it("disposes_geometries", () => {
    const group = buildRepairOverlayGroup(defective(), SQUARE, indices, kinds);
    let disposed = 0;
    group!.children[0].traverse((c) => {
      const mesh = c as THREE.Mesh;
      if (mesh.geometry !== undefined) {
        mesh.geometry.addEventListener("dispose", () => disposed++);
      }
    });
    disposeRepairOverlayGroup(group!);
    expect(disposed).toBeGreaterThan(0);
  });
});
