// Selection primitives: brush/lasso picks, component picks, grow/shrink,
// invert, isolate and the survive-or-fail selection contract.
import { afterEach, describe, expect, it } from "vitest";
import { FastMeshError, Scene, type MeshHandle } from "../src/node.js";

function codeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected a throw");
}

/** An n×n vertex grid on the XY plane. */
function grid(n: number): { positions: Float64Array; indices: Uint32Array } {
  const positions = new Float64Array(n * n * 3);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      positions.set([x, y, 0], (y * n + x) * 3);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let y = 0; y < n - 1; y++) {
    for (let x = 0; x < n - 1; x++) {
      const a = y * n + x;
      indices.set([a, a + 1, a + n + 1, a, a + n + 1, a + n], k);
      k += 6;
    }
  }
  return { positions, indices };
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

const scenes: Scene[] = [];
function scene(): Scene {
  const s = new Scene();
  scenes.push(s);
  return s;
}
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

describe("selection", () => {
  it("brush_picks_a_sphere_in_local_coords", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    // Local coordinates subtract the bounding-box centre (1, 1, 0).
    const sel = s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: 1 });
    expect(sel.mesh).toBe(mesh);
    expect(Array.from(sel.vertices)).toEqual([1, 3, 4, 5, 7]);
    expect(s.selectionAlive(sel)).toBe(true);
  });

  it("brush_radius_zero_picks_exact_hits_only", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    const sel = s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: 0 });
    expect(Array.from(sel.vertices)).toEqual([4]);
  });

  it("brush_in_world_space_follows_the_placement", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [10, 0, 0] });
    const m = s.matrix4(mesh);
    const world = [m[12], m[13], m[14]] as [number, number, number];
    const sel = s.select(mesh, { kind: "brush", center: world, radius: 1, space: "world" });
    expect(Array.from(sel.vertices)).toEqual([1, 3, 4, 5, 7]);
  });

  it("brush_skips_dead_slots", () => {
    const s = scene();
    const mesh = s.addMesh(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), [0, 1, 2]);
    s.edit(mesh, [{ op: "removeTriangles", triangles: [0] }]);
    expect(s.triangleCount(mesh)).toBe(0);
    const sel = s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: 100 });
    expect(Array.from(sel.vertices)).toEqual([]);
  });

  it("lasso_picks_a_projected_polygon", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    // Local x in [-0.5, 0.5]: the middle column.
    const sel = s.select(mesh, {
      kind: "lasso",
      polygon: [-0.5, -2, 0.5, -2, 0.5, 2, -0.5, 2],
    });
    expect(Array.from(sel.vertices)).toEqual([1, 4, 7]);
  });

  it("component_picks_the_connected_patch", () => {
    const s = scene();
    const mesh = s.addMesh(
      new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 9, 9, 9, 10, 9, 9, 9, 10, 9]),
      [0, 1, 2, 3, 4, 5],
    );
    expect(Array.from(s.select(mesh, { kind: "component", seed: 0 }).vertices)).toEqual([0, 1, 2]);
    expect(Array.from(s.select(mesh, { kind: "component", seed: 4 }).vertices)).toEqual([3, 4, 5]);
    expect(codeOf(() => s.select(mesh, { kind: "component", seed: 99 }))).toBe("INVALID_INDEX");
  });

  it("grow_then_shrink_round_trips", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    const single = s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: 0 });
    const grown = s.selectGrow(single, 1);
    expect(Array.from(grown.vertices)).toEqual([0, 1, 3, 4, 5, 7, 8]);
    const shrunk = s.selectShrink(grown, 1);
    expect(Array.from(shrunk.vertices)).toEqual([0, 4, 8]);
    expect(Array.from(s.selectGrow(single, 0).vertices)).toEqual([4]);
  });

  it("shrink_peels_the_boundary_ring", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    // Local centre (-1, 0, 0): the left column plus its neighbours.
    const half = s.select(mesh, { kind: "brush", center: [-1, 0, 0], radius: 1.5 });
    expect(Array.from(half.vertices)).toEqual([0, 1, 3, 4, 6, 7]);
    expect(Array.from(s.selectShrink(half, 1).vertices)).toEqual([0, 3, 6]);
  });

  it("invert_takes_the_complement", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    const sel = s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: 1 });
    expect(Array.from(s.selectInvert(sel).vertices)).toEqual([0, 2, 6, 8]);
  });

  it("edits_invalidate_selections_loudly", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    const sel = s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: 1 });
    expect(s.selectionAlive(sel)).toBe(true);
    s.edit(mesh, [{ op: "setPositions", vertices: [4], positions: new Float64Array([1, 1, 0.5]) }]);
    expect(s.selectionAlive(sel)).toBe(false);
    expect(codeOf(() => s.selectGrow(sel))).toBe("STALE_SELECTION");
    expect(codeOf(() => s.selectShrink(sel))).toBe("STALE_SELECTION");
    expect(codeOf(() => s.selectInvert(sel))).toBe("STALE_SELECTION");
    expect(codeOf(() => s.isolate(mesh, sel))).toBe("STALE_SELECTION");
    // Undo is a topology rewrite too: it stays stale rather than reviving.
    s.undo();
    expect(s.selectionAlive(sel)).toBe(false);
  });

  it("isolate_removes_triangles_fully_outside", () => {
    const s = scene();
    const mesh = s.addMesh(
      new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 5, 1, 0, 5, 1, 1, 5, 0, 1, 5]),
      [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
    );
    const sel = s.select(mesh, { kind: "component", seed: 0 });
    expect(Array.from(sel.vertices)).toEqual([0, 1, 2, 3]);
    const report = s.isolate(mesh, sel);
    expect(report.removedTriangles).toBe(2);
    expect(s.triangleCount(mesh)).toBe(2);
    expect(s.selectionAlive(sel)).toBe(false);
    expect(s.selectionAlive(report.selection)).toBe(true);
    expect(Array.from(report.selection.vertices)).toEqual([0, 1, 2, 3]);
    s.undo();
    expect(s.triangleCount(mesh)).toBe(4);
  });

  it("isolate_rejects_empty_and_foreign_selections", () => {
    const s = scene();
    const g = grid(3);
    const a = s.addMesh(g.positions, g.indices);
    const b = s.addMesh(g.positions, g.indices);
    const empty = s.select(a, { kind: "brush", center: [99, 99, 99], radius: 0.01 });
    expect(empty.vertices.length).toBe(0);
    expect(codeOf(() => s.isolate(a, empty))).toBe("INVALID_ARGUMENT");
    const foreign = s.select(b, { kind: "brush", center: [0, 0, 0], radius: 1 });
    expect(codeOf(() => s.isolate(a, foreign as unknown as Parameters<Scene["isolate"]>[1]))).toBe(
      "INVALID_ARGUMENT",
    );
  });

  it("rejects_bad_specs", () => {
    const s = scene();
    const g = grid(3);
    const mesh = s.addMesh(g.positions, g.indices);
    expect(codeOf(() => s.select(mesh, { kind: "brush", center: [0, 0, 0], radius: -1 }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.select(mesh, { kind: "lasso", polygon: [0, 0, 1, 1] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.selectGrow(s.select(mesh, { kind: "component", seed: 0 }), -1))).toBe(
      "INVALID_ARGUMENT",
    );
    const gone: MeshHandle = mesh;
    s.removeMesh(mesh);
    expect(codeOf(() => s.select(gone, { kind: "component", seed: 0 }))).toBe("UNKNOWN_MESH");
  });
});
