// Geometry edit op lists, compaction, analysis and repair through the Node
// entry's sources.
import { afterEach, describe, expect, it } from "vitest";
import { FastMeshError, type LoopId, Scene, type Vec3 } from "../src/node.js";

// Two triangles on the XY plane facing +Z.
const quad = [0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0];
const quadTris = [0, 1, 2, 0, 2, 3];
// Two quads facing away across the z=0..1 gap (A faces -Z, B faces +Z).
const bridgePos = [
  0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
  0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
];
const bridgeIdx = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7];
// A quarter turn about +z, row-major.
const QUARTER_Z = [0, -1, 0, 1, 0, 0, 0, 0, 1];

function codeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected a throw");
}

/** Applies a column-major 4×4 matrix to a point. */
function transform(m: Float64Array, p: ArrayLike<number>): number[] {
  return [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
}

/**
 * The open upper half of a UV sphere of radius 1 around the origin, outward
 * winding: a pole vertex and `stacks` rings of `slices` vertices, the last on
 * the equator (the rim).
 */
function hemisphere(stacks: number, slices: number): { positions: number[]; triangles: number[][] } {
  const positions = [0, 0, 1];
  for (let i = 1; i <= stacks; i++) {
    const theta = (Math.PI / 2) * (i / stacks);
    for (let j = 0; j < slices; j++) {
      const phi = (2 * Math.PI * j) / slices;
      positions.push(Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta));
    }
  }
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const triangles: number[][] = [];
  for (let j = 0; j < slices; j++) {
    triangles.push([0, ring(1, j), ring(1, j + 1)]);
    for (let i = 1; i < stacks; i++) {
      const a = ring(i, j);
      const b = ring(i + 1, j);
      const c = ring(i + 1, j + 1);
      const d = ring(i, j + 1);
      triangles.push([a, b, c], [a, c, d]);
    }
  }
  return { positions, triangles };
}

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

describe("edit", () => {
  it("edit_ops_atomic_with_refs", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    s.takeChanges();
    // Two new vertices, a triangle on them by ref, then remove an old one.
    const r = s.edit(m, [
      { op: "addVertices", positions: [3, 0, 0, 3, 2, 0] },
      { op: "addTriangles", triangles: [1, { ref: [0, 0] }, { ref: [0, 1] }, 1, { ref: [0, 1] }, 2] },
      { op: "removeTriangles", triangles: [1] },
    ]);
    expect(r.added.length).toBe(3);
    expect(Array.from(r.added[0])).toEqual([4, 5]);
    expect(Array.from(r.added[1]).length).toBe(2);
    expect(r.added[2]).toEqual(new Uint32Array(0));
    expect(s.vertexCount(m)).toBe(6);
    expect(s.triangleCount(m)).toBe(3);
    const idx = s.views(m).indices();
    const t = r.added[1][0];
    expect(Array.from(idx.subarray(3 * t, 3 * t + 3))).toEqual([1, 4, 5]);
    expect(Array.from(idx.subarray(3, 6))).toEqual([0, 0, 0]);
    s.takeChanges();

    // A bad ref writes nothing: no vertices, no undo step, no change record.
    const canUndoBefore = s.canUndo();
    const slots = s.vertexSlots(m);
    const bad = [
      // Ref to a later op.
      [
        { op: "addTriangles", triangles: [{ ref: [1, 0] }, 1, 2] },
        { op: "addVertices", positions: [9, 9, 9] },
      ],
      // Index past the op's result.
      [
        { op: "addVertices", positions: [9, 9, 9] },
        { op: "addTriangles", triangles: [{ ref: [0, 1] }, 1, 2] },
      ],
      // A ref to triangles where vertices are wanted.
      [
        { op: "addVertices", positions: [9, 9, 9] },
        { op: "addTriangles", triangles: [0, 1, { ref: [0, 0] }] },
        { op: "removeVertices", vertices: [{ ref: [1, 0] }] },
      ],
      // A stale raw id after an op that already wrote.
      [
        { op: "addVertices", positions: [9, 9, 9] },
        { op: "removeTriangles", triangles: [99] },
      ],
      // Not an integer id.
      [{ op: "removeTriangles", triangles: [0.5] }],
    ] as const;
    for (const ops of bad) {
      expect(codeOf(() => s.edit(m, ops as never))).toBe("INVALID_INDEX");
      expect(s.vertexCount(m)).toBe(6);
      expect(s.vertexSlots(m)).toBe(slots);
      expect(s.triangleCount(m)).toBe(3);
      expect(s.takeChanges()).toEqual([]);
      expect(s.canUndo()).toBe(canUndoBefore);
    }
    // Malformed lists.
    // A hole in a sparse id list is not an id.
    // eslint-disable-next-line no-sparse-arrays
    expect(codeOf(() => s.edit(m, [{ op: "removeTriangles", triangles: [0, , 2] } as never]))).toBe("INVALID_ARGUMENT");
    expect(s.triangleCount(m)).toBe(3);
    expect(codeOf(() => s.edit(m, [{ op: "addVertices", positions: [1, 2] }]))).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(() => s.edit(m, [{ op: "setPositions", vertices: [0, 1], positions: [1, 2, 3] }])),
    ).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.edit(m, [{ op: "nope" } as never]))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.edit(m, [{ op: "addTriangles", triangles: [0, 1] }]))).toBe("INVALID_ARGUMENT");
    // A vertex still in use: the store's own check, rolled back.
    expect(codeOf(() => s.edit(m, [{ op: "removeVertices", vertices: [0] }]))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(canUndoBefore);
    // An empty list records nothing.
    expect(s.edit(m, []).added).toEqual([]);
    expect(s.takeChanges()).toEqual([]);
  });

  it("edit_is_one_undo_step", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    expect(s.canUndo()).toBe(false);
    const before = Float32Array.from(s.views(m).positions());
    s.edit(m, [
      { op: "setPositions", vertices: Uint32Array.of(2), positions: [2, 2, 1] },
      { op: "addVertices", positions: [3, 1, 0] },
      { op: "addTriangles", triangles: [1, { ref: [1, 0] }, 2] },
      { op: "removeTriangles", triangles: Uint32Array.of(0) },
    ]);
    expect(s.triangleCount(m)).toBe(2);
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(s.triangleCount(m)).toBe(2);
    expect(s.vertexCount(m)).toBe(4);
    expect(Array.from(s.views(m).positions().subarray(0, 12))).toEqual(Array.from(before));
    expect(Array.from(s.views(m).indices().subarray(0, 6))).toEqual(quadTris);
    expect(s.redo()).toBe(true);
    expect(s.vertexCount(m)).toBe(5);
    // Inside a caller's step the edit joins it.
    s.beginStep();
    s.edit(m, [{ op: "setPositions", vertices: [0], positions: [0, 0, 1] }]);
    s.edit(m, [{ op: "setPositions", vertices: [1], positions: [2, 0, 1] }]);
    s.endStep();
    s.undo();
    expect(s.views(m).positions()[2]).toBeCloseTo(0, 6);
    expect(s.views(m).positions()[5]).toBeCloseTo(0, 6);
  });

  it("set_positions_world_space", () => {
    const s = scene();
    // Origin (bbox centre) at (1, 1, 0); quarter turn about z, then shift.
    const m = s.addMesh(quad, quadTris);
    s.setPlacement(m, { rotation: QUARTER_Z, translation: [10, 20, 30] });
    s.takeChanges();
    const n0 = Array.from(s.views(m).normals()!.subarray(6, 9));
    expect(n0[2]).toBeCloseTo(1, 6);
    // Lift vertex 2 by 1 in world z and move it in world xy.
    const target: Vec3 = [5, 25, 31];
    s.edit(m, [{ op: "setPositions", vertices: [2], positions: target }]);
    const p = s.views(m).positions().subarray(6, 9);
    // local = Rᵀ (q − t) − origin = Rᵀ (−5, 5, 1) − (1, 1, 0) = (5, 5, 1) − (1, 1, 0).
    expect(Array.from(p)).toEqual([4, 4, 1]);
    // The world matrix maps it back.
    const w = transform(s.matrix4(m), p);
    for (let k = 0; k < 3; k++) {
      expect(w[k]).toBeCloseTo(target[k], 9);
    }
    // Normals followed (vertex 2 is no longer flat).
    const n1 = s.views(m).normals()!.subarray(6, 9);
    expect(Math.abs(n1[2] - 1)).toBeGreaterThan(1e-3);
    const change = s.takeChanges()[0];
    expect(change.positions).toEqual([[2, 1]]);
    expect(change.normals).not.toEqual([]);
    // addVertices too.
    const r = s.edit(m, [{ op: "addVertices", positions: Float64Array.of(10, 20, 30) }]);
    const v = r.added[0][0];
    expect(Array.from(s.views(m).positions().subarray(3 * v, 3 * v + 3))).toEqual([-1, -1, 0]);
    // A position that does not fit f32 locally.
    expect(
      codeOf(() => s.edit(m, [{ op: "addVertices", positions: [1e300, 0, 0] }])),
    ).toBe("INVALID_ARGUMENT");
  });

  it("compact_reports_all_and_capacity", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    s.edit(m, [
      { op: "removeTriangles", triangles: [1] },
      { op: "removeVertices", vertices: [3] },
    ]);
    s.takeChanges();
    expect(s.vertexSlots(m)).toBe(4);
    expect(s.canUndo()).toBe(true);
    s.compact(m);
    expect(s.vertexSlots(m)).toBe(3);
    expect(s.triangleSlots(m)).toBe(1);
    const [c] = s.takeChanges();
    expect(c.mesh).toBe(m);
    expect(c.capacityChanged).toBe(true);
    expect(c.positions).toBe("all");
    expect(c.normals).toBe("all");
    expect(c.indices).toBe("all");
    expect(s.views(m).positions().length).toBe(9);
    expect(Array.from(s.views(m).indices())).toEqual([0, 1, 2]);
    // The renumbered mesh's edits leave the history.
    expect(s.canUndo()).toBe(false);
  });

  it("growth_reports_every_buffer_all", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    s.takeChanges();
    s.edit(m, [
      { op: "addVertices", positions: [4, 0, 0] },
      { op: "addTriangles", triangles: [1, { ref: [0, 0] }, 2] },
    ]);
    const [c] = s.takeChanges();
    expect(c.capacityChanged).toBe(true);
    expect([c.positions, c.normals, c.colors, c.indices]).toEqual(["all", "all", "all", "all"]);
  });

  it("slots_added_and_undone_report_capacity", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris, { normals: false });
    s.takeChanges();
    s.edit(m, [{ op: "addVertices", positions: [4, 0, 0, 5, 0, 0] }]);
    s.undo();
    expect(s.vertexSlots(m)).toBe(6);
    // The edit cancelled out, but the buffers still grew.
    const [c, ...rest] = s.takeChanges();
    expect(rest).toEqual([]);
    expect(c.mesh).toBe(m);
    expect(c.capacityChanged).toBe(true);
    expect([c.positions, c.normals, c.colors, c.indices]).toEqual(["all", "all", "all", "all"]);
    expect(s.views(m).positions().length).toBe(18);
    expect(s.takeChanges()).toEqual([]);
  });
});

/** Hemisphere with two punched holes (vertex fans removed) and two flipped triangles. */
function damagedHemisphere() {
  const stacks = 8;
  const slices = 32;
  const { positions, triangles } = hemisphere(stacks, slices);
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const holes = [ring(3, 4), ring(5, 20)];
  const kept = triangles.filter((t) => !t.some((v) => holes.includes(v)));
  // Flip two triangles far from the holes and the rim.
  for (const k of [40, 200]) {
    const [a, b, c] = kept[k];
    kept[k] = [a, c, b];
  }
  return { positions, indices: kept.flat() };
}

/** 3x3 flat grid in z=0, spacing 1, centered at the origin; vertex (x, y) has id y*3+x, wound CCW from +Z. */
function flatGrid(): { positions: number[]; indices: number[] } {
  const positions: number[] = [];
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < 3; x++) {
      positions.push(x - 1, y - 1, 0);
    }
  }
  const indices: number[] = [];
  for (let y = 0; y < 2; y++) {
    for (let x = 0; x < 2; x++) {
      const a = y * 3 + x;
      indices.push(a, a + 1, a + 4, a, a + 4, a + 3);
    }
  }
  return { positions, indices };
}

/** Square pyramid with a closed base; apex id 4 at (0.5, 0.5, 1). */
function pyramid(): { positions: number[]; indices: number[] } {
  const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1];
  const indices = [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 0, 2, 1, 0, 3, 2];
  return { positions, indices };
}

describe("analyze and repair", () => {
  it("auto_repair_is_the_op_list_in_one_call", () => {
    const { positions, indices } = damagedHemisphere();
    const a = scene();
    const m = a.addMesh(positions, indices);
    const triangles = a.triangleCount(m);
    const report = a.repair(m, { kind: "auto" });
    // The library's order and defaults.
    expect(Object.keys(report)).toEqual(["weld", "removeDegenerate", "orient", "removeSmallComponents", "fillHoles"]);
    const b = scene();
    const n = b.addMesh(positions, indices);
    const [weld, removeDegenerate, orient, removeSmallComponents, fillHoles] = b.repair(n, [
      { op: "weld", tolerance: 1e-5 },
      { op: "removeDegenerate" },
      { op: "orient" },
      { op: "removeSmallComponents", minTriangles: 500 },
      { op: "fillHoles", maxEdges: 64 },
    ]);
    expect(report).toEqual({ weld, removeDegenerate, orient, removeSmallComponents, fillHoles });
    expect(orient.flippedTriangles).toBe(2);
    expect(Array.from(a.views(m).positions())).toEqual(Array.from(b.views(n).positions()));
    expect(Array.from(a.views(m).indices())).toEqual(Array.from(b.views(n).indices()));

    // One undo step.
    expect(a.undo()).toBe(true);
    expect(a.canUndo()).toBe(false);
    expect(a.triangleCount(m)).toBe(triangles);
  });

  it("auto_repair_passes_its_options_through", () => {
    const { positions, indices } = damagedHemisphere();
    const s = scene();
    const m = s.addMesh(positions, indices);
    // Past every hole's bound: filled nothing, skipped all three.
    const none = s.repair(m, { kind: "auto", maxEdges: 3 }).fillHoles;
    expect([none.filled.length, none.skipped.length]).toEqual([0, 3]);
    s.undo();
    // Below the rim's size, then any size.
    const small = s.repair(m, { kind: "auto", maxEdges: 16 }).fillHoles;
    expect([small.filled.length, small.skipped.length]).toEqual([2, 1]);
    s.undo();
    const any = s.repair(m, { kind: "auto", maxEdges: Infinity, fair: false }).fillHoles;
    expect(any.skipped).toEqual([]);
    expect(any.filled.length + any.failed.length).toBe(3);
  });

  it("auto_repair_rejects_bad_options_and_changes_nothing", () => {
    const { positions, indices } = damagedHemisphere();
    const s = scene();
    const m = s.addMesh(positions, indices);
    const triangles = s.triangleCount(m);
    for (const bad of [
      { weldTolerance: -1 },
      { weldTolerance: Number.NaN },
      { minTriangles: 1.5 },
      { minTriangles: -1 },
      { maxEdges: 1.5 },
      { maxEdges: -1 },
      { maxEdges: Number.NaN },
    ]) {
      expect(codeOf(() => s.repair(m, { kind: "auto", ...bad }))).toBe("INVALID_ARGUMENT");
    }
    expect(s.canUndo()).toBe(false);
    expect(s.triangleCount(m)).toBe(triangles);
    // Only the auto request has a `kind`; other values are not ops either.
    expect(codeOf(() => s.repair(m, { kind: "melt" } as never))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, [{ kind: "auto" }] as never))).toBe("INVALID_ARGUMENT");
  });

  it("repair_pipeline_one_undo_step", () => {
    const s = scene();
    const { positions, indices } = damagedHemisphere();
    const m = s.addMesh(positions, indices);
    const before = s.analyze(m);
    expect(before.loops.length).toBe(3);
    expect(before.orientationConflicts).toBe(6);
    expect(before.components.length).toBe(1);
    expect(before.components[0].closed).toBe(false);
    expect(before.selfIntersects).toBeNull();
    expect(typeof before.loops[0].id.hash).toBe("bigint");
    expect(before.loops[0].vertices).toBeInstanceOf(Uint32Array);
    const rim = before.loops.find((l) => l.perimeter > 6)!;
    expect(rim.vertices.length).toBe(32);
    const triangles = s.triangleCount(m);

    const [orient, fill] = s.repair(m, [{ op: "orient" }, { op: "fillHoles", maxPerimeter: 2 }]);
    expect(orient.flippedTriangles).toBe(2);
    expect(orient.patches).toBeGreaterThan(0);
    expect(orient.nonOrientablePatches).toBe(0);
    expect(fill.filled.length).toBe(2);
    expect(fill.skipped).toEqual([{ id: rim.id, reason: "bounds" }]);
    expect(fill.failed).toEqual([]);
    for (const f of fill.filled) {
      expect(f.trianglesAdded).toBeGreaterThanOrEqual(4);
      expect(f.converged).toBe(true);
      expect(f.refineCapped).toBe(false);
      // Fairing runs only when refinement added vertices.
      if (f.verticesAdded === 0) {
        expect(f.weights).toBeNull();
      } else {
        expect(f.weights === "cotangent" || f.weights === "uniform").toBe(true);
      }
    }

    const after = s.analyze(m, { selfIntersection: true, maxListed: 5 });
    expect(after.orientationConflicts).toBe(0);
    expect(after.loops.length).toBe(1);
    expect(after.loops[0].id).toEqual(rim.id);
    expect(after.loops[0].clean).toBe(true);
    expect(after.selfIntersects).toBe(false);
    expect(after.nonManifoldEdges).toBe(0);

    // The whole pipeline is one undo step.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(s.triangleCount(m)).toBe(triangles);
    expect(s.analyze(m).orientationConflicts).toBe(6);

    // A single op returns its report, not an array.
    const one = s.repair(m, { op: "orient" });
    expect(one.flippedTriangles).toBe(2);
    expect(
      s.repair(m, { op: "removeDegenerate" }),
    ).toEqual({ removedTriangles: 0, removedVertices: 2 });
    expect(
      s.repair(m, { op: "removeSmallComponents", minTriangles: 10 }),
    ).toEqual({ removedComponents: 0, removedTriangles: 0, removedVertices: 0 });
    expect(s.repair(m, { op: "weld", tolerance: 0 })).toEqual({
      mergedVertices: 0,
      removedTriangles: 0,
    });
    // The new fill options round-trip; a generous budget caps nothing.
    const options = s.repair(m, {
      op: "fillHoles",
      maxPerimeter: 2,
      maxDihedral: Math.PI,
      maxRefineVertices: 1000,
    });
    expect(options.filled.length).toBe(2);
    expect(options.filled.every((f) => f.refineCapped === false)).toBe(true);
    expect(s.undo()).toBe(true);
    // Bad options: nothing changes.
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, [{ op: "orient" }, { op: "fillHoles" }]))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "weld", tolerance: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "fillHoles", maxEdges: 1.5 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "fillHoles", maxPerimeter: Number.NaN }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.repair(m, { op: "fillHoles", maxDihedral: Number.NaN }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.repair(m, { op: "fillHoles", maxRefineVertices: 1.5 }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.repair(m, { op: "melt" } as never))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.analyze(m, { weldTolerance: -1 }))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
  });

  it("inflate_moves_brushed_vertices_along_normal", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    expect(s.canUndo()).toBe(false);
    const before = Float32Array.from(s.views(m).positions());
    // Brush center in local coordinates: the middle vertex (id 4).
    const center: Vec3 = [before[12], before[13], before[14]];
    const r = s.repair(m, { op: "inflate", center, radius: 0.6, falloff: "linear", strength: 2 });
    expect(r.moved).toBe(1);
    expect(r.maxDisplacement).toBeCloseTo(2, 4);
    const after = s.views(m).positions();
    // The brushed vertex moved along z only (a flat grid's smooth normal is
    // ±z); every other vertex is bit-identical.
    expect(after[12] - before[12]).toBeCloseTo(0, 6);
    expect(after[13] - before[13]).toBeCloseTo(0, 6);
    expect(Math.abs(after[14] - before[14])).toBeCloseTo(2, 4);
    for (const v of [0, 1, 2, 3, 5, 6, 7, 8]) {
      expect(Array.from(after.subarray(3 * v, 3 * v + 3))).toEqual(
        Array.from(before.subarray(3 * v, 3 * v + 3)),
      );
    }
    // One undo entry for the call.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
    // The "smooth" falloff wires through too (weight 1 at the center).
    const r2 = s.repair(m, { op: "inflate", center, radius: 0.6, falloff: "smooth", strength: 2 });
    expect(r2.moved).toBe(1);
    expect(r2.maxDisplacement).toBeCloseTo(2, 4);
  });

  it("inflate_rejects_invalid_brush", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const center: Vec3 = [before[12], before[13], before[14]];
    const undoable = s.canUndo();
    expect(
      codeOf(() => s.repair(m, { op: "inflate", center, radius: -1, falloff: "linear", strength: 1 })),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(() => s.repair(m, { op: "inflate", center, radius: 0, falloff: "linear", strength: 1 })),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(
        () => s.repair(m, { op: "inflate", center, radius: 0.6, falloff: "linear", strength: NaN }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(
        () => s.repair(m, { op: "inflate", center, radius: NaN, falloff: "linear", strength: 1 }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(
        () =>
          s.repair(m, { op: "inflate", center, radius: 0.6, falloff: "cubic" as never, strength: 1 }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(
        () =>
          s.repair(m, {
            op: "inflate",
            center: [0, 0] as never,
            radius: 0.6,
            falloff: "linear",
            strength: 1,
          }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("smooth_reduces_spike", () => {
    const s = scene();
    const { positions, indices } = pyramid();
    const m = s.addMesh(positions, indices);
    expect(s.canUndo()).toBe(false);
    const before = Float32Array.from(s.views(m).positions());
    const apex: Vec3 = [before[12], before[13], before[14]];
    const distBefore = apex[2] - before[2];
    expect(distBefore).toBeCloseTo(1, 6);
    const r = s.repair(m, {
      op: "smooth",
      center: apex,
      radius: 0.6,
      falloff: "smooth",
      strength: 1,
      cycles: 1,
    });
    expect(r.moved).toBe(1);
    const after = Float32Array.from(s.views(m).positions());
    const distAfter = after[14] - after[2];
    expect(distAfter).toBeLessThan(distBefore);
    expect(distAfter).toBeCloseTo(0.8978, 3);
    expect(r.maxDisplacement).toBeCloseTo(distBefore - distAfter, 4);
    for (const v of [0, 1, 2, 3]) {
      expect(Array.from(after.subarray(3 * v, 3 * v + 3))).toEqual(
        Array.from(before.subarray(3 * v, 3 * v + 3)),
      );
    }
    // One undo entry for the call.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
    // Zero cycles move nothing.
    const r0 = s.repair(m, {
      op: "smooth",
      center: apex,
      radius: 0.6,
      falloff: "smooth",
      strength: 1,
      cycles: 0,
    });
    expect(r0).toEqual({ moved: 0, maxDisplacement: 0 });
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("smooth_rejects_invalid_brush", () => {
    const s = scene();
    const { positions, indices } = pyramid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const apex: Vec3 = [before[12], before[13], before[14]];
    const good = { op: "smooth", center: apex, radius: 0.6, falloff: "smooth", strength: 1, cycles: 1 } as const;
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { ...good, radius: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, strength: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, cycles: 1.5 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, cycles: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, cycles: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, cycles: 2 ** 32 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, falloff: "cubic" as never }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("grab_drags_region_along_direction", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    expect(s.canUndo()).toBe(false);
    const before = Float32Array.from(s.views(m).positions());
    // Brush center in local coordinates: the middle vertex (id 4).
    const center: Vec3 = [before[12], before[13], before[14]];
    const r = s.repair(m, {
      op: "grab",
      center,
      radius: 0.6,
      falloff: "linear",
      strength: 0.5,
      direction: [2, 0, 0],
    });
    expect(r.moved).toBe(1);
    expect(r.maxDisplacement).toBeCloseTo(1, 4);
    const after = s.views(m).positions();
    // Strength times direction, exactly; every other vertex is bit-identical.
    expect(after[12] - before[12]).toBeCloseTo(1, 4);
    expect(after[13] - before[13]).toBeCloseTo(0, 6);
    expect(after[14] - before[14]).toBeCloseTo(0, 6);
    for (const v of [0, 1, 2, 3, 5, 6, 7, 8]) {
      expect(Array.from(after.subarray(3 * v, 3 * v + 3))).toEqual(
        Array.from(before.subarray(3 * v, 3 * v + 3)),
      );
    }
    // One undo entry, restoring bit-identical positions.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
    // A zero direction moves nothing and writes nothing.
    const r0 = s.repair(m, {
      op: "grab",
      center,
      radius: 0.6,
      falloff: "linear",
      strength: 1,
      direction: [0, 0, 0],
    });
    expect(r0).toEqual({ moved: 0, maxDisplacement: 0 });
    expect(s.canUndo()).toBe(false);
  });

  it("grab_rejects_invalid_brush", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const center: Vec3 = [before[12], before[13], before[14]];
    const good = {
      op: "grab",
      center,
      radius: 0.6,
      falloff: "linear",
      strength: 1,
      direction: [1, 0, 0],
    } as const;
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { ...good, radius: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, radius: 0 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, strength: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, center: [0, 0] as never }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, falloff: "cubic" as never }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, direction: [0, 0] as never }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, direction: [1, NaN, 0] as never }))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("relax_melts_a_spike_laplacian", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const center: Vec3 = [before[12], before[13], before[14]];
    // Raise a spike, then relax it back down (re-centering each pass, the
    // way a brush tracks the surface).
    s.repair(m, { op: "grab", center, radius: 0.6, falloff: "linear", strength: 1, direction: [0, 0, 2] });
    const raised = Float32Array.from(s.views(m).positions());
    expect(raised[14] - before[14]).toBeCloseTo(2, 4);
    const mid = (): Vec3 => {
      const p = s.views(m).positions();
      return [p[12], p[13], p[14]];
    };
    const maxZ = (): number => {
      const p = s.views(m).positions();
      let max = -Infinity;
      for (let i = 2; i < p.length; i += 3) {
        if (p[i] > max) {
          max = p[i];
        }
      }
      return max;
    };
    const r = s.repair(m, { op: "relax", center: mid(), radius: 0.6, falloff: "linear", strength: 0.5, cycles: 1 });
    expect(r.moved).toBeGreaterThan(0);
    expect(r.maxDisplacement).toBeGreaterThan(0);
    const melted = maxZ();
    expect(melted).toBeLessThan(raised[14]);
    // And it keeps melting: a second pass drops the peak further still.
    s.repair(m, { op: "relax", center: mid(), radius: 0.6, falloff: "linear", strength: 0.5, cycles: 1 });
    expect(maxZ()).toBeLessThan(melted);
  });

  it("protect_holds_listed_vertices_byte_identical_and_moves_the_rest", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const center: Vec3 = [before[12], before[13], before[14]];
    const held = [0, 1, 4];
    const brush = { center, radius: 2, falloff: "linear", strength: 0.2 } as const;
    const report = s.repair(m, { op: "inflate", ...brush, protect: { vertices: held } });
    expect(report.moved).toBeGreaterThan(0);
    const after = s.views(m).positions();
    for (const v of held) {
      expect(Array.from(after.subarray(3 * v, 3 * v + 3))).toEqual(Array.from(before.subarray(3 * v, 3 * v + 3)));
    }
    expect(Array.from(after)).not.toEqual(Array.from(before));
    // One undo step, and it restores everything.
    s.undo();
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("protect_scales_a_move_by_its_weight", () => {
    const run = (weight: number | null) => {
      const s = scene();
      const { positions, indices } = flatGrid();
      const m = s.addMesh(positions, indices);
      const p = s.views(m).positions();
      const center: Vec3 = [p[12], p[13], p[14]];
      const op = { op: "grab", center, radius: 3, falloff: "linear", strength: 1, direction: [0, 0, 1] } as const;
      s.repair(m, weight === null ? op : { ...op, protect: { vertices: [4], weights: [weight] } });
      return s.views(m).positions()[14];
    };
    expect(run(0.5)).toBeCloseTo((run(null) + run(0)) * 0.5, 5);
    expect(run(1)).toBe(run(null));
    expect(run(0)).not.toBe(run(null));
  });

  it("protect_rejects_bad_input_and_non_brush_ops", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const center: Vec3 = [before[12], before[13], before[14]];
    const good = { op: "inflate", center, radius: 2, falloff: "linear", strength: 0.2 } as const;
    expect(codeOf(() => s.repair(m, { ...good, protect: { vertices: [99999] } }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, protect: { vertices: [0.5] } }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, protect: { vertices: [0], weights: [2] } }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, protect: { vertices: [0], weights: [] } }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "orient", protect: { vertices: [0] } } as never))).toBe("INVALID_ARGUMENT");
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("relax_rejects_invalid_brush", () => {
    const s = scene();
    const { positions, indices } = flatGrid();
    const m = s.addMesh(positions, indices);
    const before = Float32Array.from(s.views(m).positions());
    const center: Vec3 = [before[12], before[13], before[14]];
    const good = { op: "relax", center, radius: 0.6, falloff: "linear", strength: 1, cycles: 1 } as const;
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { ...good, radius: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, strength: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, center: [0, 0] as never }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, falloff: "cubic" as never }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, cycles: -1 }))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("remesh_refines_sparse_region", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    expect(s.canUndo()).toBe(false);
    const beforePos = Float32Array.from(s.views(m).positions());
    const beforeIdx = Uint32Array.from(s.views(m).indices());
    const r = s.repair(m, { op: "remesh", center: [1, 1, 0], radius: 10, targetLength: 0.5 });
    expect(r.splits).toBeGreaterThan(0);
    expect(s.views(m).indices().length).toBeGreaterThan(beforeIdx.length);
    // The original corners are untouched; new vertices append past them.
    expect(Array.from(s.views(m).positions().subarray(0, 12))).toEqual(Array.from(beforePos));
    // One undo entry for the call, restoring positions and topology (dead
    // slots past the original vertices read back as zeros).
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions().subarray(0, beforePos.length))).toEqual(
      Array.from(beforePos),
    );
    expect(Array.from(s.views(m).indices().subarray(0, beforeIdx.length))).toEqual(
      Array.from(beforeIdx),
    );
  });

  it("remesh_split_only_never_collapses", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    const beforeIdx = Uint32Array.from(s.views(m).indices());
    const r = s.repair(m, { op: "remesh", center: [1, 1, 0], radius: 10, targetLength: 0.5, splitOnly: true });
    expect(r.splits).toBeGreaterThan(0);
    expect(r.collapses).toBe(0);
    expect(r.flips).toBe(0);
    expect(s.views(m).indices().length).toBeGreaterThan(beforeIdx.length);
    expect(s.undo()).toBe(true);
    expect(Array.from(s.views(m).indices().subarray(0, beforeIdx.length))).toEqual(
      Array.from(beforeIdx),
    );
  });

  it("splitStretched_leaves_everything_it_did_not_stretch_identical", () => {
    // A grid with cells from nearly nothing to coarse: a mean length fits neither end.
    const n = 30;
    const warp = (v: number) => n * (v / n) ** 3;
    const pos: number[] = [];
    const tris: number[] = [];
    for (let y = 0; y <= n; y++) {
      for (let x = 0; x <= n; x++) {
        pos.push(warp(x), warp(y), 0);
      }
    }
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x;
        tris.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
      }
    }
    const s = scene();
    const m = s.addMesh(Float32Array.from(pos), Uint32Array.from(tris));
    const center = [15, 15, 0] as const;
    expect(() => s.repair(m, { op: "splitStretched", center, radius: 100 })).toThrow();
    expect(s.repair(m, { op: "captureRestSizes" })).toEqual({});
    expect(s.canUndo()).toBe(false);
    expect(() => s.repair(m, [{ op: "orient" }, { op: "captureRestSizes" }])).toThrow();

    // Nothing stretched yet: not one split, however coarse the far corner is.
    expect(s.repair(m, { op: "splitStretched", center, radius: 100 })).toEqual({ splits: 0, remaining: 0, vertices: new Uint32Array() });
    expect(s.canUndo()).toBe(false);

    s.repair(m, { op: "grab", center, radius: 5, falloff: "smooth", strength: 6, direction: [0, 0, 1] });
    const dabbed = Float32Array.from(s.views(m).positions());
    const before = Uint32Array.from(s.views(m).indices());
    const moved = (v: number) => dabbed[3 * v + 2] !== 0;

    const r = s.repair(m, { op: "splitStretched", center, radius: 100 });
    expect(r.splits).toBeGreaterThan(0);
    expect(r.remaining).toBe(0);

    const positions = s.views(m).positions();
    const indices = s.views(m).indices();
    expect(Array.from(positions.subarray(0, dabbed.length))).toEqual(Array.from(dabbed));
    expect(indices.length).toBeGreaterThan(before.length);
    let unstretched = 0;
    for (let t = 0; t < before.length; t += 3) {
      if (!moved(before[t]) && !moved(before[t + 1]) && !moved(before[t + 2])) {
        unstretched++;
        expect(Array.from(indices.subarray(t, t + 3))).toEqual(Array.from(before.subarray(t, t + 3)));
      }
    }
    expect(unstretched).toBeGreaterThan(1000);

    expect(s.undo()).toBe(true);
    expect(Array.from(s.views(m).indices().subarray(0, before.length))).toEqual(Array.from(before));
    expect(Array.from(s.views(m).positions().subarray(0, dabbed.length))).toEqual(Array.from(dabbed));
    expect(s.redo()).toBe(true);
    expect(s.repair(m, { op: "splitStretched", center, radius: 100 }).splits).toBe(0);
  });

  it("splitStretched_needs_new_rest_sizes_after_an_edit_outdates_them", () => {
    const n = 20;
    const pos: number[] = [];
    const tris: number[] = [];
    for (let y = 0; y <= n; y++) {
      for (let x = 0; x <= n; x++) {
        pos.push(x, y, 0);
      }
    }
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x;
        tris.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
      }
    }
    const s = scene();
    const m = s.addMesh(Float32Array.from(pos), Uint32Array.from(tris));
    const center = [0, 0, 0] as const;
    const pass = () => s.repair(m, { op: "splitStretched", center, radius: 1000 });
    s.repair(m, { op: "captureRestSizes", onlyIfMissing: true });
    expect(pass().splits).toBe(0);
    // A simplify outdates them: longer edges must not read as stretched.
    s.repair(m, { op: "simplify", targetTriangles: 200 });
    expect(pass).toThrow();
    s.repair(m, { op: "captureRestSizes", onlyIfMissing: true });
    expect(pass().splits).toBe(0);
    // Undoing the simplify brings back the sizes from before it, still true.
    expect(s.undo()).toBe(true);
    expect(pass().splits).toBe(0);

    // Undoing an edit from before the capture forgets it.
    const fresh = s.addMesh(Float32Array.from(pos), Uint32Array.from(tris));
    const passFresh = () => s.repair(fresh, { op: "splitStretched", center, radius: 1000 });
    s.repair(fresh, { op: "simplify", targetTriangles: 200 });
    s.repair(fresh, { op: "captureRestSizes", onlyIfMissing: true });
    expect(passFresh().splits).toBe(0);
    expect(s.undo()).toBe(true);
    expect(passFresh).toThrow();
    s.repair(fresh, { op: "captureRestSizes", onlyIfMissing: true });
    expect(passFresh().splits).toBe(0);
  });

  it("bridgeLoops_joins_two_rims", () => {
    const s = scene();
    const m = s.addMesh(bridgePos, bridgeIdx);
    const loops = s.analyze(m).loops;
    expect(loops.length).toBe(2);
    const ids: LoopId[] = loops.map((l) => l.id);
    const beforePos = Float32Array.from(s.views(m).positions());
    const beforeIdx = Uint32Array.from(s.views(m).indices());
    const r = s.repair(m, { op: "bridgeLoops", loops: [ids[0], ids[1]] });
    expect(r.trianglesAdded).toBe(8);
    expect(s.views(m).indices().length).toBe(beforeIdx.length + 24);
    // One undo entry for the call, restoring positions and topology (dead
    // slots past the originals read back as zeros).
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions().subarray(0, beforePos.length))).toEqual(
      Array.from(beforePos),
    );
    expect(Array.from(s.views(m).indices().subarray(0, beforeIdx.length))).toEqual(
      Array.from(beforeIdx),
    );
  });

  it("remesh_rejects_invalid_region", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    const beforePos = Float32Array.from(s.views(m).positions());
    const beforeIdx = Uint32Array.from(s.views(m).indices());
    const good = { op: "remesh", center: [1, 1, 0], radius: 10, targetLength: 0.5 } as const;
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { ...good, radius: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, radius: 0 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, radius: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, center: [0, 0] as never }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.repair(m, { ...good, targetLength: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, maxRounds: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, maxRounds: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, maxRounds: 1.5 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, maxRounds: 2 ** 32 }))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(beforePos));
    expect(Array.from(s.views(m).indices())).toEqual(Array.from(beforeIdx));
  });

  it("bridgeLoops_rejects_bad_selection", () => {
    const s = scene();
    const m = s.addMesh(bridgePos, bridgeIdx);
    const loops = s.analyze(m).loops;
    expect(loops.length).toBe(2);
    const ids: LoopId[] = loops.map((l) => l.id);
    const beforePos = Float32Array.from(s.views(m).positions());
    const beforeIdx = Uint32Array.from(s.views(m).indices());
    const good = { op: "bridgeLoops", loops: [ids[0], ids[1]] } as const;
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { op: "bridgeLoops", loops: [ids[0], ids[0]] }))).toBe(
      "INVALID_ARGUMENT",
    );
    const fake = { ...ids[1], hash: ids[1].hash ^ 1n };
    expect(codeOf(() => s.repair(m, { op: "bridgeLoops", loops: [ids[0], fake] }))).toBe(
      "STALE_LOOPS",
    );
    expect(codeOf(() => s.repair(m, { op: "bridgeLoops", loops: [ids[0]] as never }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.repair(m, { op: "bridgeLoops", loops: "nope" as never }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.repair(m, { ...good, maxEdges: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, maxEdges: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { ...good, maxEdges: 1.5 }))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(beforePos));
    expect(Array.from(s.views(m).indices())).toEqual(Array.from(beforeIdx));
  });

  it("join_merges_source_into_target", () => {
    const s = scene();
    const a = s.addMesh(bridgePos.slice(0, 12), bridgeIdx.slice(0, 6));
    const b = s.addMesh(
      bridgePos.slice(12),
      bridgeIdx.slice(6).map((v) => v - 4),
    );
    const loopA = s.analyze(a).loops[0].id;
    const loopB = s.analyze(b).loops[0].id;
    const r = s.join(a, b, { loopA, loopB });
    expect(r).toEqual({ verticesAdded: 4, trianglesAdded: 2, bridgeTrianglesAdded: 8 });
    expect(s.triangleCount(a)).toBe(2 + 2 + 8);
    expect(codeOf(() => s.triangleCount(b))).toBe("UNKNOWN_MESH");
    // One undo step restores the target; the source stays removed.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(s.triangleCount(a)).toBe(2);
    expect(codeOf(() => s.triangleCount(b))).toBe("UNKNOWN_MESH");
    expect(s.redo()).toBe(true);
    expect(s.triangleCount(a)).toBe(2 + 2 + 8);
  });

  it("join_rejects_bad_selection", () => {
    const s = scene();
    const a = s.addMesh(bridgePos.slice(0, 12), bridgeIdx.slice(0, 6));
    const b = s.addMesh(
      bridgePos.slice(12),
      bridgeIdx.slice(6).map((v) => v - 4),
    );
    const loopA = s.analyze(a).loops[0].id;
    const loopB = s.analyze(b).loops[0].id;
    const good = { loopA, loopB };
    expect(codeOf(() => s.join(a, a, good))).toBe("INVALID_ARGUMENT");
    const fake = { ...loopB, hash: loopB.hash ^ 1n };
    expect(codeOf(() => s.join(a, b, { loopA, loopB: fake }))).toBe("STALE_LOOPS");
    expect(codeOf(() => s.join(a, b, { loopA: fake, loopB }))).toBe("STALE_LOOPS");
    expect(codeOf(() => s.join(a, b, { ...good, maxEdges: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.join(a, b, { loopA, loopB: "nope" as never }))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(false);
    expect(s.triangleCount(a)).toBe(2);
    expect(s.triangleCount(b)).toBe(2);
  });

  it("stale_loops_error_code", () => {    const s = scene();
    const { positions, indices } = damagedHemisphere();
    const m = s.addMesh(positions, indices, { mode: "manual" });
    s.repair(m, { op: "orient" });
    const holes = s.analyze(m).loops.filter((l) => l.perimeter < 6);
    expect(holes.length).toBe(2);
    const ids: LoopId[] = holes.map((l) => l.id);
    const r = s.repair(m, { op: "fillHoles", loops: [ids[0]], fair: false });
    expect(r.filled.map((f) => f.id)).toEqual([ids[0]]);
    expect(s.canUndo()).toBe(true);
    s.clearHistory();
    // The filled loop is gone: its id no longer matches.
    expect(codeOf(() => s.repair(m, { op: "fillHoles", loops: [ids[0]] }))).toBe("STALE_LOOPS");
    // A made-up hash too, and nothing was recorded.
    const fake = { ...ids[1], hash: ids[1].hash ^ 1n };
    expect(codeOf(() => s.repair(m, { op: "fillHoles", loops: [ids[1], fake] }))).toBe("STALE_LOOPS");
    expect(s.canUndo()).toBe(false);
    expect(codeOf(() => s.repair(m, { op: "fillHoles", loops: [{ ...fake, hash: -1n }] }))).toBe(
      "INVALID_ARGUMENT",
    );
    // A Manual mesh's stale BVH cannot check self-intersection.
    expect(codeOf(() => s.analyze(m, { selfIntersection: true }))).toBe("STALE_BVH");
    s.update(m);
    expect(s.analyze(m, { selfIntersection: true }).selfIntersects).toBe(false);
  });

  it("offset_moves_along_normals", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    expect(s.canUndo()).toBe(false);
    const before = Float32Array.from(s.views(m).positions());
    const r = s.repair(m, { op: "offset", distance: 0.5 });
    expect(r).toEqual({ moved: 4, maxDisplacement: 0.5, degenerateRemoved: 0, verticesRemoved: 0 });
    const after = s.views(m).positions();
    // A flat quad wound from +z: every vertex moves exactly 0.5 up.
    for (let v = 0; v < 4; v++) {
      expect(after[3 * v] - before[3 * v]).toBeCloseTo(0, 6);
      expect(after[3 * v + 1] - before[3 * v + 1]).toBeCloseTo(0, 6);
      expect(after[3 * v + 2] - before[3 * v + 2]).toBeCloseTo(0.5, 6);
    }
    // One undo entry for the call, restoring bit-identical positions.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
    // A negative distance sinks back below the start.
    const r2 = s.repair(m, { op: "offset", distance: -1 });
    expect(r2.moved).toBe(4);
    expect(r2.maxDisplacement).toBeCloseTo(1, 6);
    expect(s.views(m).positions()[2]).toBeCloseTo(before[2] - 1, 6);
    expect(s.canUndo()).toBe(true);
  });

  it("offset_rejects_bad_distance", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    const before = Float32Array.from(s.views(m).positions());
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { op: "offset", distance: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "offset", distance: Infinity }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "offset" } as never))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "offset", distance: "0.5" as never }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });

  it("shell_closes_an_open_mesh", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    const beforePos = Float32Array.from(s.views(m).positions());
    const r = s.repair(m, { op: "shell", thickness: 0.5 });
    expect(r).toEqual({
      verticesAdded: 4,
      trianglesAdded: 2 + 8,
      loopsClosed: 1,
      degenerateRemoved: 0,
      verticesRemoved: 0,
    });
    expect(s.triangleCount(m)).toBe(2 + 2 + 8);
    expect(s.analyze(m).loops).toEqual([]);
    // The outer sheet stayed; the copy sits one thickness below it.
    const after = s.views(m).positions();
    expect(Array.from(after.subarray(0, beforePos.length))).toEqual(Array.from(beforePos));
    for (let v = 0; v < 4; v++) {
      expect(after[(v + 4) * 3 + 2]).toBeCloseTo(-0.5, 6);
    }
    // One undo entry for the call; redo closes it again.
    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(s.triangleCount(m)).toBe(2);
    expect(s.redo()).toBe(true);
    expect(s.triangleCount(m)).toBe(2 + 2 + 8);
  });

  it("shell_rejects_bad_thickness", () => {
    const s = scene();
    const m = s.addMesh(quad, quadTris);
    const beforePos = Float32Array.from(s.views(m).positions());
    const beforeIdx = Uint32Array.from(s.views(m).indices());
    const undoable = s.canUndo();
    expect(codeOf(() => s.repair(m, { op: "shell", thickness: 0 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "shell", thickness: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "shell", thickness: NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "shell", thickness: Infinity }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.repair(m, { op: "shell" } as never))).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(undoable);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(beforePos));
    expect(Array.from(s.views(m).indices())).toEqual(Array.from(beforeIdx));
  });
});
