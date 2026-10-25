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

  it("stale_loops_error_code", () => {
    const s = scene();
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
});
