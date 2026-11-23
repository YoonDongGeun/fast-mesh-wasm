// The multi-mesh boolean and isotropic remesh logic (#254): inputs from the
// multi-selection, the CSG trees, the geometry-moving edit ops, the edge
// statistics behind the default target, the panel fields and the reports.
import type { CsgNode, MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import type { Item } from "../src/app/state/app-state";
import {
  batchBlocked,
  batchBooleanTree,
  batchInputs,
  batchResultName,
  clearMeshOps,
  describeBatchBoolean,
  emptyResultMessage,
  liveTriangleIds,
  localToWorld,
  remapCorners,
  usedVertexIds,
} from "../src/features/mesh-edit/ux/batch-boolean";
import {
  creaseAngleInput,
  defaultTargetEdge,
  describeIsotropicRemesh,
  edgeStats,
  estimateRemeshTriangles,
  isotropicRemeshOptions,
  remeshBlocked,
  remeshIterationsInput,
} from "../src/features/mesh-edit/ux/isotropic-remesh";

let nextMesh = 0;
const newMesh = () => ({ __mesh: ++nextMesh }) as unknown as MeshHandle;

function item(id: number, name: string): Item {
  return { id, mesh: newMesh(), name } as unknown as Item;
}

/** The leaves of a tree, left to right, and the ops it uses. */
function walk(node: CsgNode, leaves: number[] = [], ops: string[] = []): { leaves: number[]; ops: string[] } {
  if (node.kind === "leaf") {
    leaves.push("meshIndex" in node ? node.meshIndex : -1);
  } else {
    ops.push(node.op);
    walk(node.left, leaves, ops);
    walk(node.right, leaves, ops);
  }
  return { leaves, ops };
}

function depth(node: CsgNode): number {
  return node.kind === "leaf" ? 0 : 1 + Math.max(depth(node.left), depth(node.right));
}

describe("multi-mesh boolean", () => {
  it("reads its inputs from the multi-selection, primary first, skipping meshes no longer listed", () => {
    const [a, b, c] = [item(1, "A"), item(2, "B"), item(3, "C")];
    const items = [a, b, c];
    expect(batchInputs([c.mesh, a.mesh, b.mesh], c.mesh, items)).toEqual([c, a, b]);
    // A removed mesh drops out; a repeat counts once.
    expect(batchInputs([b.mesh, newMesh(), b.mesh, a.mesh], b.mesh, items)).toEqual([b, a]);
    // A list that does not start at the primary is stale: the primary alone.
    expect(batchInputs([a.mesh, b.mesh], c.mesh, items)).toEqual([c]);
    expect(batchInputs([], null, items)).toEqual([]);
  });

  it("needs two meshes", () => {
    expect(batchBlocked(0)).not.toBeNull();
    expect(batchBlocked(1)).not.toBeNull();
    expect(batchBlocked(2)).toBeNull();
    expect(batchBlocked(5)).toBeNull();
    expect(() => batchBooleanTree("union", 1)).toThrow();
  });

  it("folds union and intersection over every input as a balanced tree", () => {
    for (const op of ["union", "intersect"] as const) {
      for (const n of [2, 3, 4, 7, 16]) {
        const tree = batchBooleanTree(op, n);
        const { leaves, ops } = walk(tree);
        expect(leaves).toEqual(Array.from({ length: n }, (_, i) => i));
        expect(ops.every((o) => o === op)).toBe(true);
        expect(ops.length).toBe(n - 1);
        expect(depth(tree)).toBe(Math.ceil(Math.log2(n)));
      }
    }
  });

  it("takes the union of the others from the primary", () => {
    const tree = batchBooleanTree("subtract", 4);
    expect(tree.kind === "op" && tree.op).toBe("subtract");
    if (tree.kind !== "op") {
      throw new Error("not an op");
    }
    expect(tree.left).toEqual({ kind: "leaf", meshIndex: 0 });
    const right = walk(tree.right);
    expect(right.leaves).toEqual([1, 2, 3]);
    expect(right.ops.every((o) => o === "union")).toBe(true);
    expect(batchBooleanTree("subtract", 2)).toEqual({
      kind: "op",
      op: "subtract",
      left: { kind: "leaf", meshIndex: 0 },
      right: { kind: "leaf", meshIndex: 1 },
    });
  });

  it("names the kept result after the primary and the count of others", () => {
    const inputs = [item(1, "A"), item(2, "B"), item(3, "C")];
    expect(batchResultName("union", inputs)).toBe("A union +2");
    expect(batchResultName("subtract", inputs)).toBe("A difference +2");
    expect(batchResultName("intersect", inputs.slice(0, 2))).toBe("A intersection +1");
  });

  it("clears a mesh's live triangles and the vertices they use", () => {
    // Slots: triangle 1 removed (0, 0, 0); vertex 4 unused.
    const indices = Uint32Array.from([0, 1, 2, 0, 0, 0, 2, 1, 3]);
    expect(Array.from(liveTriangleIds(indices))).toEqual([0, 2]);
    expect(Array.from(usedVertexIds(5, indices))).toEqual([0, 1, 2, 3]);
    expect(clearMeshOps(5, indices)).toEqual([
      { op: "removeTriangles", triangles: Uint32Array.from([0, 2]) },
      { op: "removeVertices", vertices: Uint32Array.from([0, 1, 2, 3]) },
    ]);
    expect(clearMeshOps(0, new Uint32Array(0))).toEqual([]);
  });

  it("maps local positions to world and renumbers corners to the added vertex ids", () => {
    // Column-major: scale 2 on x, then translate (10, 20, 30).
    const m = Float64Array.from([2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]);
    expect(Array.from(localToWorld(m, [1, 2, 3, -1, 0, 0]))).toEqual([12, 22, 33, 8, 20, 30]);
    expect(Array.from(remapCorners([0, 1, 2, 2, 1, 0], [7, 9, 4]))).toEqual([7, 9, 4, 4, 9, 7]);
    expect(() => remapCorners([0, 3], [7, 9])).toThrow();
  });

  it("says why a result came out empty", () => {
    expect(emptyResultMessage("intersect")).toMatch(/do not all overlap/);
    expect(emptyResultMessage("subtract")).toMatch(/cover the whole primary/);
    expect(emptyResultMessage("union")).toMatch(/no triangles/);
  });

  it("reports what happened and what undo does", () => {
    const base = { op: "union" as const, inputs: 3, name: "A union +2", triangles: 12345 };
    expect(describeBatchBoolean({ ...base, keptInputs: true })).toBe(
      "Union of 3 meshes → A union +2: 12,345 triangles (inputs kept; Ctrl+Z removes the result)",
    );
    expect(describeBatchBoolean({ ...base, op: "subtract", name: "A", keptInputs: false })).toBe(
      "Difference of 3 meshes → A: 12,345 triangles (the others removed; Ctrl+Z restores every input)",
    );
  });
});

describe("isotropic remesh", () => {
  // A unit right triangle pair (a square of side 1) plus a removed slot.
  const square = {
    positions: Float64Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    indices: Uint32Array.from([0, 1, 2, 0, 0, 0, 0, 2, 3]),
  };

  it("measures the live triangles' mean edge length and area", () => {
    const s = edgeStats(square.positions, square.indices);
    expect(s.triangles).toBe(2);
    expect(s.area).toBeCloseTo(1, 12);
    // Each triangle: two sides of 1 and the diagonal √2.
    expect(s.meanEdge).toBeCloseTo((2 + Math.SQRT2) / 3, 12);
    expect(edgeStats(new Float64Array(0), new Uint32Array(0))).toEqual({ triangles: 0, meanEdge: 0, area: 0 });
  });

  it("defaults the target to the mean edge length, to 3 significant digits", () => {
    expect(defaultTargetEdge(0.0366942)).toBe(0.0367);
    expect(defaultTargetEdge(12.3456)).toBe(12.3);
    expect(defaultTargetEdge(0)).toBe(0);
    expect(defaultTargetEdge(Number.NaN)).toBe(0);
  });

  it("estimates the triangle count from the area over an equilateral triangle's", () => {
    // Area 433.01... is 1000 equilateral triangles of edge 1.
    expect(estimateRemeshTriangles((Math.sqrt(3) / 4) * 1000, 1)).toBe(1000);
    // Halving the edge length quadruples the count.
    expect(estimateRemeshTriangles((Math.sqrt(3) / 4) * 1000, 0.5)).toBe(4000);
    expect(estimateRemeshTriangles(10, 0)).toBe(0);
    expect(estimateRemeshTriangles(0, 1)).toBe(0);
    expect(remeshBlocked(0)).toBe(true);
    expect(remeshBlocked(-1)).toBe(true);
    expect(remeshBlocked(Number.POSITIVE_INFINITY)).toBe(true);
    expect(remeshBlocked(0.2)).toBe(false);
  });

  it("parses the passes and crease angle fields", () => {
    expect(remeshIterationsInput("8")).toBe(8);
    expect(remeshIterationsInput("2.6")).toBe(3);
    expect(remeshIterationsInput("0")).toBe(1);
    expect(remeshIterationsInput("500")).toBe(64);
    expect(remeshIterationsInput("")).toBe(5);
    expect(remeshIterationsInput("x")).toBe(5);
    expect(creaseAngleInput("45")).toBe(45);
    expect(creaseAngleInput("-3")).toBe(0);
    expect(creaseAngleInput("200")).toBe(180);
    expect(creaseAngleInput("")).toBe(30);
  });

  it("builds the request, with the crease angle only while creases are kept", () => {
    const fields = { targetEdge: 0.4, iterations: 7, lockBoundaries: true, preserveCreases: true, creaseAngleDeg: 40 };
    expect(isotropicRemeshOptions(fields)).toEqual({
      targetEdgeLength: 0.4,
      iterations: 7,
      lockBoundaries: true,
      lockCreases: true,
      creaseAngleDeg: 40,
    });
    expect(isotropicRemeshOptions({ ...fields, preserveCreases: false, lockBoundaries: false })).toEqual({
      targetEdgeLength: 0.4,
      iterations: 7,
      lockBoundaries: false,
      lockCreases: false,
    });
  });

  it("reports the triangle counts before and after", () => {
    expect(
      describeIsotropicRemesh({
        initialTriangles: 12288,
        finalTriangles: 8428,
        splits: 7728,
        collapses: 9658,
        flips: 1,
        iterationsRun: 5,
        uvsDropped: true,
      }),
    ).toBe("12,288 → 8,428 triangles: 7,728 splits, 9,658 collapses, 1 flip in 5 passes (UVs dropped)");
  });
});
