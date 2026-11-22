// The multi-mesh boolean's pure parts (#254): the inputs read from the
// multi-selection (primary first), the CSG tree for each operation, the
// result's name, and the edit ops that move geometry between meshes so the
// whole boolean lands as one undo step. No React, no engine calls.
import type { BooleanMethod, BooleanOp, CsgNode, EditOp, MeshHandle } from "fast-mesh-wasm";
import type { Item } from "../../../app/state/app-state";

/** The operations over N meshes and their button labels. */
export const BATCH_BOOLEAN_OPERATIONS: readonly (readonly [BooleanOp, string])[] = [
  ["union", "Union all"],
  ["subtract", "Primary minus others"],
  ["intersect", "Intersect all"],
];

/** The boolean backends and their labels: exact first (the default). */
export const BATCH_BOOLEAN_METHODS: readonly (readonly [BooleanMethod, string])[] = [
  ["manifold", "Exact"],
  ["bvhSdf", "Voxel (BVH SDF)"],
  ["fhEdt", "Voxel (FH EDT)"],
];

/** The fewest meshes a multi-mesh boolean takes. */
export const BATCH_BOOLEAN_MIN_INPUTS = 2;

/**
 * The boolean's inputs: the multi-selection's listed meshes in click order,
 * the primary first. A list that does not start at the store's primary is
 * stale and reads as the primary alone (as `clickMulti` rebases it).
 */
export function batchInputs(
  multi: readonly MeshHandle[],
  primary: MeshHandle | null,
  items: readonly Item[],
): Item[] {
  const handles = primary !== null && multi[0] !== primary ? [primary] : multi;
  const out: Item[] = [];
  for (const m of handles) {
    const item = items.find((i) => i.mesh === m);
    if (item !== undefined && !out.includes(item)) {
      out.push(item);
    }
  }
  return out;
}

/** Why the boolean cannot run on `count` inputs, or `null` when it can. */
export function batchBlocked(count: number): string | null {
  return count < BATCH_BOOLEAN_MIN_INPUTS
    ? "Shift-click two or more meshes (the first one clicked is the primary)."
    : null;
}

/** Leaves `start .. start + count - 1` folded by `op` into a balanced tree. */
function balancedTree(op: BooleanOp, start: number, count: number): CsgNode {
  if (count === 1) {
    return { kind: "leaf", meshIndex: start };
  }
  const half = Math.floor(count / 2);
  return {
    kind: "op",
    op,
    left: balancedTree(op, start, half),
    right: balancedTree(op, start + half, count - half),
  };
}

/**
 * The CSG tree over `count` meshes (index 0 is the primary): union or
 * intersection of all of them as a balanced tree, or the primary minus the
 * union of the others (the same as subtracting each in turn, in one cut).
 */
export function batchBooleanTree(op: BooleanOp, count: number): CsgNode {
  if (!Number.isInteger(count) || count < BATCH_BOOLEAN_MIN_INPUTS) {
    throw new Error(`A multi-mesh boolean needs at least ${BATCH_BOOLEAN_MIN_INPUTS} meshes`);
  }
  if (op === "subtract") {
    return { kind: "op", op: "subtract", left: { kind: "leaf", meshIndex: 0 }, right: balancedTree("union", 1, count - 1) };
  }
  return balancedTree(op, 0, count);
}

/** The operation as the report and the result's name say it. */
export function batchOpLabel(op: BooleanOp): string {
  return op === "union" ? "union" : op === "intersect" ? "intersection" : "difference";
}

/** The new mesh's name when the inputs are kept: `"<primary> union +2"`. */
export function batchResultName(op: BooleanOp, inputs: readonly Pick<Item, "name">[]): string {
  const primary = inputs[0]?.name ?? "mesh";
  return `${primary} ${batchOpLabel(op)} +${Math.max(0, inputs.length - 1)}`;
}

/** The live triangle ids of an index view (a removed triangle reads `0, 0, 0`). */
export function liveTriangleIds(indices: ArrayLike<number>): Uint32Array {
  let n = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] !== 0 || indices[k + 1] !== 0 || indices[k + 2] !== 0) {
      n++;
    }
  }
  const out = new Uint32Array(n);
  let at = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] !== 0 || indices[k + 1] !== 0 || indices[k + 2] !== 0) {
      out[at++] = k / 3;
    }
  }
  return out;
}

/** The vertex ids the live triangles use, ascending. */
export function usedVertexIds(vertexCount: number, indices: ArrayLike<number>): Uint32Array {
  const used = new Uint8Array(vertexCount);
  let n = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const a = indices[k];
    const b = indices[k + 1];
    const c = indices[k + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    for (const v of [a, b, c]) {
      if (v < vertexCount && used[v] === 0) {
        used[v] = 1;
        n++;
      }
    }
  }
  const out = new Uint32Array(n);
  let at = 0;
  for (let v = 0; v < vertexCount; v++) {
    if (used[v] === 1) {
      out[at++] = v;
    }
  }
  return out;
}

/**
 * The edit ops that clear a mesh: its live triangles, then the vertices they
 * used. Empty when there is nothing to clear.
 */
export function clearMeshOps(vertexCount: number, indices: ArrayLike<number>): EditOp[] {
  const triangles = liveTriangleIds(indices);
  const vertices = usedVertexIds(vertexCount, indices);
  const ops: EditOp[] = [];
  if (triangles.length > 0) {
    ops.push({ op: "removeTriangles", triangles });
  }
  if (vertices.length > 0) {
    ops.push({ op: "removeVertices", vertices });
  }
  return ops;
}

/** Local positions mapped to world by a column-major 4x4 matrix (`Scene.matrix4`). */
export function localToWorld(matrix: ArrayLike<number>, local: ArrayLike<number>): Float64Array {
  const out = new Float64Array(local.length);
  for (let i = 0; i + 2 < local.length; i += 3) {
    const x = local[i];
    const y = local[i + 1];
    const z = local[i + 2];
    out[i] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    out[i + 1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    out[i + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  }
  return out;
}

/** Compact triangle corners renumbered to the ids `addVertices` gave them. */
export function remapCorners(indices: ArrayLike<number>, newIds: ArrayLike<number>): Uint32Array {
  const out = new Uint32Array(indices.length);
  for (let k = 0; k < indices.length; k++) {
    const id = newIds[indices[k]];
    if (id === undefined) {
      throw new Error(`triangle corner ${indices[k]} has no added vertex`);
    }
    out[k] = id;
  }
  return out;
}

/** Why a boolean came out empty, for its error line. */
export function emptyResultMessage(op: BooleanOp): string {
  return op === "intersect"
    ? "The intersection is empty: the meshes do not all overlap."
    : op === "subtract"
      ? "The difference is empty: the others cover the whole primary."
      : "The union is empty: the inputs have no triangles.";
}

/** The report line for a finished boolean. */
export function describeBatchBoolean(r: {
  op: BooleanOp;
  inputs: number;
  name: string;
  triangles: number;
  keptInputs: boolean;
}): string {
  const label = batchOpLabel(r.op);
  const head = `${label[0].toUpperCase()}${label.slice(1)} of ${r.inputs} meshes → ${r.name}: ${r.triangles.toLocaleString("en")} triangles`;
  return r.keptInputs
    ? `${head} (inputs kept; Ctrl+Z empties the result)`
    : `${head} (the others emptied; Ctrl+Z restores every input)`;
}
