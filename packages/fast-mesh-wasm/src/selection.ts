// Vertex selections over one mesh: sphere-brush and lasso painting,
// connected-component picks, grow/shrink/invert and isolate. Pure over the
// mesh's local positions and indices (no wasm changes); `Scene.select` and
// friends in `scene.ts` wire these helpers to views, placements and stamps.
//
// A `VertexSelection` is a snapshot: the mesh, its ascending vertex ids and
// the mesh's content stamp when taken. Any topology edit (edit, repair, cut,
// join, boolean, compact, undo, redo, clearHistory) bumps the stamp and
// invalidates earlier selections: the Scene methods throw `STALE_SELECTION`
// instead of silently remapping ids. Reselect after each edit, or derive the
// next selection from the previous one (grow/shrink/invert re-stamp).
// Placement-only changes keep selections valid: ids are
// placement-independent, and world-space brushes convert at call time.
//
// The selectable universe is the vertices a live triangle uses. A removed
// triangle slot reads `0, 0, 0`, so triples of three zeros are skipped; a
// genuine triangle with all three vertices at slot 0 is degenerate (zero
// area) and losing it from adjacency is harmless. Dead or never-used slots
// are never selected.
import type { Vec3 } from "./api.js";
import { FastMeshError } from "./core.js";
import type { MeshHandle } from "./scene.js";

/** A snapshot of selected vertices of one mesh; see the module docs. */
export interface VertexSelection {
  /** The mesh selected from. */
  mesh: MeshHandle;
  /** The selected vertex slots, ascending. */
  vertices: Uint32Array;
  /** The mesh's content stamp when taken; see `Scene.selectionAlive`. */
  stamp: number;
}

/** A sphere brush: every used vertex within `radius` of `center`. */
export interface BrushSelect {
  kind: "brush";
  /** Brush centre; `space` says which coordinates it is in. */
  center: Readonly<Vec3>;
  /** Radius in the same units; 0 selects vertices exactly at `center`. */
  radius: number;
  /**
   * `"local"` (mesh render coordinates, as in `views(mesh).positions()`) or
   * `"world"` (converted with the mesh's placement at call time). Defaults
   * to `"local"`.
   */
  space?: "local" | "world";
}

/**
 * A 2D polygon: every used vertex whose projection falls inside it.
 * Project your screen-space polygon to mesh-local coordinates with your
 * camera first; `axis` picks the projection plane.
 */
export interface LassoSelect {
  kind: "lasso";
  /** Flat `[x0, y0, x1, y1, ...]` polygon corners in local coordinates. */
  polygon: Float32Array | readonly number[];
  /** The local plane projected onto. Defaults to `"xy"`. */
  axis?: "xy" | "xz" | "yz";
}

/** The connected component (over shared triangle edges) holding `seed`. */
export interface ComponentSelect {
  kind: "component";
  /** A vertex slot of the component to pick; it must be alive. */
  seed: number;
}

/** What `Scene.select` takes. */
export type SelectSpec = BrushSelect | LassoSelect | ComponentSelect;

/** What `Scene.isolate` did. */
export interface IsolateReport {
  /** Triangles fully outside the selection that were removed. */
  removedTriangles: number;
  /** The same vertex set with the post-edit stamp. */
  selection: VertexSelection;
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function stale(message: string): FastMeshError {
  return new FastMeshError("STALE_SELECTION", message);
}

/** Checks a spec's shape; the Scene methods check the mesh and the stamp. */
export function checkSpec(spec: SelectSpec): void {
  if (spec === null || typeof spec !== "object") {
    throw invalid("spec must be an object");
  }
  switch (spec.kind) {
    case "brush": {
      const c = spec.center as readonly number[] | undefined;
      if (c?.length !== 3 || ![c[0], c[1], c[2]].every(Number.isFinite)) {
        throw invalid("brush center must be 3 finite numbers");
      }
      if (typeof spec.radius !== "number" || !(spec.radius >= 0)) {
        throw invalid("brush radius must be a number, zero or greater");
      }
      if (spec.space !== undefined && spec.space !== "local" && spec.space !== "world") {
        throw invalid('brush space must be "local" or "world"');
      }
      return;
    }
    case "lasso": {
      const p = spec.polygon;
      const n = Array.isArray(p) ? p.length : p instanceof Float32Array ? p.length : -1;
      if (n < 6 || n % 2 !== 0) {
        throw invalid("lasso polygon needs at least 3 [x, y] corners");
      }
      const pts: ArrayLike<number> = p as ArrayLike<number>;
      for (let k = 0; k < n; k++) {
        if (!Number.isFinite(pts[k])) {
          throw invalid("lasso polygon must hold finite numbers");
        }
      }
      if (spec.axis !== undefined && spec.axis !== "xy" && spec.axis !== "xz" && spec.axis !== "yz") {
        throw invalid('lasso axis must be "xy", "xz" or "yz"');
      }
      return;
    }
    case "component": {
      if (!Number.isInteger(spec.seed) || spec.seed < 0 || spec.seed >= 2 ** 32) {
        throw new FastMeshError("INVALID_INDEX", "component seed is not a vertex id");
      }
      return;
    }
    default:
      throw invalid(`unknown select kind "${(spec as SelectSpec).kind}"`);
  }
}

/** Per-vertex adjacency and liveness over the live triangles of `indices`. */
export interface Topology {
  /** `neighbors[v]`: the vertices sharing a live triangle edge with `v`. */
  neighbors: number[][];
  /** Whether `v` is used by a live triangle. */
  used: Uint8Array;
  /** The live triangle slots, ascending. */
  liveTriangles: Uint32Array;
}

/**
 * Builds adjacency from index triples, skipping removed slots (`0, 0, 0`).
 * `slots` is the vertex slot count.
 */
export function buildTopology(indices: Uint32Array, slots: number): Topology {
  const sets: Set<number>[] = Array.from({ length: slots }, () => new Set<number>());
  const used = new Uint8Array(slots);
  const live: number[] = [];
  const link = (a: number, b: number): void => {
    if (a !== b && a < slots && b < slots) {
      sets[a].add(b);
      sets[b].add(a);
    }
  };
  for (let t = 0; t * 3 + 2 < indices.length; t++) {
    const a = indices[t * 3];
    const b = indices[t * 3 + 1];
    const c = indices[t * 3 + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a >= slots || b >= slots || c >= slots) {
      continue;
    }
    live.push(t);
    used[a] = 1;
    used[b] = 1;
    used[c] = 1;
    link(a, b);
    link(b, c);
    link(c, a);
  }
  return { neighbors: sets.map((s) => [...s].sort((x, y) => x - y)), used, liveTriangles: Uint32Array.from(live) };
}

/** Every used vertex within `radius` of `center` (squared distances). */
export function brushPick(positions: Float32Array, used: Uint8Array, center: Readonly<Vec3>, radius: number): number[] {
  const out: number[] = [];
  const r2 = radius * radius;
  for (let v = 0; v < used.length; v++) {
    if (used[v] === 0) {
      continue;
    }
    const dx = positions[v * 3] - center[0];
    const dy = positions[v * 3 + 1] - center[1];
    const dz = positions[v * 3 + 2] - center[2];
    if (dx * dx + dy * dy + dz * dz <= r2) {
      out.push(v);
    }
  }
  return out;
}

/** Even-odd point-in-polygon over flat `[x0, y0, ...]`. */
function insidePolygon(polygon: ArrayLike<number>, x: number, y: number): boolean {
  let inside = false;
  const n = polygon.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i * 2];
    const yi = polygon[i * 2 + 1];
    const xj = polygon[j * 2];
    const yj = polygon[j * 2 + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** Every used vertex whose `axis` projection falls inside `polygon`. */
export function lassoPick(
  positions: Float32Array,
  used: Uint8Array,
  polygon: Float32Array | readonly number[],
  axis: "xy" | "xz" | "yz",
): number[] {
  const pick = axis === "xy" ? [0, 1] : axis === "xz" ? [0, 2] : [1, 2];
  const out: number[] = [];
  for (let v = 0; v < used.length; v++) {
    if (used[v] === 0) {
      continue;
    }
    if (insidePolygon(polygon, positions[v * 3 + pick[0]], positions[v * 3 + pick[1]])) {
      out.push(v);
    }
  }
  return out;
}

/**
 * The connected component holding `seed` (over triangle edges). Throws
 * `INVALID_ARGUMENT` when the seed is out of range or not alive.
 */
export function componentOf(topology: Topology, seed: number): number[] {
  if (!Number.isInteger(seed) || seed < 0 || seed >= topology.used.length) {
    throw new FastMeshError("INVALID_INDEX", "component seed is not a vertex id");
  }
  if (topology.used[seed] === 0) {
    throw invalid("component seed is not on a live triangle");
  }
  const seen = new Uint8Array(topology.used.length);
  const queue = [seed];
  seen[seed] = 1;
  while (queue.length > 0) {
    const v = queue.pop() as number;
    for (const w of topology.neighbors[v]) {
      if (seen[w] === 0) {
        seen[w] = 1;
        queue.push(w);
      }
    }
  }
  const out: number[] = [];
  for (let v = 0; v < seen.length; v++) {
    if (seen[v] === 1) {
      out.push(v);
    }
  }
  return out;
}

/**
 * Grows `ids` by `steps` rings of edge neighbours (staying on live
 * vertices). `steps` 0 returns `ids`.
 */
export function growIds(topology: Topology, ids: Iterable<number>, steps: number): number[] {
  if (!Number.isInteger(steps) || steps < 0) {
    throw invalid("steps must be a non-negative integer");
  }
  const inSet = new Uint8Array(topology.used.length);
  for (const v of ids) {
    if (v < inSet.length) {
      inSet[v] = 1;
    }
  }
  let frontier = [...ids].filter((v) => v < inSet.length);
  for (let s = 0; s < steps; s++) {
    const next: number[] = [];
    for (const v of frontier) {
      for (const w of topology.neighbors[v]) {
        if (inSet[w] === 0) {
          inSet[w] = 1;
          next.push(w);
        }
      }
    }
    if (next.length === 0) {
      break;
    }
    frontier = next;
  }
  const out: number[] = [];
  for (let v = 0; v < inSet.length; v++) {
    if (inSet[v] === 1) {
      out.push(v);
    }
  }
  return out;
}

/**
 * Shrinks `ids` by `steps` rings: each step drops the members with a live
 * neighbour outside the set. Interior selections (no boundary) survive
 * unchanged.
 */
export function shrinkIds(topology: Topology, ids: Iterable<number>, steps: number): number[] {
  if (!Number.isInteger(steps) || steps < 0) {
    throw invalid("steps must be a non-negative integer");
  }
  let current = new Set([...ids].filter((v) => v < topology.used.length));
  for (let s = 0; s < steps; s++) {
    const peel: number[] = [];
    for (const v of current) {
      if (topology.neighbors[v].some((w) => !current.has(w))) {
        peel.push(v);
      }
    }
    if (peel.length === 0) {
      break;
    }
    for (const v of peel) {
      current.delete(v);
    }
  }
  return [...current].sort((a, b) => a - b);
}

/** Throws `STALE_SELECTION` when `sel` is not for `mesh` or predates its stamp. */
export function checkSelection(sel: VertexSelection, mesh: MeshHandle, stamp: number): void {
  if (sel === null || typeof sel !== "object" || sel.mesh !== mesh) {
    throw invalid("the selection is not for this mesh");
  }
  if (!Number.isInteger(sel.stamp) || sel.stamp !== stamp) {
    throw stale("the selection predates the mesh's latest topology edit; reselect it");
  }
}

/**
 * The inverse of a column-major 4x4 (a placement's `matrix4`), for
 * world-space brushes. Throws `INVALID_ARGUMENT` when not invertible.
 */
export function invertMatrix4(m: Float64Array): Float64Array {
  const inv = new Float64Array(16);
  const a00 = m[0];
  const a01 = m[1];
  const a02 = m[2];
  const a03 = m[3];
  const a10 = m[4];
  const a11 = m[5];
  const a12 = m[6];
  const a13 = m[7];
  const a20 = m[8];
  const a21 = m[9];
  const a22 = m[10];
  const a23 = m[11];
  const a30 = m[12];
  const a31 = m[13];
  const a32 = m[14];
  const a33 = m[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0 || !Number.isFinite(det)) {
    throw invalid("the mesh placement is not invertible");
  }
  const s = 1 / det;
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * s;
  inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * s;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * s;
  inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * s;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * s;
  inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * s;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * s;
  inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * s;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * s;
  inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * s;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * s;
  inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * s;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * s;
  inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * s;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * s;
  inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * s;
  return inv;
}

/** Applies a column-major 4x4 to a point. */
export function transformPoint(m: Float64Array, p: Readonly<Vec3>): Vec3 {
  const x = p[0];
  const y = p[1];
  const z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w === 0 || !Number.isFinite(w)) {
    throw invalid("the mesh placement does not map the brush centre");
  }
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}
