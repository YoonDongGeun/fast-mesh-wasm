// The isotropic remesh's pure parts (#254): the mesh's edge statistics (the
// default target is its mean edge length), the triangle count a target
// edge length leads to, the panel's fields, the `remeshSurface` request and
// the report line. No React, no engine calls.
import type { IsotropicRemeshOptions, SurfaceRemeshReport } from "fast-mesh-wasm";

/** What the remesh section shows about a mesh before it runs. */
export interface EdgeStats {
  /** Live triangles. */
  triangles: number;
  /** Mean edge length in local mesh units (0 without triangles). */
  meanEdge: number;
  /** Surface area in local mesh units squared. */
  area: number;
}

/** The engine's pass cap for `remeshSurface`. */
export const MAX_REMESH_ITERATIONS = 64;
/** The engine's default pass count. */
export const DEFAULT_REMESH_ITERATIONS = 5;
/** The engine's default crease threshold, in degrees. */
export const DEFAULT_CREASE_ANGLE_DEG = 30;

/**
 * Triangle count, mean edge length and area over the live triangles (a
 * removed triangle reads `0, 0, 0`). Each triangle adds its three edges, so
 * an interior edge counts twice, like the boundary edges once: the mean the
 * remesher's per-triangle sizing sees. One pass, no allocation.
 */
export function edgeStats(positions: ArrayLike<number>, indices: ArrayLike<number>): EdgeStats {
  let triangles = 0;
  let edgeSum = 0;
  let area2 = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] === 0 && indices[k + 1] === 0 && indices[k + 2] === 0) {
      continue;
    }
    const a = 3 * indices[k];
    const b = 3 * indices[k + 1];
    const c = 3 * indices[k + 2];
    const abx = positions[b] - positions[a];
    const aby = positions[b + 1] - positions[a + 1];
    const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a];
    const acy = positions[c + 1] - positions[a + 1];
    const acz = positions[c + 2] - positions[a + 2];
    const bcx = positions[c] - positions[b];
    const bcy = positions[c + 1] - positions[b + 1];
    const bcz = positions[c + 2] - positions[b + 2];
    edgeSum += Math.hypot(abx, aby, abz) + Math.hypot(acx, acy, acz) + Math.hypot(bcx, bcy, bcz);
    area2 += Math.hypot(aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx);
    triangles++;
  }
  return { triangles, meanEdge: triangles === 0 ? 0 : edgeSum / (3 * triangles), area: area2 / 2 };
}

/** The default target edge length: the mean edge length to 3 significant digits (0 without one). */
export function defaultTargetEdge(meanEdge: number): number {
  return meanEdge > 0 && Number.isFinite(meanEdge) ? Number(meanEdge.toPrecision(3)) : 0;
}

/**
 * About how many triangles an isotropic remesh of `area` at edge length
 * `edge` gives: the area over an equilateral triangle's (`√3/4 · edge²`).
 * 0 for a target that is not a positive finite length.
 */
export function estimateRemeshTriangles(area: number, edge: number): number {
  if (!(edge > 0) || !Number.isFinite(edge) || !(area > 0)) {
    return 0;
  }
  return Math.round(area / ((Math.sqrt(3) / 4) * edge * edge));
}

/** The iterations field: a whole number of passes in `1 ..= 64`; an empty or invalid entry reads as 5. */
export function remeshIterationsInput(text: string): number {
  const n = Math.round(Number(text));
  if (text.trim() === "" || !Number.isFinite(n)) {
    return DEFAULT_REMESH_ITERATIONS;
  }
  return Math.max(1, Math.min(MAX_REMESH_ITERATIONS, n));
}

/** The crease angle field, in degrees `0 ..= 180`; an empty or invalid entry reads as 30. */
export function creaseAngleInput(text: string): number {
  const n = Number(text);
  if (text.trim() === "" || !Number.isFinite(n)) {
    return DEFAULT_CREASE_ANGLE_DEG;
  }
  return Math.max(0, Math.min(180, n));
}

/** Whether the target edge length cannot run: not a positive finite length. */
export function remeshBlocked(targetEdge: number): boolean {
  return !(targetEdge > 0) || !Number.isFinite(targetEdge);
}

/** The remesh section's fields. */
export interface IsotropicRemeshFields {
  /** Target edge length in local mesh units (the showcase's millimetres). */
  targetEdge: number;
  iterations: number;
  lockBoundaries: boolean;
  /** Keep sharp creases and corners (the engine's `lockCreases`). */
  preserveCreases: boolean;
  creaseAngleDeg: number;
}

/** The `remesh.isotropic` options for the fields (the crease angle only while creases are kept). */
export function isotropicRemeshOptions(fields: IsotropicRemeshFields): IsotropicRemeshOptions {
  return {
    targetEdgeLength: fields.targetEdge,
    iterations: remeshIterationsInput(String(fields.iterations)),
    lockBoundaries: fields.lockBoundaries,
    lockCreases: fields.preserveCreases,
    ...(fields.preserveCreases ? { creaseAngleDeg: fields.creaseAngleDeg } : {}),
  };
}

/** The report line: triangle counts before and after, then what the passes did. */
export function describeIsotropicRemesh(r: SurfaceRemeshReport): string {
  const n = (v: number) => v.toLocaleString("en");
  const s = (v: number, one: string, many = `${one}s`) => `${n(v)} ${v === 1 ? one : many}`;
  const dropped = r.uvsDropped ? " (UVs dropped)" : "";
  return (
    `${n(r.initialTriangles)} → ${n(r.finalTriangles)} triangles: ` +
    `${s(r.splits, "split")}, ${s(r.collapses, "collapse")}, ${s(r.flips, "flip")} ` +
    `in ${s(r.iterationsRun, "pass", "passes")}${dropped}`
  );
}
