// What the repair-issue overlays draw, derived from `analyze` (read-only):
// hole boundary polylines, non-manifold edges, degenerate triangle edges and
// the small components' triangles, as flat position arrays the three.js
// overlay in `ui/repair-overlays.ts` wraps in buffers, plus how many
// primitives each kind would draw (for the test hook).
import type { Analysis, BoundaryLoop, MeshComponent } from "fast-mesh-wasm";

/** Copies vertex `v`'s position into `out`; `false` when `v` is out of range. */
function at(positions: ArrayLike<number>, v: number, out: number[]): boolean {
  const o = 3 * v;
  if (!Number.isInteger(v) || v < 0 || o + 2 >= positions.length) {
    return false;
  }
  out[0] = positions[o];
  out[1] = positions[o + 1];
  out[2] = positions[o + 2];
  return true;
}

/** Pushes the segment `a`-`b` unless it is degenerate (a point). */
function pushSegment(segs: number[], a: number[], b: number[]): void {
  if (a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2]) {
    segs.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
}

/**
 * Segment endpoints `[x0, y0, z0, x1, y1, z1, ...]` of every loop's
 * boundary polyline (closed loops wrap around; open chains do not). Loops
 * with fewer than two known vertices draw nothing. Out-of-range vertices
 * (an analysis older than the positions) break the polyline there.
 */
export function holeSegments(loops: readonly BoundaryLoop[], positions: ArrayLike<number>): Float32Array {
  const segs: number[] = [];
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  for (const loop of loops) {
    const n = loop.vertices.length;
    if (n < 2) {
      continue;
    }
    const edges = loop.closed ? n : n - 1;
    for (let k = 0; k < edges; k++) {
      if (!at(positions, loop.vertices[k], a) || !at(positions, loop.vertices[(k + 1) % n], b)) {
        continue;
      }
      pushSegment(segs, a, b);
    }
  }
  return Float32Array.from(segs);
}

/**
 * Segment endpoints of the non-manifold edges (`[min, max]` vertex pairs).
 * Out-of-range ends (an analysis older than the positions) are skipped.
 */
export function nonManifoldSegments(
  edges: readonly (readonly [min: number, max: number])[],
  positions: ArrayLike<number>,
): Float32Array {
  const segs: number[] = [];
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  for (const [u, v] of edges) {
    if (!at(positions, u, a) || !at(positions, v, b)) {
      continue;
    }
    pushSegment(segs, a, b);
  }
  return Float32Array.from(segs);
}

/**
 * Segment endpoints of every listed triangle's three edges. `tris` holds
 * triangle slots into `indices` (as `analyze`'s triangle lists do); slots
 * out of range, and corners out of range, are skipped. Shared edges repeat:
 * overdraw is harmless (one draw call, depth test off).
 */
export function triangleEdgeSegments(
  tris: ArrayLike<number>,
  indices: ArrayLike<number>,
  positions: ArrayLike<number>,
): Float32Array {
  const segs: number[] = [];
  const corners = [0, 0, 0];
  const p = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const slots = indices.length / 3;
  for (let k = 0; k < tris.length; k++) {
    const t = tris[k];
    if (!Number.isInteger(t) || t < 0 || t >= slots) {
      continue;
    }
    let known = true;
    for (let c = 0; c < 3; c++) {
      corners[c] = indices[3 * t + c];
      if (!at(positions, corners[c], p[c])) {
        known = false;
        break;
      }
    }
    if (!known) {
      continue;
    }
    pushSegment(segs, p[0], p[1]);
    pushSegment(segs, p[1], p[2]);
    pushSegment(segs, p[2], p[0]);
  }
  return Float32Array.from(segs);
}

/**
 * The triangle slots of every component but the largest (by triangle
 * count): the "small components" a removal would drop. Ties keep the first
 * largest; a single component (or none) highlights nothing.
 */
export function smallComponentTris(components: readonly MeshComponent[]): Uint32Array {
  if (components.length < 2) {
    return new Uint32Array(0);
  }
  let largest = 0;
  for (let k = 1; k < components.length; k++) {
    if (components[k].triangles.length > components[largest].triangles.length) {
      largest = k;
    }
  }
  let n = 0;
  for (let k = 0; k < components.length; k++) {
    if (k !== largest) {
      n += components[k].triangles.length;
    }
  }
  const out = new Uint32Array(n);
  let offset = 0;
  for (let k = 0; k < components.length; k++) {
    if (k !== largest) {
      out.set(components[k].triangles, offset);
      offset += components[k].triangles.length;
    }
  }
  return out;
}

/**
 * A non-indexed triangle soup of `tris` (slots into `indices`): the small
 * components' fill. Slots or corners out of range are skipped, so a soup
 * may hold fewer triangles than listed.
 */
export function triangleSoup(
  tris: ArrayLike<number>,
  indices: ArrayLike<number>,
  positions: ArrayLike<number>,
): Float32Array {
  const soup: number[] = [];
  const corner = [0, 0, 0];
  const slots = indices.length / 3;
  for (let k = 0; k < tris.length; k++) {
    const t = tris[k];
    if (!Number.isInteger(t) || t < 0 || t >= slots) {
      continue;
    }
    let known = true;
    for (let c = 0; c < 3; c++) {
      if (!at(positions, indices[3 * t + c], corner)) {
        known = false;
        break;
      }
      soup.push(corner[0], corner[1], corner[2]);
    }
    if (!known) {
      // Drop the partial triangle (0, 3 or 6 numbers just pushed).
      soup.length = Math.floor(soup.length / 9) * 9;
    }
  }
  return Float32Array.from(soup);
}

/** How many primitives each kind would draw (for the test hook). */
export interface RepairOverlayCounts {
  holes: number;
  nonManifold: number;
  degenerate: number;
  components: number;
}

/** Counts the overlay primitives without building three.js objects. */
export function countRepairOverlays(analysis: Analysis): RepairOverlayCounts {
  return {
    holes: analysis.loops.reduce(
      (n, l) => n + (l.vertices.length >= 2 ? (l.closed ? l.vertices.length : l.vertices.length - 1) : 0),
      0,
    ),
    nonManifold: analysis.nonManifoldEdgeList.length,
    degenerate: analysis.degenerateList.length + analysis.duplicateList.length,
    components: smallComponentTris(analysis.components).length,
  };
}
