// Face selections as sorted triangle index sets (#233): which index slots
// hold live triangles, and the replace/add/subtract/intersect, sanitize and
// invert operations the selection tools and modifiers combine them with.
// Pure, for tests.
import type { Wave6SelectState } from "../../../app/state/wave6-state";

/** The empty selection. */
export const EMPTY_U32 = new Uint32Array(0);

/** True when a triangle index slot holds a non-dead `[0, 0, 0]` triangle. */
export function isLiveTriangleSlot(indices: ArrayLike<number>, tri: number): boolean {
  const base = tri * 3;
  if (tri < 0 || base + 2 >= indices.length) {
    return false;
  }
  const a = indices[base];
  const b = indices[base + 1];
  const c = indices[base + 2];
  return !(a === 0 && b === 0 && c === 0);
}

/** Counts non-dead triangles in an index buffer. */
export function countLiveTriangles(indices: ArrayLike<number>): number {
  const total = Math.floor(indices.length / 3);
  let live = 0;
  for (let t = 0; t < total; t++) {
    if (isLiveTriangleSlot(indices, t)) {
      live++;
    }
  }
  return live;
}

/** Whether a `Uint32Array` is strictly ascending (sorted, no duplicates). */
function isStrictlyAscending(values: Uint32Array): boolean {
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      return false;
    }
  }
  return true;
}

/**
 * Returns sorted, deduplicated non-negative integer triangle indices. An
 * already ascending `Uint32Array` (what every tool stores) comes back as
 * is, so large selections cost one pass.
 */
function toSortedUniqueUint32(values: ArrayLike<number>): Uint32Array {
  if (values.length === 0) {
    return EMPTY_U32;
  }
  if (values instanceof Uint32Array && isStrictlyAscending(values)) {
    return values;
  }
  const clean: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isInteger(v) && v >= 0) {
      clean.push(v);
    }
  }
  if (clean.length === 0) {
    return EMPTY_U32;
  }
  clean.sort((a, b) => a - b);
  let unique = 1;
  for (let i = 1; i < clean.length; i++) {
    if (clean[i] !== clean[unique - 1]) {
      clean[unique++] = clean[i];
    }
  }
  const out = new Uint32Array(unique);
  for (let i = 0; i < unique; i++) {
    out[i] = clean[i];
  }
  return out;
}

/** Combines `current` and `incoming` triangle selections according to `mode`. */
export function combineFaceSelection(
  current: Uint32Array | ArrayLike<number>,
  incoming: Uint32Array | ArrayLike<number>,
  mode: Wave6SelectState["combineMode"],
): Uint32Array {
  const b = toSortedUniqueUint32(incoming);
  if (mode === "replace") {
    return b;
  }
  const a = toSortedUniqueUint32(current);
  if (mode === "add") {
    const out: number[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] < b[j]) {
        out.push(a[i++]);
      } else if (b[j] < a[i]) {
        out.push(b[j++]);
      } else {
        out.push(a[i]);
        i++;
        j++;
      }
    }
    while (i < a.length) out.push(a[i++]);
    while (j < b.length) out.push(b[j++]);
    return Uint32Array.from(out);
  }
  if (mode === "subtract") {
    if (a.length === 0 || b.length === 0) {
      return a;
    }
    const out: number[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] < b[j]) {
        out.push(a[i++]);
      } else if (b[j] < a[i]) {
        j++;
      } else {
        i++;
        j++;
      }
    }
    while (i < a.length) out.push(a[i++]);
    return Uint32Array.from(out);
  }
  // mode === "intersect"
  if (a.length === 0 || b.length === 0) {
    return EMPTY_U32;
  }
  const out: number[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] < b[j]) {
      i++;
    } else if (b[j] < a[i]) {
      j++;
    } else {
      out.push(a[i]);
      i++;
      j++;
    }
  }
  return Uint32Array.from(out);
}

/** Filters `selected` to unique live triangle slots within `indices`. */
export function sanitizeSelectedTriangles(
  indices: ArrayLike<number>,
  selected: Uint32Array | ArrayLike<number>,
): Uint32Array {
  const sorted = toSortedUniqueUint32(selected);
  if (sorted.length === 0) {
    return EMPTY_U32;
  }
  let allLive = true;
  for (let i = 0; i < sorted.length && allLive; i++) {
    allLive = isLiveTriangleSlot(indices, sorted[i]);
  }
  if (allLive) {
    return sorted;
  }
  const out: number[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const tri = sorted[i];
    if (isLiveTriangleSlot(indices, tri)) {
      out.push(tri);
    }
  }
  return out.length === 0 ? EMPTY_U32 : Uint32Array.from(out);
}

/** Inverts `selected` over all live triangle slots in `indices`. */
export function invertFaceSelection(
  indices: ArrayLike<number>,
  selected: Uint32Array | ArrayLike<number>,
): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  if (triCount <= 0) {
    return EMPTY_U32;
  }
  const excluded = new Set<number>();
  for (let i = 0; i < selected.length; i++) {
    excluded.add(selected[i]);
  }
  const out: number[] = [];
  for (let t = 0; t < triCount; t++) {
    if (!excluded.has(t) && isLiveTriangleSlot(indices, t)) {
      out.push(t);
    }
  }
  return Uint32Array.from(out);
}
