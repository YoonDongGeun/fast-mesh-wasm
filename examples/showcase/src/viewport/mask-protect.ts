// Per-vertex mask protection: what a dab may move near the painted mask.
//
// A dab goes to the engine with a `protect` list: the masked vertices it can
// reach at weight 0, so they stay byte-identical, and the free vertices in a
// soft band around the mask at a weight rising to 1, so no crease forms along
// the border. The brush itself is untouched; this only decorates its op.
import type { RepairOp } from "fast-mesh-wasm";

/** The soft band's width, as a fraction of the brush radius. */
export const MASK_SOFT_BAND = 0.25;

/**
 * How far past the brush radius a dab lists vertices. A remesh in flight can
 * still move vertices the mirror shows elsewhere; listing more only costs a
 * few comparisons, since the engine writes back only what moved.
 */
const PROTECT_REACH = 1.25;

/** The mask's protection over one mesh's buffers (see {@link maskEdgeField}). */
export interface MaskField {
  /** 1 on a masked slot. */
  flags: Uint8Array;
  /** The masked slots and the band's: everything a dab may have to hold back. */
  ids: Uint32Array;
  /** Per id: 0 on a masked slot, rising to 1 at the band's outer edge. */
  weights: Float32Array;
  /** The same weights per slot, 1 outside the field: for a move written directly (a grab). */
  scale: Float32Array;
}

/**
 * Builds the field: the distance along mesh edges from the mask, out to
 * `band`, as smoothstep weights. O(triangles) for the adjacency, then the
 * band; built once per mask, slot count and radius, not per dab, so distances
 * are as of the build. They only shape the soft edge: masked slots are
 * flagged exactly, whatever moved since.
 */
export function maskEdgeField(
  masked: ArrayLike<number>,
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  slots: number,
  band: number,
): MaskField {
  const flags = new Uint8Array(slots);
  let maskedCount = 0;
  for (let k = 0; k < masked.length; k++) {
    const v = masked[k];
    if (v >= 0 && v < slots && flags[v] === 0) {
      flags[v] = 1;
      maskedCount++;
    }
  }
  // The live triangles' adjacency, compressed: `near[starts[v] .. starts[v + 1]]`.
  const starts = new Uint32Array(slots + 2);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    starts[a + 2] += 2;
    starts[b + 2] += 2;
    starts[c + 2] += 2;
  }
  for (let v = 2; v < slots + 2; v++) {
    starts[v] += starts[v - 1];
  }
  const near = new Uint32Array(starts[slots + 1]);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    near[starts[a + 1]++] = b;
    near[starts[a + 1]++] = c;
    near[starts[b + 1]++] = a;
    near[starts[b + 1]++] = c;
    near[starts[c + 1]++] = a;
    near[starts[c + 1]++] = b;
  }
  // A label-correcting walk out from the mask, bounded by the band.
  const dist = new Float32Array(slots).fill(Number.POSITIVE_INFINITY);
  const queue = new Uint32Array(Math.max(slots, 1));
  const queued = new Uint8Array(slots);
  let head = 0;
  let waiting = 0;
  const enqueue = (v: number) => {
    if (queued[v] === 0) {
      queued[v] = 1;
      queue[(head + waiting) % queue.length] = v;
      waiting++;
    }
  };
  for (let v = 0; v < slots; v++) {
    if (flags[v] === 1) {
      dist[v] = 0;
      enqueue(v);
    }
  }
  while (waiting > 0) {
    const v = queue[head];
    head = (head + 1) % queue.length;
    waiting--;
    queued[v] = 0;
    for (let k = starts[v]; k < starts[v + 1]; k++) {
      const n = near[k];
      if (flags[n] === 1) {
        continue;
      }
      const dx = positions[3 * v] - positions[3 * n];
      const dy = positions[3 * v + 1] - positions[3 * n + 1];
      const dz = positions[3 * v + 2] - positions[3 * n + 2];
      const d = dist[v] + Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < band && d < dist[n]) {
        dist[n] = d;
        enqueue(n);
      }
    }
  }
  let count = maskedCount;
  for (let v = 0; v < slots; v++) {
    if (flags[v] === 0 && dist[v] < band) {
      count++;
    }
  }
  const ids = new Uint32Array(count);
  const weights = new Float32Array(count);
  const scale = new Float32Array(slots).fill(1);
  let at = 0;
  for (let v = 0; v < slots; v++) {
    if (flags[v] === 1) {
      scale[v] = 0;
      ids[at++] = v;
    } else if (dist[v] < band) {
      const s = dist[v] / band;
      weights[at] = scale[v] = s * s * (3 - 2 * s);
      ids[at++] = v;
    }
  }
  return { flags, ids, weights, scale };
}

/**
 * Whether the mask leaves a dab nothing to move: its ball holds a live vertex
 * and every live vertex in it is masked. Such a dab is dropped before the
 * scene, so a fully masked press leaves no undo step behind. Stops at the
 * first free vertex, so a dab with any free vertex pays little.
 */
export function maskCoversDab(
  flags: Uint8Array,
  live: Uint8Array,
  positions: ArrayLike<number>,
  center: readonly [number, number, number],
  radius: number,
): boolean {
  const r2 = radius * radius;
  let any = false;
  for (let v = 0; v < live.length; v++) {
    if (live[v] === 0) {
      continue;
    }
    const dx = positions[3 * v] - center[0];
    const dy = positions[3 * v + 1] - center[1];
    const dz = positions[3 * v + 2] - center[2];
    if (dx * dx + dy * dy + dz * dz <= r2) {
      if (flags[v] === 0) {
        return false;
      }
      any = true;
    }
  }
  return any;
}

/**
 * Whether a remesh of this region could take a masked vertex. A remesh tests
 * edge midpoints, not vertices, so it reaches past its radius along any edge
 * with a midpoint inside; and a collapse there leaves a new edge that the next
 * round may take. So: any live edge with a masked end, whose midpoint is within
 * the radius plus that edge's length. O(triangles), once per stroke end.
 */
export function maskMeetsRemesh(
  masked: ArrayLike<number>,
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  center: readonly [number, number, number],
  radius: number,
): boolean {
  if (masked.length === 0) {
    return false;
  }
  const slots = positions.length / 3;
  const flags = new Uint8Array(slots);
  for (let k = 0; k < masked.length; k++) {
    if (masked[k] < slots) {
      flags[masked[k]] = 1;
    }
  }
  const reaches = (a: number, b: number) => {
    if (flags[a] === 0 && flags[b] === 0) {
      return false;
    }
    const [ax, ay, az] = [positions[3 * a], positions[3 * a + 1], positions[3 * a + 2]];
    const [bx, by, bz] = [positions[3 * b], positions[3 * b + 1], positions[3 * b + 2]];
    const d = Math.hypot((ax + bx) / 2 - center[0], (ay + by) / 2 - center[1], (az + bz) / 2 - center[2]);
    return d <= radius + Math.hypot(ax - bx, ay - by, az - bz);
  };
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (reaches(a, b) || reaches(b, c) || reaches(c, a)) {
      return true;
    }
  }
  return false;
}

/**
 * `op` with the field's vertices within its reach protected, or `op` itself
 * when it is not a brush or none are near. O(field) per dab.
 */
export function protectDab(op: RepairOp, field: MaskField, positions: ArrayLike<number>): RepairOp {
  if (op.op !== "inflate" && op.op !== "relax" && op.op !== "smooth" && op.op !== "grab") {
    return op;
  }
  const [cx, cy, cz] = op.center;
  const reach = op.radius * PROTECT_REACH;
  const r2 = reach * reach;
  const { ids, weights } = field;
  const near: number[] = [];
  for (let k = 0; k < ids.length; k++) {
    const v = ids[k];
    const dx = positions[3 * v] - cx;
    const dy = positions[3 * v + 1] - cy;
    const dz = positions[3 * v + 2] - cz;
    if (dx * dx + dy * dy + dz * dz <= r2) {
      near.push(k);
    }
  }
  if (near.length === 0) {
    return op;
  }
  const vertices = new Uint32Array(near.length);
  const scaled = new Float32Array(near.length);
  for (let k = 0; k < near.length; k++) {
    vertices[k] = ids[near[k]];
    scaled[k] = weights[near[k]];
  }
  return { ...op, protect: { vertices, weights: scaled } };
}
