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

/** Past this many cells around a dab (a brush far wider than the field's), walking the field is cheaper. */
const MAX_CELLS = 512;

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
  /** Per id, where it rested at the build (`xyz`): masked ones must still be there. */
  rest: Float32Array;
  /** The ids by cell, so a dab finds those near it without walking them all. `null` without a band. */
  grid: FieldGrid | null;
}

/** A uniform grid over the field's rest positions, in `[starts[c], starts[c + 1])` of `items`. */
interface FieldGrid {
  size: number;
  cells: Map<number, number>;
  starts: Uint32Array;
  /** Indices into the field's `ids`. */
  items: Uint32Array;
}

const cellKey = (x: number, y: number, z: number): number => (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);

function fieldGrid(rest: Float32Array, size: number): FieldGrid {
  const n = rest.length / 3;
  const cells = new Map<number, number>();
  const slotOf = new Uint32Array(n);
  const counts: number[] = [];
  for (let k = 0; k < n; k++) {
    const key = cellKey(Math.floor(rest[3 * k] / size), Math.floor(rest[3 * k + 1] / size), Math.floor(rest[3 * k + 2] / size));
    let c = cells.get(key);
    if (c === undefined) {
      c = counts.length;
      cells.set(key, c);
      counts.push(0);
    }
    counts[c]++;
    slotOf[k] = c;
  }
  const starts = new Uint32Array(counts.length + 1);
  for (let c = 0; c < counts.length; c++) {
    starts[c + 1] = starts[c] + counts[c];
  }
  const fill = starts.slice(0, counts.length);
  const items = new Uint32Array(n);
  for (let k = 0; k < n; k++) {
    items[fill[slotOf[k]]++] = k;
  }
  return { size, cells, starts, items };
}

/**
 * Whether every masked vertex is still where the field took it. A dab finds them
 * by that position, so a move by anything else (a repair, an undo) would hide them
 * from it. Own dabs never move them. O(masked): a stroke asks once, at its press.
 */
export function maskFieldCurrent(field: MaskField, positions: ArrayLike<number>): boolean {
  const { ids, flags, rest } = field;
  for (let k = 0; k < ids.length; k++) {
    const v = ids[k];
    if (
      flags[v] === 1 &&
      (positions[3 * v] !== rest[3 * k] || positions[3 * v + 1] !== rest[3 * k + 1] || positions[3 * v + 2] !== rest[3 * k + 2])
    ) {
      return false;
    }
  }
  return true;
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
  const rest = new Float32Array(3 * count);
  for (let k = 0; k < count; k++) {
    rest[3 * k] = positions[3 * ids[k]];
    rest[3 * k + 1] = positions[3 * ids[k] + 1];
    rest[3 * k + 2] = positions[3 * ids[k] + 2];
  }
  // A cell as wide as a dab's reach: a dab looks at the 27 cells around it.
  const grid = band > 0 ? fieldGrid(rest, (band / MASK_SOFT_BAND) * PROTECT_REACH) : null;
  return { flags, ids, weights, scale, rest, grid };
}

/**
 * The vertices a split added between two held ones (`new, a, b` triples, as
 * `splitStretched` reports them, in order, so a parent can be new itself). A
 * masked edge splits when it stretched before the mask was painted. Left free,
 * the new vertex would move under the next dab and the mask would not hold.
 */
export function inheritedSplits(held: (id: number) => boolean, splits: ArrayLike<number>): number[] {
  const fresh: number[] = [];
  const added = new Set<number>();
  const isHeld = (id: number) => added.has(id) || held(id);
  for (let k = 0; k + 2 < splits.length; k += 3) {
    if (isHeld(splits[k + 1]) && isHeld(splits[k + 2])) {
      added.add(splits[k]);
      fresh.push(splits[k]);
    }
  }
  return fresh;
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
 * when it is not a brush or none are near. The cost follows the field's
 * vertices around the brush, not the whole field: masked vertices rest where the
 * grid has them (`maskFieldCurrent`), and a band vertex a stroke moved out of
 * its cell only loses some softness.
 */
export function protectDab(op: RepairOp, field: MaskField, positions: ArrayLike<number>): RepairOp {
  if (op.op !== "inflate" && op.op !== "relax" && op.op !== "smooth" && op.op !== "grab") {
    return op;
  }
  const [cx, cy, cz] = op.center;
  const reach = op.radius * PROTECT_REACH;
  const r2 = reach * reach;
  const { ids, weights, grid } = field;
  const near: number[] = [];
  const consider = (k: number) => {
    const v = ids[k];
    const dx = positions[3 * v] - cx;
    const dy = positions[3 * v + 1] - cy;
    const dz = positions[3 * v + 2] - cz;
    if (dx * dx + dy * dy + dz * dz <= r2) {
      near.push(k);
    }
  };
  const span = grid === null ? Number.POSITIVE_INFINITY : (2 * reach) / grid.size + 2;
  if (grid === null || span * span * span > MAX_CELLS) {
    for (let k = 0; k < ids.length; k++) {
      consider(k);
    }
  } else {
    // Cells can share a key: each is walked once.
    const seen = new Set<number>();
    for (let x = Math.floor((cx - reach) / grid.size); x <= Math.floor((cx + reach) / grid.size); x++) {
      for (let y = Math.floor((cy - reach) / grid.size); y <= Math.floor((cy + reach) / grid.size); y++) {
        for (let z = Math.floor((cz - reach) / grid.size); z <= Math.floor((cz + reach) / grid.size); z++) {
          const c = grid.cells.get(cellKey(x, y, z));
          if (c !== undefined && !seen.has(c)) {
            seen.add(c);
            for (let i = grid.starts[c]; i < grid.starts[c + 1]; i++) {
              consider(grid.items[i]);
            }
          }
        }
      }
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
