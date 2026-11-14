// Per-vertex mask protection: what a dab may move near the painted mask.
//
// A dab goes to the engine with a `protect` list: the masked vertices it can
// reach at weight 0, so they stay byte-identical, and the free vertices in a
// soft band around the mask at a weight rising to 1, so no crease forms along
// the border. The brush itself is untouched; this only decorates its op.
import type { MeshChange, MeshHandle, RepairOp } from "fast-mesh-wasm";
import { reflectCenters, unionMaskIds, widestGrabReach } from "./sculpt";

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
  if (
    op.op !== "inflate" &&
    op.op !== "relax" &&
    op.op !== "smooth" &&
    op.op !== "grab" &&
    op.op !== "flatten" &&
    op.op !== "crease" &&
    op.op !== "clay" &&
    op.op !== "twist"
  ) {
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

export interface MaskLiveCache {
  slots: number;
  live: Uint8Array;
}

/** Materializes triangle-used vertex liveness once when no incremental cache exists. */
export function buildMaskLiveCache(indices: ArrayLike<number>, slots: number): MaskLiveCache {
  const live = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a >= 0 && a < slots) {
      live[a] = 1;
    }
    if (b >= 0 && b < slots) {
      live[b] = 1;
    }
    if (c >= 0 && c < slots) {
      live[c] = 1;
    }
  }
  return { slots, live };
}

/**
 * Updates `cache` incrementally over dirty triangle `[start, count]` ranges in
 * `O(dirty_ranges)`. Returns `null` when slots shrank or any dirty triangle is
 * a zeroed `(0, 0, 0)` removal slot that may have killed a vertex.
 */
export function updateMaskLiveFromRanges(
  cache: MaskLiveCache,
  indices: ArrayLike<number>,
  slots: number,
  ranges: readonly (readonly [number, number])[],
): MaskLiveCache | null {
  if (slots < cache.slots) {
    return null;
  }
  const triLimit = Math.floor(indices.length / 3);
  for (const [start, count] of ranges) {
    if (!Number.isInteger(start) || !Number.isInteger(count) || start < 0 || count < 0 || start + count > triLimit) {
      return null;
    }
    const end = start + count;
    for (let t = start; t < end; t++) {
      const i = 3 * t;
      const a = indices[i];
      const b = indices[i + 1];
      const c = indices[i + 2];
      if (
        (a === 0 && b === 0 && c === 0) ||
        !Number.isInteger(a) ||
        a < 0 ||
        a >= slots ||
        !Number.isInteger(b) ||
        b < 0 ||
        b >= slots ||
        !Number.isInteger(c) ||
        c < 0 ||
        c >= slots
      ) {
        return null;
      }
    }
  }
  let live = cache.live;
  if (slots > cache.slots) {
    live = new Uint8Array(slots);
    live.set(cache.live);
  }
  for (const [start, count] of ranges) {
    const end = start + count;
    for (let t = start; t < end; t++) {
      const i = 3 * t;
      live[indices[i]] = 1;
      live[indices[i + 1]] = 1;
      live[indices[i + 2]] = 1;
    }
  }
  return { slots, live };
}

/**
 * Updates `cache` in `O(splits)` across `splitStretched` capacity growth:
 * edge splits never kill existing live vertices and only append live midpoint
 * slots `[new, a, b]`.
 */
export function updateMaskLiveFromSplits(
  cache: MaskLiveCache,
  slots: number,
  splitTriples?: ArrayLike<number>,
): MaskLiveCache {
  const target = Math.max(cache.slots, slots);
  let live = cache.live;
  if (target > cache.slots) {
    live = new Uint8Array(target);
    live.set(cache.live);
    live.fill(1, cache.slots, target);
  }
  if (splitTriples !== undefined) {
    for (let k = 0; k < splitTriples.length; k++) {
      const id = splitTriples[k];
      if (id >= 0 && id < target) {
        live[id] = 1;
      }
    }
  }
  return { slots: target, live };
}

function opQueryRadius(
  op: Extract<
    RepairOp,
    { op: "inflate" | "relax" | "smooth" | "grab" | "flatten" | "crease" | "clay" | "twist" }
  >,
): number {
  if (op.op === "clay" && op.profile === "strips") {
    return op.radius * Math.sqrt(3);
  }
  if (op.op === "grab" && op.adaptiveReach === true) {
    return widestGrabReach(op.radius, op.falloff, op.method ?? "sphere", op.strength, op.direction);
  }
  return op.radius;
}

/**
 * Decorates a brush `op` with engine-localized soft-band protection (#197):
 * scans ONLY `masked` (`O(|masked|)` — zero triangle-index scans or CSR
 * builds), keeping live masked vertices near any active symmetry sphere at
 * weight `0` with `band = op.radius * MASK_SOFT_BAND`.
 */
export function protectMaskedDab(
  op: RepairOp,
  masked: ArrayLike<number>,
  positions: ArrayLike<number>,
  live: Uint8Array,
): RepairOp {
  if (
    op.op !== "inflate" &&
    op.op !== "relax" &&
    op.op !== "smooth" &&
    op.op !== "grab" &&
    op.op !== "flatten" &&
    op.op !== "crease" &&
    op.op !== "clay" &&
    op.op !== "twist"
  ) {
    return op;
  }
  if (masked.length === 0) {
    return op;
  }
  const band = op.radius * MASK_SOFT_BAND;
  const reach = (opQueryRadius(op) + band) * PROTECT_REACH;
  const r2 = reach * reach;
  const centers = reflectCenters(op.center, op.symmetry);
  const near: number[] = [];
  let prev = -1;
  for (let k = 0; k < masked.length; k++) {
    const id = masked[k];
    if (id === prev || id < 0 || id >= live.length || 3 * id + 2 >= positions.length) {
      continue;
    }
    if (live[id] === 0) {
      // Dead inherited splits (after undo) or stale slots cannot be passed to check_protection.
      continue;
    }
    const px = positions[3 * id];
    const py = positions[3 * id + 1];
    const pz = positions[3 * id + 2];
    let hit = false;
    for (let m = 0; m < centers.length; m++) {
      const c = centers[m];
      const dx = px - c[0];
      const dy = py - c[1];
      const dz = pz - c[2];
      if (dx * dx + dy * dy + dz * dz <= r2) {
        hit = true;
        break;
      }
    }
    if (hit) {
      near.push(id);
      prev = id;
    }
  }
  if (near.length === 0) {
    return op;
  }
  const vertices = Uint32Array.from(near);
  const weights = new Float32Array(vertices.length);
  return { ...op, protect: { vertices, weights, band } };
}

export interface MaskLiveScene {
  views(mesh: MeshHandle): { positions(): ArrayLike<number>; indices(): ArrayLike<number> };
  onChanges(listener: (changes: MeshChange[]) => void): () => void;
  status?(key: string): { status?: string };
  onStatus?(listener: () => void): () => void;
  undo?: () => unknown;
  redo?: () => unknown;
  cancelStep?: () => unknown;
  repair?: (...args: never[]) => unknown;
  edit?: (...args: never[]) => unknown;
  sculpt?: (...args: never[]) => unknown;
  compact?: (...args: never[]) => unknown;
  join?: (...args: never[]) => unknown;
  removeMesh?: (...args: never[]) => unknown;
}

export interface MaskLiveStore {
  subscribe?(listener: () => void): () => void;
  getSnapshot?(): { status?: string; history?: { status?: string; canUndo?: boolean; canRedo?: boolean } };
  undo?: () => void;
  redo?: () => void;
  repair?: (...args: never[]) => unknown;
  edit?: (...args: never[]) => unknown;
  sculpt?: (...args: never[]) => unknown;
  removeMesh?: (mesh: MeshHandle) => void;
}

export interface LiveMaskSnapshot {
  vertices: Uint32Array;
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
  live: Uint8Array;
}

export interface MaskPaintState {
  mesh: MeshHandle;
  vertices: Uint32Array;
  slots: number;
  inherited?: ReadonlySet<number>;
}

const EMPTY_LIVE = new Uint8Array(0);

const POSITION_OR_CAPTURE_OPS = new Set([
  "inflate",
  "relax",
  "smooth",
  "grab",
  "flatten",
  "crease",
  "clay",
  "twist",
  "captureRestSizes",
  "captureStrokeState",
]);

function isPositionOrCaptureSingle(op: unknown): boolean {
  return (
    typeof op === "object" &&
    op !== null &&
    typeof (op as { op?: unknown }).op === "string" &&
    POSITION_OR_CAPTURE_OPS.has((op as { op: string }).op)
  );
}

function isPositionOrCaptureOp(op: unknown): boolean {
  return Array.isArray(op) ? op.every(isPositionOrCaptureSingle) : isPositionOrCaptureSingle(op);
}

function isSplitOp(op: unknown): boolean {
  return !Array.isArray(op) && typeof op === "object" && op !== null && (op as { op?: unknown }).op === "splitStretched";
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (
    (typeof value === "object" || typeof value === "function") &&
    value !== null &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

function hookMethod(
  target: object,
  key: string,
  makeWrapper: (orig: (...args: unknown[]) => unknown) => (...args: unknown[]) => unknown,
): (() => void) | null {
  const rec = target as Record<string, unknown>;
  const fn = rec[key];
  if (typeof fn !== "function") {
    return null;
  }
  const orig = fn as (...args: unknown[]) => unknown;
  const existed = Object.prototype.hasOwnProperty.call(target, key);
  const prevDesc = Object.getOwnPropertyDescriptor(target, key);
  const wrapper = makeWrapper(orig);
  try {
    Object.defineProperty(target, key, {
      value: wrapper,
      configurable: true,
      writable: true,
      enumerable: prevDesc?.enumerable ?? false,
    });
  } catch {
    return null;
  }
  return () => {
    const curDesc = Object.getOwnPropertyDescriptor(target, key);
    if (curDesc?.value !== wrapper) {
      return;
    }
    if (existed && prevDesc !== undefined) {
      Object.defineProperty(target, key, prevDesc);
    } else {
      Reflect.deleteProperty(target, key);
    }
  };
}

interface PendingSplit {
  prevSlots: number | null;
  expectedSlots?: number;
  expectedTris?: number;
}

/**
 * Tracks live triangle vertex flags across `Scene` / `WorkerScene` frames:
 * materializes `buildMaskLiveCache` at most once per untrusted non-empty mask,
 * updates incrementally across `splitStretched` (including `commitFrame()` runs
 * after `splitStretched` settles) and partial dirty triangle ranges, and lets
 * mask painting mark freshly queried `verticesInSphere` masks live in O(1).
 *
 * Construct with no side effects; call `attach()` / `dispose()` in `useEffect`
 * so React `<StrictMode>` unmount/remount cycles re-subscribe cleanly.
 */
export class MaskLiveTracker {
  readonly #scene: MaskLiveScene;
  readonly #store: MaskLiveStore | null;
  #off: (() => void) | null = null;
  #cache: MaskLiveCache | null = null;
  #cacheMesh: MeshHandle | null = null;
  #checkedMask: Uint32Array | null = null;
  #trustedMesh: MeshHandle | null = null;
  #trustedSlots = 0;
  #trustedClean = false;
  #trustedLive: Uint8Array | null = null;
  #trustedLiveMesh: MeshHandle | null = null;
  #trustedLiveMask: Uint32Array | null = null;
  #splittingMesh: MeshHandle | null = null;
  readonly #splitStartSlots = new Map<MeshHandle, number | null>();
  readonly #splitPending = new Map<MeshHandle, PendingSplit>();
  readonly #splitSuppressed = new Set<MeshHandle>();
  readonly #topoVersion = new Map<MeshHandle, number>();
  #globalTopoEpoch = 0;
  #pendingAsyncHistory = 0;

  constructor(scene: MaskLiveScene, store?: MaskLiveStore) {
    this.#scene = scene;
    this.#store = store ?? null;
  }

  #isHistoryComputing(): boolean {
    const historyStatus = this.#scene.status?.("history")?.status;
    if (historyStatus !== undefined) {
      return historyStatus === "computing";
    }
    const snap = this.#store?.getSnapshot?.();
    if (snap?.history?.status !== undefined) {
      return snap.history.status === "computing";
    }
    return snap?.status === "computing";
  }

  attach(): void {
    if (this.#off !== null) {
      return;
    }
    this.invalidate();
    this.#splitSuppressed.clear();
    this.#pendingAsyncHistory = 0;
    this.#globalTopoEpoch++;

    const cleanups: (() => void)[] = [];
    cleanups.push(this.#scene.onChanges((changes) => this.#onChanges(changes)));

    const addHook = (target: object | null, key: string, wrap: (orig: (...args: unknown[]) => unknown) => (...args: unknown[]) => unknown) => {
      if (target === null) {
        return;
      }
      const unhook = hookMethod(target, key, wrap);
      if (unhook !== null) {
        cleanups.push(unhook);
      }
    };

    const wrapGlobalMutation = (target: object, trackAsyncStatus: boolean) => (orig: (...args: unknown[]) => unknown) =>
      function (this: unknown, ...args: unknown[]): unknown {
        if (trackAsyncStatus) {
          tracker.#pendingAsyncHistory++;
        }
        tracker.noteTopologyMutation();
        try {
          const result = orig.apply(this ?? target, args);
          if (trackAsyncStatus && !tracker.#isHistoryComputing()) {
            tracker.#pendingAsyncHistory = Math.max(0, tracker.#pendingAsyncHistory - 1);
          }
          if (isPromiseLike(result)) {
            void Promise.resolve(result).then(
              () => tracker.noteTopologyMutation(),
              () => tracker.noteTopologyMutation(),
            );
          }
          return result;
        } catch (err) {
          if (trackAsyncStatus) {
            tracker.#pendingAsyncHistory = Math.max(0, tracker.#pendingAsyncHistory - 1);
          }
          throw err;
        }
      };

    const wrapMeshMutation = (target: object) => (orig: (...args: unknown[]) => unknown) =>
      function (this: unknown, ...args: unknown[]): unknown {
        const mesh = args[0] as MeshHandle | undefined;
        tracker.noteTopologyMutation(mesh);
        const result = orig.apply(this ?? target, args);
        if (isPromiseLike(result)) {
          void Promise.resolve(result).then(
            () => tracker.noteTopologyMutation(mesh),
            () => tracker.noteTopologyMutation(mesh),
          );
        }
        return result;
      };

    const wrapRepairOrSculpt = (target: object) => (orig: (...args: unknown[]) => unknown) =>
      function (this: unknown, ...args: unknown[]): unknown {
        const mesh = args[0] as MeshHandle | undefined;
        const op = args[1];
        if (isPositionOrCaptureOp(op) || isSplitOp(op)) {
          if (mesh !== undefined) {
            tracker.#bumpTopo(mesh);
          } else {
            tracker.#globalTopoEpoch++;
          }
          const result = orig.apply(this ?? target, args);
          if (isPromiseLike(result)) {
            void Promise.resolve(result).then(
              () => {
                if (mesh !== undefined) {
                  tracker.#bumpTopo(mesh);
                } else {
                  tracker.#globalTopoEpoch++;
                }
              },
              () => {},
            );
          }
          return result;
        }
        tracker.noteTopologyMutation(mesh);
        const result = orig.apply(this ?? target, args);
        if (isPromiseLike(result)) {
          void Promise.resolve(result).then(
            () => tracker.noteTopologyMutation(mesh),
            () => tracker.noteTopologyMutation(mesh),
          );
        }
        return result;
      };

    const tracker = this;
    addHook(this.#scene, "undo", wrapGlobalMutation(this.#scene, false));
    addHook(this.#scene, "redo", wrapGlobalMutation(this.#scene, false));
    addHook(this.#scene, "cancelStep", wrapGlobalMutation(this.#scene, false));
    addHook(this.#scene, "join", wrapGlobalMutation(this.#scene, false));
    addHook(this.#scene, "compact", wrapMeshMutation(this.#scene));
    addHook(this.#scene, "edit", wrapMeshMutation(this.#scene));
    addHook(this.#scene, "removeMesh", wrapMeshMutation(this.#scene));
    addHook(this.#scene, "repair", wrapRepairOrSculpt(this.#scene));
    addHook(this.#scene, "sculpt", wrapRepairOrSculpt(this.#scene));

    const clearHistoryIfDone = () => {
      if (this.#pendingAsyncHistory > 0) {
        this.noteTopologyMutation();
        if (!this.#isHistoryComputing()) {
          this.#pendingAsyncHistory = 0;
        }
      }
    };

    if (typeof this.#scene.onStatus === "function") {
      cleanups.push(this.#scene.onStatus(clearHistoryIfDone));
    }

    if (this.#store !== null) {
      addHook(this.#store, "undo", wrapGlobalMutation(this.#store, true));
      addHook(this.#store, "redo", wrapGlobalMutation(this.#store, true));
      addHook(this.#store, "edit", wrapMeshMutation(this.#store));
      addHook(this.#store, "removeMesh", wrapMeshMutation(this.#store));
      addHook(this.#store, "repair", wrapRepairOrSculpt(this.#store));
      addHook(this.#store, "sculpt", wrapRepairOrSculpt(this.#store));
      if (typeof this.#store.subscribe === "function") {
        cleanups.push(this.#store.subscribe(clearHistoryIfDone));
      }
    }

    this.#off = () => {
      for (let i = cleanups.length - 1; i >= 0; i--) {
        cleanups[i]();
      }
    };
  }

  dispose(): void {
    if (this.#off !== null) {
      const off = this.#off;
      this.#off = null;
      off();
    }
    this.#pendingAsyncHistory = 0;
    this.invalidate();
    this.#splitSuppressed.clear();
    this.#globalTopoEpoch++;
  }

  topologyVersion(mesh: MeshHandle): number {
    return (this.#topoVersion.get(mesh) ?? 0) + this.#globalTopoEpoch;
  }

  #bumpTopo(mesh: MeshHandle): void {
    this.#topoVersion.set(mesh, (this.#topoVersion.get(mesh) ?? 0) + 1);
  }

  /**
   * Synchronously invalidates cached/trusted masks and increments the topology
   * version before `commitFrame()` flushes `onChanges`.
   */
  noteTopologyMutation(mesh?: MeshHandle): void {
    if (mesh === undefined) {
      this.#globalTopoEpoch++;
      if (this.#splittingMesh !== null) {
        this.#splitSuppressed.add(this.#splittingMesh);
      }
      for (const pendingMesh of this.#splitPending.keys()) {
        this.#splitSuppressed.add(pendingMesh);
      }
    } else {
      this.#bumpTopo(mesh);
      if (this.#splittingMesh === mesh || this.#splitPending.has(mesh)) {
        this.#splitSuppressed.add(mesh);
      }
    }
    this.invalidate(mesh);
  }

  #clearCacheAndTrust(mesh?: MeshHandle): void {
    if (mesh === undefined || this.#cacheMesh === mesh) {
      this.#cache = null;
      this.#cacheMesh = null;
      this.#checkedMask = null;
    }
    if (mesh === undefined || this.#trustedMesh === mesh) {
      this.#trustedMesh = null;
      this.#trustedClean = false;
      this.#clearTrustedLive();
      if (this.#cacheMesh === null) {
        this.#checkedMask = null;
      }
    }
  }

  invalidate(mesh?: MeshHandle): void {
    this.#clearCacheAndTrust(mesh);
    if (mesh === undefined) {
      if (this.#splittingMesh !== null) {
        this.#splitSuppressed.add(this.#splittingMesh);
      }
      this.#splittingMesh = null;
      this.#splitStartSlots.clear();
      this.#splitPending.clear();
    } else {
      if (this.#splittingMesh === mesh) {
        this.#splitSuppressed.add(mesh);
        this.#splittingMesh = null;
      }
      this.#splitStartSlots.delete(mesh);
      this.#splitPending.delete(mesh);
    }
  }

  beginSplit(mesh: MeshHandle): void {
    this.#splitSuppressed.delete(mesh);
    this.#splittingMesh = mesh;
    let prevSlots: number | null = null;
    try {
      prevSlots = this.#scene.views(mesh).positions().length / 3;
    } catch {
      prevSlots =
        this.#cacheMesh === mesh && this.#cache !== null
          ? this.#cache.slots
          : this.#trustedMesh === mesh
            ? this.#trustedSlots
            : null;
    }
    this.#splitStartSlots.set(mesh, prevSlots);
  }

  endSplit(mesh: MeshHandle, changed: boolean, slots?: number, splitVertices?: ArrayLike<number>): void {
    const prevSlots = this.#splitStartSlots.get(mesh) ?? null;
    this.#splitStartSlots.delete(mesh);
    if (this.#splittingMesh === mesh) {
      this.#splittingMesh = null;
    }
    const suppressed = this.#splitSuppressed.has(mesh);
    if (!changed || suppressed) {
      this.#splitPending.delete(mesh);
      if (suppressed) {
        this.#clearCacheAndTrust(mesh);
      }
      return;
    }
    let expectedTris: number | undefined;
    try {
      expectedTris = Math.floor(this.#scene.views(mesh).indices().length / 3);
    } catch {
      expectedTris = undefined;
    }
    this.#splitPending.set(mesh, { prevSlots, expectedSlots: slots, expectedTris });
    if (this.#cache !== null && this.#cacheMesh === mesh && slots !== undefined) {
      this.#cache = updateMaskLiveFromSplits(this.#cache, slots, splitVertices);
      if (this.#trustedMesh === mesh) {
        this.#trustedSlots = this.#cache.slots;
      }
    } else if (this.#trustedMesh === mesh && slots !== undefined) {
      this.#trustedSlots = Math.max(this.#trustedSlots, slots);
      this.#clearTrustedLive();
    }
  }

  trust(mesh: MeshHandle, vertices: Uint32Array, slots: number, clean = true): void {
    this.#trustedMesh = mesh;
    this.#trustedSlots = slots;
    this.#checkedMask = vertices;
    this.#trustedClean = clean;
    this.#clearTrustedLive();
  }

  trustSplits(mesh: MeshHandle, prevVertices: Uint32Array, nextVertices: Uint32Array, slots: number): void {
    if (this.#trustedMesh !== mesh || this.#checkedMask !== prevVertices || !this.#trustedClean) {
      return;
    }
    this.#trustedSlots = slots;
    this.#checkedMask = nextVertices;
    this.#clearTrustedLive();
  }

  #clearTrustedLive(): void {
    this.#trustedLive = null;
    this.#trustedLiveMesh = null;
    this.#trustedLiveMask = null;
  }

  #ensureTrustedLive(mesh: MeshHandle, vertices: Uint32Array, slots: number): Uint8Array {
    if (
      this.#trustedLive !== null &&
      this.#trustedLive.length === slots &&
      this.#trustedLiveMesh === mesh &&
      this.#trustedLiveMask === vertices
    ) {
      return this.#trustedLive;
    }
    const live = new Uint8Array(slots);
    for (let k = 0; k < vertices.length; k++) {
      const id = vertices[k];
      if (id >= 0 && id < slots) {
        live[id] = 1;
      }
    }
    this.#trustedLive = live;
    this.#trustedLiveMesh = mesh;
    this.#trustedLiveMask = vertices;
    return live;
  }

  checkMask(
    mesh: MeshHandle,
    mask: { mesh: MeshHandle; vertices: Uint32Array; inherited?: ReadonlySet<number> } | null,
  ): { ok: boolean; slots: number } | null {
    if (mask === null || mask.mesh !== mesh) {
      return null;
    }
    let positions: ArrayLike<number>;
    let indices: ArrayLike<number>;
    try {
      const v = this.#scene.views(mesh);
      positions = v.positions();
      indices = v.indices();
    } catch {
      return null;
    }
    const slots = positions.length / 3;
    if (mask.vertices.length === 0) {
      return { ok: true, slots };
    }
    if (this.#trustedMesh === mesh && this.#trustedSlots === slots && this.#checkedMask === mask.vertices) {
      return { ok: true, slots };
    }
    const snap = this.#snapshotWith(mesh, mask, positions, indices, slots);
    if (snap === null) {
      return null;
    }
    return { ok: snap !== "stale", slots };
  }

  liveSnapshot(
    mesh: MeshHandle,
    mask: { mesh: MeshHandle; vertices: Uint32Array; inherited?: ReadonlySet<number> } | null,
  ): LiveMaskSnapshot | "stale" | null {
    if (mask === null || mask.mesh !== mesh) {
      return null;
    }
    let positions: ArrayLike<number>;
    let indices: ArrayLike<number>;
    try {
      const v = this.#scene.views(mesh);
      positions = v.positions();
      indices = v.indices();
    } catch {
      return null;
    }
    const slots = positions.length / 3;
    if (mask.vertices.length === 0) {
      const live =
        this.#cache !== null && this.#cacheMesh === mesh && this.#cache.slots === slots ? this.#cache.live : EMPTY_LIVE;
      return { vertices: mask.vertices, positions, indices, live };
    }
    return this.#snapshotWith(mesh, mask, positions, indices, slots);
  }

  #snapshotWith(
    mesh: MeshHandle,
    mask: { vertices: Uint32Array; inherited?: ReadonlySet<number> },
    positions: ArrayLike<number>,
    indices: ArrayLike<number>,
    slots: number,
  ): LiveMaskSnapshot | "stale" | null {
    if (this.#cache === null || this.#cacheMesh !== mesh || this.#cache.slots !== slots) {
      if (
        this.#trustedMesh === mesh &&
        this.#trustedSlots === slots &&
        this.#checkedMask === mask.vertices &&
        (this.#trustedClean || mask.inherited === undefined || mask.inherited.size === 0)
      ) {
        return {
          vertices: mask.vertices,
          positions,
          indices,
          live: this.#ensureTrustedLive(mesh, mask.vertices, slots),
        };
      }
      this.#checkedMask = null;
      this.#trustedMesh = null;
      this.#trustedClean = false;
      this.#clearTrustedLive();
      this.#cache = buildMaskLiveCache(indices, slots);
      this.#cacheMesh = mesh;
    }
    if (this.#checkedMask !== mask.vertices) {
      for (let k = 0; k < mask.vertices.length; k++) {
        const id = mask.vertices[k];
        if ((id < 0 || id >= this.#cache.live.length || this.#cache.live[id] === 0) && mask.inherited?.has(id) !== true) {
          return "stale";
        }
      }
      this.#checkedMask = mask.vertices;
      this.#trustedMesh = mesh;
      this.#trustedSlots = slots;
      this.#trustedClean = mask.inherited === undefined || mask.inherited.size === 0;
    }
    return { vertices: mask.vertices, positions, indices, live: this.#cache.live };
  }

  #onChanges(changes: readonly MeshChange[]): void {
    for (const c of changes) {
      if (c.removed) {
        this.#clearCacheAndTrust(c.mesh);
        if (this.#splittingMesh === c.mesh) {
          this.#splittingMesh = null;
        }
        this.#splitStartSlots.delete(c.mesh);
        this.#splitPending.delete(c.mesh);
        this.#splitSuppressed.delete(c.mesh);
        this.#topoVersion.delete(c.mesh);
        continue;
      }
      if (!c.capacityChanged && c.indices !== "all" && c.indices.length === 0) {
        continue;
      }
      this.#bumpTopo(c.mesh);
      const pending = this.#splitPending.get(c.mesh);
      this.#splitPending.delete(c.mesh);
      const suppressed = this.#splitSuppressed.delete(c.mesh);
      const activeSplit = this.#splittingMesh === c.mesh;
      const fromSplit = !suppressed && (activeSplit || pending !== undefined);

      if (fromSplit) {
        // An "all" rebuild without capacity growth (undo back to same capacity,
        // remesh, overflow, or coalesced rebuild) cannot come from a pure split.
        if (c.indices === "all" && !c.capacityChanged) {
          this.invalidate(c.mesh);
          continue;
        }
        try {
          const v = this.#scene.views(c.mesh);
          const slots = v.positions().length / 3;
          const indices = v.indices();
          const triSlots = Math.floor(indices.length / 3);
          const startSlots = pending !== undefined ? pending.prevSlots : (this.#splitStartSlots.get(c.mesh) ?? null);
          const splitDimsMatch =
            pending !== undefined
              ? pending.expectedSlots !== undefined &&
                slots === pending.expectedSlots &&
                (pending.expectedTris === undefined || triSlots === pending.expectedTris) &&
                (startSlots === null || slots > startSlots)
              : startSlots !== null && slots > startSlots;

          if (c.indices === "all") {
            if (!splitDimsMatch) {
              this.invalidate(c.mesh);
              continue;
            }
            if (this.#cacheMesh === c.mesh && this.#cache !== null) {
              this.#cache = updateMaskLiveFromSplits(this.#cache, slots);
              if (this.#trustedMesh === c.mesh) {
                this.#trustedSlots = this.#cache.slots;
              }
            } else if (this.#trustedMesh === c.mesh) {
              this.#trustedSlots = slots;
              this.#clearTrustedLive();
            }
            continue;
          }

          // Dirty triangle ranges alongside split: never skip range validation.
          if (
            pending !== undefined &&
            ((pending.expectedSlots !== undefined && slots !== pending.expectedSlots) ||
              (pending.expectedTris !== undefined && triSlots !== pending.expectedTris))
          ) {
            this.invalidate(c.mesh);
            continue;
          }
          if (this.#trustedMesh === c.mesh && c.indices.length > 0) {
            this.#trustedMesh = null;
            this.#trustedClean = false;
            this.#clearTrustedLive();
            this.#checkedMask = null;
          }
          if (this.#cacheMesh === c.mesh && this.#cache !== null) {
            const grown = updateMaskLiveFromSplits(this.#cache, slots);
            const updated = updateMaskLiveFromRanges(grown, indices, slots, c.indices);
            if (updated === null) {
              this.invalidate(c.mesh);
              continue;
            }
            this.#cache = updated;
            continue;
          }
          if (c.indices.length > 0) {
            this.invalidate(c.mesh);
            continue;
          }
        } catch {
          this.invalidate(c.mesh);
          continue;
        }
      }

      if (c.indices === "all" || c.capacityChanged) {
        this.invalidate(c.mesh);
        continue;
      }
      if (this.#trustedMesh === c.mesh) {
        this.#trustedMesh = null;
        this.#trustedClean = false;
        this.#clearTrustedLive();
        this.#checkedMask = null;
      }
      if (this.#cacheMesh === c.mesh && this.#cache !== null) {
        try {
          const v = this.#scene.views(c.mesh);
          const updated = updateMaskLiveFromRanges(this.#cache, v.indices(), v.positions().length / 3, c.indices);
          this.#cache = updated;
          if (updated === null) {
            this.invalidate(c.mesh);
          }
        } catch {
          this.invalidate(c.mesh);
        }
      }
    }
  }
}

/**
 * Executes a single serialized mask-paint step against `verticesInSphere`, checking
 * `tracker.topologyVersion(mesh)` across the async `await` BEFORE touching
 * `views(mesh)`, `checkMask`, or `trust` so an undo/redo/cancelStep/non-split
 * repair settling in the microtask window before `commitFrame()` cannot trust
 * stale vertex IDs.
 */
export async function applyMaskPaintStep(params: {
  mesh: MeshHandle;
  centers: readonly (readonly [number, number, number])[];
  radius: number;
  verticesInSphere: (mesh: MeshHandle, center: readonly [number, number, number], radius: number) => Promise<Uint32Array>;
  views: (mesh: MeshHandle) => { positions(): ArrayLike<number> };
  tracker: MaskLiveTracker;
  isAlive: () => boolean;
  getPrevMask: () => MaskPaintState | null;
  setMask: (next: MaskPaintState | null) => void;
  onStale: () => void;
  clearNotice?: () => void;
}): Promise<boolean> {
  const { mesh, centers, radius, verticesInSphere, views, tracker, isAlive, getPrevMask, setMask, onStale, clearNotice } =
    params;
  const epochBefore = tracker.topologyVersion(mesh);
  let picked: Uint32Array[];
  try {
    picked = await Promise.all(centers.map((c) => verticesInSphere(mesh, c, radius)));
  } catch {
    return false;
  }
  if (!isAlive() || epochBefore !== tracker.topologyVersion(mesh)) {
    return false;
  }
  let slots: number;
  try {
    slots = views(mesh).positions().length / 3;
  } catch {
    return false;
  }
  if (epochBefore !== tracker.topologyVersion(mesh)) {
    return false;
  }
  let fresh = picked[0] ?? new Uint32Array(0);
  for (let k = 1; k < picked.length; k++) {
    fresh = unionMaskIds(fresh, picked[k]);
  }
  const prev = getPrevMask();
  if (prev !== null && prev.mesh === mesh) {
    const check = tracker.checkMask(mesh, prev);
    if (check === null || epochBefore !== tracker.topologyVersion(mesh)) {
      return false;
    }
    if (!check.ok) {
      // Loud, never a silent reinterpretation: drop it, count it, paint fresh.
      tracker.invalidate(mesh);
      onStale();
      if (epochBefore === tracker.topologyVersion(mesh)) {
        tracker.trust(mesh, fresh, slots);
      }
      setMask({ mesh, vertices: fresh, slots });
      return true;
    }
    const nextVertices = unionMaskIds(prev.vertices, fresh);
    tracker.trust(mesh, nextVertices, slots, prev.inherited === undefined || prev.inherited.size === 0);
    setMask({ mesh, vertices: nextVertices, slots, inherited: prev.inherited });
  } else {
    tracker.trust(mesh, fresh, slots);
    setMask({ mesh, vertices: fresh, slots });
  }
  clearNotice?.();
  return true;
}
