// The sculpt stroke: what one dab applies, and one press-and-hold stroke
// over the meshes (the viewport's ring and pointer loop drive it).
import type { FastMeshErrorCode, MeshHandle, RepairOp } from "fast-mesh-wasm";

/** A sculpt brush armed on a mesh (app-state `sculpt`). */
export interface SculptBrush {
  mesh: MeshHandle;
  op: "inflate" | "relax" | "grab";
  radius: number;
  strength: number;
  falloff: "linear" | "smooth";
  cycles: number;
  /** The grab drag vector in local coordinates (the grab tool only). */
  direction: [number, number, number];
}

/** The toolbox's brush picker (the viewport paints its strokes). */
export type SculptTool = "inflate" | "deflate" | "smooth" | "grab";

/**
 * The repair op a tool paints (deflate is inflate with a flipped strength;
 * smooth paints plain-Laplacian relax: Taubin λ/μ converges too slowly to
 * melt under a brush).
 */
export function sculptOpForTool(tool: SculptTool): "inflate" | "relax" | "grab" {
  if (tool === "smooth") {
    return "relax";
  }
  return tool === "grab" ? "grab" : "inflate";
}

/**
 * Each tool's starting strength (grab at 1 follows the pointer exactly).
 * Inflate stays small: dabs overlap heavily along a sweep, so a dab-sized
 * 0.05 already builds a visible bump without folding.
 */
export const DEFAULT_STRENGTHS: Record<SculptTool, number> = { inflate: 0.05, deflate: 0.05, smooth: 0.5, grab: 1 };

/** Laplacian passes per smooth dab (each melts the full weight toward the centroid). */
export const RELAX_PASSES = 2;

/**
 * Arms (or steers) the sculpt brush for `tool` with a dab-sized strength.
 * The toolbox calls it live, so the radius/strength sliders stay adjustable
 * mid-stroke (the stroke reads its brush through a getter).
 */
export function sculptBrush(
  tool: SculptTool,
  mesh: MeshHandle,
  radius: number,
  strength: number,
  falloff: "linear" | "smooth",
): SculptBrush {
  return {
    mesh,
    op: sculptOpForTool(tool),
    radius,
    strength: tool === "deflate" ? -Math.abs(strength) : Math.abs(strength),
    falloff,
    cycles: RELAX_PASSES,
    direction: [0, 0, 0],
  };
}

/** Past this many spacings a dab jumps to the pointer instead of creeping (see `SculptStroke.dab`). */
export const CREEP_SPACINGS = 4;

/**
 * Past this many triangles sculpt remeshes stand down: the passes run
 * over the whole mesh (measured ~2.6 s floor on 300 k, minutes for wide
 * regions), which would stall the worker and the dabs behind it. Dabs
 * themselves stay fluid (~20-30 ms), so sculpting keeps working.
 */
export const REMESH_TRI_LIMIT = 100_000;

/** Whether `triangles` stands sculpt remeshes (live and stroke-end) down. */
export function remeshTooBig(triangles: number): boolean {
  return triangles > REMESH_TRI_LIMIT;
}

/**
 * Live remesh while sculpting (#129): the per-frame budget.
 *
 * At most one live pass per stroke region per `LIVE_REMESH_MIN_MS`, and only
 * once `LIVE_REMESH_MIN_DABS` dabs landed since the last pass, never two in
 * flight. The pass itself is split-only (`splitOnly`, `maxRounds: 2` at the
 * call site), so the triangle count only grows mid-stroke and the surface
 * never pops down; the stroke-end pass evens the density out fully. Past
 * `REMESH_TRI_LIMIT` triangles both passes stand down: a whole-mesh remesh
 * would stall the worker behind the dabs (#64 chunks a repair list one op
 * per message-loop turn, but a single remesh op is still one long chunk).
 */
export const LIVE_REMESH_MIN_DABS = 4;
/** Minimum milliseconds between two live remesh passes of one stroke. */
export const LIVE_REMESH_MIN_MS = 600;

/**
 * Whether a live remesh pass may fire: enough new dabs since the last pass,
 * enough time since it, and none still in flight.
 */
export function liveRemeshDue(dabsSinceLive: number, msSinceLive: number, running: boolean): boolean {
  return !running && dabsSinceLive >= LIVE_REMESH_MIN_DABS && msSinceLive >= LIVE_REMESH_MIN_MS;
}

/**
 * Undo semantics for sculpt remeshes (the #129 open question): SAME step.
 *
 * Both the live passes and the stroke-end evening-out land inside the
 * stroke's open undo step on the sync scene, so one Ctrl+Z takes the whole
 * gesture (dabs and density fixes) back. A separate remesh step would fight
 * the stroke: undoing the stroke would leave its splits behind, and undoing
 * the remesh first would strand the dabs on a density they never painted.
 * The live pass must therefore never open or close a step itself; it just
 * repairs while the stroke's step is open. (On the Worker there is no step
 * nesting — one step per dab — so the passes land as their own steps
 * there; that divergence is inherent to the Worker scene, not a choice.)
 */

/** Dab spacing: 15% of the radius (never zero, so dabs terminate). */
export function dabSpacing(radius: number): number {
  return Math.max(radius * 0.15, 1e-6);
}

/** Whether `at` is at least `spacing` from the last dab: a new dab. */
export function farEnough(
  last: readonly [number, number, number],
  at: readonly [number, number, number],
  spacing: number,
): boolean {
  const dx = at[0] - last[0];
  const dy = at[1] - last[1];
  const dz = at[2] - last[2];
  return dx * dx + dy * dy + dz * dz >= spacing * spacing;
}

/** The repair op one dab applies at `center` (local coordinates). */
export function dabOp(brush: SculptBrush, center: readonly [number, number, number]): RepairOp {
  const shared = {
    center: [center[0], center[1], center[2]] as [number, number, number],
    radius: brush.radius,
    falloff: brush.falloff,
    strength: brush.strength,
  };
  if (brush.op === "relax") {
    return { op: "relax", ...shared, cycles: brush.cycles };
  }
  if (brush.op === "grab") {
    return { op: "grab", ...shared, direction: [...brush.direction] as [number, number, number] };
  }
  return { op: "inflate", ...shared };
}

/** The unit face normal of the triangle `(a, b, c)` over flat positions. */
export function faceNormal(positions: ArrayLike<number>, a: number, b: number, c: number): [number, number, number] {
  const ux = positions[3 * b] - positions[3 * a];
  const uy = positions[3 * b + 1] - positions[3 * a + 1];
  const uz = positions[3 * b + 2] - positions[3 * a + 2];
  const vx = positions[3 * c] - positions[3 * a];
  const vy = positions[3 * c + 1] - positions[3 * a + 1];
  const vz = positions[3 * c + 2] - positions[3 * a + 2];
  const n: [number, number, number] = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
  const len = Math.hypot(n[0], n[1], n[2]);
  return len === 0 ? [0, 0, 1] : [n[0] / len, n[1] / len, n[2] / len];
}

/** World units per CSS pixel at `distance` from a perspective camera. */
export function worldPerPixel(distance: number, fovDegrees: number, canvasCssHeight: number): number {
  return (2 * distance * Math.tan((fovDegrees * Math.PI) / 360)) / canvasCssHeight;
}

/** Rotates `v` by the unit quaternion `q` ([x, y, z, w]). */
function rotateByQuat(
  v: readonly [number, number, number],
  q: readonly [number, number, number, number],
): [number, number, number] {
  const [qx, qy, qz, qw] = q;
  // t = 2 * cross(q.xyz, v); v + qw * t + cross(q.xyz, t).
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return [
    v[0] + qw * tx + (qy * tz - qz * ty),
    v[1] + qw * ty + (qz * tx - qx * tz),
    v[2] + qw * tz + (qx * ty - qy * tx),
  ];
}

/**
 * A grab dab's drag vector in the mesh's local frame: the screen delta
 * (`pixelDx`, `pixelDy`, y down) at `wpp` world units per pixel, along the
 * camera's `camRight`/`camUp`, carried by `meshQuatConj` (the conjugate of
 * the mesh's world quaternion).
 */
export function grabLocalDelta(
  pixelDx: number,
  pixelDy: number,
  wpp: number,
  camRight: readonly [number, number, number],
  camUp: readonly [number, number, number],
  meshQuatConj: readonly [number, number, number, number],
): [number, number, number] {
  const sx = pixelDx * wpp;
  const sy = -pixelDy * wpp;
  const world: [number, number, number] = [
    camRight[0] * sx + camUp[0] * sy,
    camRight[1] * sx + camUp[1] * sy,
    camRight[2] * sx + camUp[2] * sy,
  ];
  return rotateByQuat(world, meshQuatConj);
}

/**
 * A grab stroke's locked anchor: dabs land here however the pointer's
 * raycast hits swim (off the lump, onto another mesh), so a drag never
 * re-grabs elsewhere mid-stroke. It advances by each consumed drag delta,
 * so the grasped lump stays inside the brush ball.
 */
export class GrabAnchor {
  #center: [number, number, number];

  constructor(start: readonly [number, number, number]) {
    this.#center = [start[0], start[1], start[2]];
  }

  /** The next dab's center (a copy). */
  get center(): [number, number, number] {
    return [...this.#center];
  }

  /** Advances the anchor by a consumed drag delta. */
  advance(delta: readonly [number, number, number]): void {
    this.#center[0] += delta[0];
    this.#center[1] += delta[1];
    this.#center[2] += delta[2];
  }
}

/**
 * The mean live-edge length over flat positions and a triangle index: each
 * undirected edge once, zero-length (dead-slot) and non-finite edges
 * skipped. `null` when no edge survives. A sculpt session anchors its
 * remeshes to this (taken when the brush arms), so repeated strokes converge
 * back to the armed density instead of ratcheting ever finer.
 */
export function meanEdgeLength(positions: ArrayLike<number>, indices: ArrayLike<number>): number | null {
  const seen = new Set<number>();
  let sum = 0;
  let count = 0;
  const edge = (a: number, b: number) => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = lo * 4294967296 + hi;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    const dx = positions[3 * a] - positions[3 * b];
    const dy = positions[3 * a + 1] - positions[3 * b + 1];
    const dz = positions[3 * a + 2] - positions[3 * b + 2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(len > 0) || !Number.isFinite(len)) {
      return;
    }
    sum += len;
    count++;
  };
  for (let t = 0; t + 2 < indices.length; t += 3) {
    edge(indices[t], indices[t + 1]);
    edge(indices[t + 1], indices[t + 2]);
    edge(indices[t + 2], indices[t]);
  }
  return count === 0 ? null : sum / count;
}

/** Interpolates a raycast hit over flat local positions by its barycentric weights. */
export function interpolateHit(
  positions: ArrayLike<number>,
  corners: ArrayLike<number>,
  barycentric: readonly [number, number, number],
): [number, number, number] {
  const [u, v, w] = barycentric;
  const at = (s: number): readonly [number, number, number] => [
    positions[3 * s],
    positions[3 * s + 1],
    positions[3 * s + 2],
  ];
  const [a, b, c] = [at(corners[0]), at(corners[1]), at(corners[2])];
  return [u * a[0] + v * b[0] + w * c[0], u * a[1] + v * b[1] + w * c[1], u * a[2] + v * b[2] + w * c[2]];
}

/** The scene's step nesting, where it has one (the sync scene; the Worker applies one step per dab). */
export interface StrokeSteps {
  beginStep?: () => void;
  endStep?: () => void;
  cancelStep?: () => void;
}

/**
 * One press-and-hold stroke: dabs through `apply`, coalescing to one undo
 * step where the scene nests steps. Lazy: no step opens until the first dab
 * lands, and a stroke whose dabs all failed cancels its step instead.
 */
export class SculptStroke {
  readonly #brush: () => SculptBrush;
  readonly #steps: StrokeSteps;
  readonly #apply: (op: RepairOp) => Promise<unknown>;
  #last: [number, number, number] | null = null;
  #min: [number, number, number] | null = null;
  #max: [number, number, number] | null = null;
  #running: Promise<void> | null = null;
  #begun = false;
  #landed = false;
  #ended = false;
  #dabs = 0;
  #close!: () => void;
  /** Resolves once the stroke is ended and its step closed. */
  readonly done: Promise<void>;

  /**
   * `brush` is read live, per dab: pass a getter and the brush inputs stay
   * adjustable mid-stroke (a plain brush is read as-is).
   */
  constructor(brush: SculptBrush | (() => SculptBrush), steps: StrokeSteps, apply: (op: RepairOp) => Promise<unknown>) {
    this.#brush = typeof brush === "function" ? brush : () => brush;
    this.#steps = steps;
    this.#apply = apply;
    this.done = new Promise<void>((resolve) => {
      this.#close = resolve;
    });
  }

  /** Dabs attempted so far. */
  get dabs(): number {
    return this.#dabs;
  }

  /**
   * Applies a dab at `center` (local coordinates): `skipped` when it is too
   * close to the last (unless `force`, the hold-still buildup), `busy` while
   * one runs, `ended` once the stroke closed. A dab within reach creeps one
   * spacing toward the pointer (no dotted trails on fast moves); past
   * `CREEP_SPACINGS` spacings it jumps straight there.
   */
  dab(center: readonly [number, number, number], force = false): "stamped" | "skipped" | "busy" | "ended" {
    if (this.#ended) {
      return "ended";
    }
    if (this.#running !== null) {
      return "busy";
    }
    const brush = this.#brush();
    const spacing = dabSpacing(brush.radius);
    let at: [number, number, number] = [center[0], center[1], center[2]];
    if (!force && this.#last !== null) {
      const dx = center[0] - this.#last[0];
      const dy = center[1] - this.#last[1];
      const dz = center[2] - this.#last[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < spacing) {
        return "skipped";
      }
      if (dist <= CREEP_SPACINGS * spacing) {
        const k = spacing / dist;
        at = [this.#last[0] + dx * k, this.#last[1] + dy * k, this.#last[2] + dz * k];
      }
    }
    if (!this.#begun) {
      this.#begun = true;
      this.#steps.beginStep?.();
    }
    this.#dabs++;
    this.#last = at;
    if (this.#min === null || this.#max === null) {
      this.#min = [...at];
      this.#max = [...at];
    } else {
      for (let k = 0; k < 3; k++) {
        this.#min[k] = Math.min(this.#min[k], at[k]);
        this.#max[k] = Math.max(this.#max[k], at[k]);
      }
    }
    let settled: Promise<void>;
    try {
      settled = Promise.resolve(this.#apply(dabOp(brush, at))).then(
        () => {
          this.#landed = true;
        },
        () => {
          // A dab rejected (the mesh was removed meanwhile): the stroke
          // keeps its step, and cancels it when nothing landed.
        },
      );
    } catch {
      // The applier threw outright: a failed dab, like a rejection.
      settled = Promise.resolve();
    }
    this.#running = settled.finally(() => {
      this.#running = null;
      if (this.#ended) {
        this.#finish();
      }
    });
    return "stamped";
  }

  /**
   * The stamped dabs' bounding sphere, grown by twice the brush radius: the
   * region a stroke-end remesh should even out. The extra margin covers
   * hold-buildup growth past the dab centers (a spike outgrows its ball).
   * `null` before the first dab.
   */
  bounds(): { center: [number, number, number]; radius: number } | null {
    if (this.#min === null || this.#max === null) {
      return null;
    }
    const center: [number, number, number] = [
      (this.#min[0] + this.#max[0]) / 2,
      (this.#min[1] + this.#max[1]) / 2,
      (this.#min[2] + this.#max[2]) / 2,
    ];
    const dx = this.#max[0] - this.#min[0];
    const dy = this.#max[1] - this.#min[1];
    const dz = this.#max[2] - this.#min[2];
    const grown = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2 + 2 * this.#brush().radius;
    return { center, radius: grown };
  }

  /** Closes the stroke: its step ends (or cancels, when nothing landed), once the running dab settles. */
  end(): void {
    if (this.#ended) {
      return;
    }
    this.#ended = true;
    if (this.#running === null) {
      this.#finish();
    }
  }

  #finish(): void {
    if (this.#begun) {
      if (this.#landed) {
        this.#steps.endStep?.();
      } else {
        this.#steps.cancelStep?.();
      }
    }
    this.#close();
  }
}

// ---- Sculpt mask (#130) ----
//
// A painted protected region the deform brushes skip: the mask stores the
// painted vertex slots, and every dab whose ball touches one is dropped
// before it reaches the scene.
//
// The storage contract answers #130's question against #106's selection
// contract. A selection snapshot pins the mesh's content stamp, so the
// engine throws `STALE_SELECTION` after any edit — including the stroke's
// own position-only dabs, which would kill a mask mid-stroke. The showcase
// mask therefore pins the vertex ids instead of the stamp: ids are stable
// across position-only edits (inflate/relax/grab move vertices without
// touching slots), so the mask survives the strokes it protects. It fails
// loudly — `checkMask` throws `STALE_SELECTION`, the same code — as soon as
// the ids stop naming live vertices: a masked slot out of range, or no
// longer used by a live triangle (a collapse, compact, weld or isolate took
// it). Surviving cases re-verify cheaply: splits add slots but keep the old
// vertices alive, so a remeshed stroke's mask carries over (the new vertices
// land unmasked); undo/redo restores the buffers, and the mask with them
// when its ids are live again. The documented hole: a merge or compact can
// remap ids while keeping them live, silently re-pointing the mask —
// reselect after structural repairs, as with #106 selections.

/**
 * A painted protected region on one mesh: the masked vertex slots,
 * ascending, and the vertex slot count when painted (a topology edit that
 * grows or compacts the slots re-verifies against this).
 */
export interface SculptMask {
  /** The mesh the slots belong to. */
  mesh: MeshHandle;
  /** The masked vertex slots, ascending. */
  vertices: Uint32Array;
  /** `positions().length / 3` when painted; see the contract above. */
  slots: number;
}

/**
 * Whether `id` names a slot a triangle still uses: in range and on a live
 * (non-removed) triangle. Removed slots read `0, 0, 0`, so triples of three
 * zeros are skipped — mirroring the engine's `buildTopology`, where a
 * genuine all-zero triangle is degenerate (zero area) and losing it is
 * harmless.
 */
function maskIdLive(indices: ArrayLike<number>, id: number, slots: number): boolean {
  if (!Number.isInteger(id) || id < 0 || id >= slots) {
    return false;
  }
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a === id || b === id || c === id) {
      return true;
    }
  }
  return false;
}

/**
 * Whether the mask still names live vertices of `mesh` over its current
 * buffers: every masked slot in range and used by a live triangle. The
 * per-dab gate (position moves never change this); the stroke start and
 * paint/invert paths call `checkMask` to fail loudly instead.
 */
export function maskUsable(
  mask: Pick<SculptMask, "vertices" | "slots">,
  indices: ArrayLike<number>,
  slots: number,
): boolean {
  // A slots mismatch (splits grew the mesh, a compact shrank it) never
  // fails on its own: the ids may still name the same vertices (splits only
  // add), so liveness below decides. An empty mask protects nothing.
  for (let k = 0; k < mask.vertices.length; k++) {
    if (!maskIdLive(indices, mask.vertices[k], slots)) {
      return false;
    }
  }
  return true;
}

/**
 * What {@link checkMask} throws: the showcase twin of the engine's
 * `FastMeshError`, carrying the same machine-readable `code`
 * (`"INVALID_ARGUMENT"` for the wrong mesh, `"STALE_SELECTION"` once a
 * topology edit takes a masked vertex). A local class — not the engine's —
 * so this pure module stays importable without the wasm build (like every
 * other headless-tested showcase module); catch sites treat both by `code`.
 */
export class MaskError extends Error {
  /** What went wrong; the same codes the engine reports stale selections with. */
  readonly code: FastMeshErrorCode;

  constructor(code: FastMeshErrorCode, message: string) {
    super(message);
    this.name = "MaskError";
    this.code = code;
  }
}

/**
 * Fails loudly when `mask` is not for `mesh` or no longer usable over its
 * current buffers: `INVALID_ARGUMENT` for the wrong mesh (like the
 * engine's `checkSelection`), `STALE_SELECTION` once a topology edit took a
 * masked vertex (repaint it). Callers clear the stored mask on the throw.
 */
export function checkMask(
  mask: SculptMask,
  mesh: MeshHandle,
  indices: ArrayLike<number>,
  slots: number,
): void {
  if (mask === null || typeof mask !== "object" || mask.mesh !== mesh) {
    throw new MaskError("INVALID_ARGUMENT", "the mask is not for this mesh");
  }
  if (!maskUsable(mask, indices, slots)) {
    throw new MaskError("STALE_SELECTION", "the mask predates the mesh's latest topology edit; repaint it");
  }
}

/**
 * The sphere-brush paint: every live vertex within `radius` of `center`
 * (squared distances, `<=`, like the engine's `brushPick`), ascending. Dead
 * slots and out-of-range ids never paint.
 */
export function paintMaskIds(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  slots: number,
  center: readonly [number, number, number],
  radius: number,
): Uint32Array {
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a >= 0 && a < slots) {
      used[a] = 1;
    }
    if (b >= 0 && b < slots) {
      used[b] = 1;
    }
    if (c >= 0 && c < slots) {
      used[c] = 1;
    }
  }
  const r2 = radius * radius;
  const out: number[] = [];
  for (let v = 0; v < slots; v++) {
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
  return Uint32Array.from(out);
}

/** The sorted union of two ascending id lists (paint accumulates through this). */
export function unionMaskIds(a: ArrayLike<number>, b: ArrayLike<number>): Uint32Array {
  const out: number[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const x = a[i];
    const y = b[j];
    if (x === y) {
      out.push(x);
      i++;
      j++;
    } else if (x < y) {
      out.push(x);
      i++;
    } else {
      out.push(y);
      j++;
    }
  }
  while (i < a.length) {
    out.push(a[i++]);
  }
  while (j < b.length) {
    out.push(b[j++]);
  }
  return Uint32Array.from(out);
}

/**
 * Inverts a mask over the mesh's current buffers: every live vertex outside
 * it, ascending. The engine's `selectInvert` needs a live selection; this
 * takes the same universe without one, so invert works freshly painted.
 */
export function invertMaskIds(
  masked: ArrayLike<number>,
  indices: ArrayLike<number>,
  slots: number,
): Uint32Array {
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a >= 0 && a < slots) {
      used[a] = 1;
    }
    if (b >= 0 && b < slots) {
      used[b] = 1;
    }
    if (c >= 0 && c < slots) {
      used[c] = 1;
    }
  }
  const inSet = new Set<number>();
  for (let k = 0; k < masked.length; k++) {
    inSet.add(masked[k]);
  }
  const out: number[] = [];
  for (let v = 0; v < slots; v++) {
    if (used[v] === 1 && !inSet.has(v)) {
      out.push(v);
    }
  }
  return Uint32Array.from(out);
}

/**
 * Whether a dab at `center` is protected: any masked slot within its ball
 * (`<= radius`, the paint's own measure). The dab is then dropped before it
 * reaches the scene — the brushes skip the region instead of scaling it, so
 * a fully masked lump never drifts under buildup.
 */
export function maskBlocksDab(
  masked: ArrayLike<number>,
  positions: ArrayLike<number>,
  center: readonly [number, number, number],
  radius: number,
): boolean {
  const r2 = radius * radius;
  for (let k = 0; k < masked.length; k++) {
    const v = masked[k];
    const dx = positions[v * 3] - center[0];
    const dy = positions[v * 3 + 1] - center[1];
    const dz = positions[v * 3 + 2] - center[2];
    if (dx * dx + dy * dy + dz * dz <= r2) {
      return true;
    }
  }
  return false;
}
