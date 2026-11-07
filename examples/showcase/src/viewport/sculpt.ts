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
 * Inflate's is the height of one dab over the brush radius (see `dabOp`), so
 * a stroke looks the same at any model size. It stays small: dabs overlap
 * heavily along a sweep, so a tenth of the radius per dab already builds a
 * visible bump without folding.
 */
export const DEFAULT_STRENGTHS: Record<SculptTool, number> = { inflate: 0.1, deflate: 0.1, smooth: 0.5, grab: 1 };

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

/**
 * Past this many brush radii between two pointer samples the stroke jumps
 * instead of laying dabs along the way: that is the pointer leaving one part
 * of the surface for another, not a fast move (see `SculptStroke.dab`).
 */
export const TELEPORT_RADII = 8;

/**
 * Past this many triangles the optional "even out" remesh stands down: it
 * runs over the whole mesh (measured ~2.6 s floor on 300 k, minutes for wide
 * regions), which would stall the worker and the dabs behind it. Dabs cost by
 * the brush region, and the stretch splits by the stretched edges: neither has
 * such a limit.
 */
export const REMESH_TRI_LIMIT = 100_000;

/** Whether `triangles` stands the optional stroke-end remesh down. */
export function remeshTooBig(triangles: number): boolean {
  return triangles > REMESH_TRI_LIMIT;
}

/**
 * Stretch splits while sculpting (#129, #175): the per-frame budget.
 *
 * At most one live pass per stroke region per `LIVE_REMESH_MIN_MS`, and only
 * once `LIVE_REMESH_MIN_DABS` dabs landed since the last pass, never two in
 * flight. A pass splits only the edges the stroke stretched past their limit
 * (`splitStretched`): what the stroke did not stretch keeps its vertices and
 * triangles, and the vertex ids a mask stores stay valid. Each stroke asks for
 * rest sizes before its first dab (`captureRestSizes`, `onlyIfMissing`), so
 * repeated strokes measure against the same shape instead of ratcheting, until
 * an edit outdates it and the engine drops the sizes.
 */
export const LIVE_REMESH_MIN_DABS = 4;
/** Minimum milliseconds between two live remesh passes of one stroke. */
export const LIVE_REMESH_MIN_MS = 150;
/** Most interior splits one pass makes; a later pass takes the rest, longest first. */
export const LIVE_MAX_SPLITS = 50_000;

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
 * nesting — one step per batch of dabs — so the passes land as their own steps
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

/**
 * The repair op one dab applies at `center` (local coordinates): the
 * displacement step of a painted stroke. Inflate's strength is relative to
 * the brush radius; the engine takes it in mesh units.
 */
export function dabOp(brush: SculptBrush, center: readonly [number, number, number]): RepairOp {
  const shared = {
    center: [center[0], center[1], center[2]] as [number, number, number],
    radius: brush.radius,
    falloff: brush.falloff,
    strength: brush.op === "inflate" ? brush.strength * brush.radius : brush.strength,
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

/** The falloff weight at normalized distance `t`, as the engine's brushes compute it. */
export function falloffWeight(falloff: "linear" | "smooth", t: number): number {
  if (!(t < 1)) {
    return 0;
  }
  const s = 1 - t;
  return falloff === "linear" ? s : s * s * (3 - 2 * s);
}

/**
 * What a grab stroke holds on to: the vertices under the brush at the press,
 * where they rested and their weights. Each move is written from these and the
 * total drag, so the shape depends on where the pointer is and not on how many
 * events it took to get there. The displacement step of a grab stroke:
 * `positionsAt` is all the stroke asks of it.
 */
export class GrabRegion {
  /** The grabbed vertex slots, ascending. */
  ids: Uint32Array;
  #weights: Float32Array;
  #rest: Float32Array;
  readonly #center: readonly [number, number, number];
  readonly #radius: number;
  readonly #falloff: "linear" | "smooth";
  readonly #strength: number;
  readonly #origin: readonly [number, number, number];
  /** The drag of the last `positionsAt`: what the mesh shows, once it landed. */
  #drag: readonly [number, number, number] = [0, 0, 0];

  constructor(
    positions: ArrayLike<number>,
    indices: ArrayLike<number>,
    center: readonly [number, number, number],
    radius: number,
    falloff: "linear" | "smooth",
    strength: number,
    origin: readonly [number, number, number] = [0, 0, 0],
  ) {
    this.#origin = [origin[0], origin[1], origin[2]];
    this.#center = [center[0], center[1], center[2]];
    this.#radius = radius;
    this.#falloff = falloff;
    this.#strength = strength;
    this.ids = new Uint32Array();
    this.#weights = new Float32Array();
    this.#rest = new Float32Array();
    this.rebase(positions, indices);
  }

  /** The weight of a vertex resting at `x, y, z`. */
  #weight(x: number, y: number, z: number): number {
    const c = this.#center;
    const t = Math.hypot(x - c[0], y - c[1], z - c[2]) / this.#radius;
    return this.#strength * falloffWeight(this.#falloff, t);
  }

  /**
   * Takes the region again over `positions`, with every move sent so far
   * landed: for a pass that added vertices or moved some mid-stroke. A vertex
   * that is still where this region put it keeps its rest. A vertex from
   * `splits` (`new, a, b` triples, as `splitStretched` reports them) gets the
   * mean weight of its edge: its position alone cannot say, where the drag has
   * folded the sheet over itself. Any other gets the weight of the rest position
   * the drag would have carried here (see `#weightAt`). Each then takes exactly
   * the rest that lands on it now, so that nothing jumps.
   */
  rebase(positions: ArrayLike<number>, indices: ArrayLike<number>, splits?: ArrayLike<number>): void {
    const drag = this.#drag;
    const slots = positions.length / 3;
    const reach = this.#radius + Math.abs(this.#strength) * Math.hypot(drag[0], drag[1], drag[2]);
    const known = new Map<number, number>();
    this.ids.forEach((id, k) => known.set(id, k));
    const split = new Map<number, number>();
    const weightOf = (id: number): number => {
      const k = known.get(id);
      return split.get(id) ?? (k === undefined ? 0 : this.#weights[k]);
    };
    for (let k = 0; splits !== undefined && k + 2 < splits.length; k += 3) {
      split.set(splits[k], (weightOf(splits[k + 1]) + weightOf(splits[k + 2])) / 2);
    }
    const inReach = paintMaskIds(positions, indices, slots, this.#center, reach);
    const candidates = Uint32Array.from(new Set([...inReach, ...split.keys()])).sort();
    const ids: number[] = [];
    const weights: number[] = [];
    const rest: number[] = [];
    const tolerance = 1e-5 * this.#radius;
    for (let k = 0; k < candidates.length; k++) {
      const id = candidates[k];
      const [px, py, pz] = [positions[3 * id], positions[3 * id + 1], positions[3 * id + 2]];
      const old = known.get(id);
      if (old !== undefined) {
        const w = this.#weights[old];
        const [rx, ry, rz] = [this.#rest[3 * old], this.#rest[3 * old + 1], this.#rest[3 * old + 2]];
        if (Math.hypot(rx + w * drag[0] - px, ry + w * drag[1] - py, rz + w * drag[2] - pz) <= tolerance) {
          ids.push(id);
          weights.push(w);
          rest.push(rx, ry, rz);
          continue;
        }
      }
      const w = split.get(id) ?? this.#weightAt(px, py, pz, drag);
      if (w !== 0) {
        ids.push(id);
        weights.push(w);
        rest.push(px - w * drag[0], py - w * drag[1], pz - w * drag[2]);
      }
    }
    this.ids = Uint32Array.from(ids);
    this.#weights = Float32Array.from(weights);
    this.#rest = Float32Array.from(rest);
  }

  /**
   * The weight `w` of a vertex at `x, y, z` under `drag`: one with
   * `w = weight(p - w * drag)`. Fixed-point steps diverge once the drag is
   * longer than the falloff, which leaves the vertices on a long lump's sides
   * behind, so this solves for the share `u = w / strength` directly.
   *
   * The falloff is largest at `top`, where the line through `p` along the drag
   * passes closest to the center, and falls beyond it. So above `top`, the
   * excess `falloff - u` only falls and has at most one root, found by
   * bisection. That largest root is the one on the surface: a vertex that was
   * dragged here rested where the brush weighed it most, and the smaller
   * roots, down to 0 for a point now outside the ball, are off the surface.
   * With no root above `top`, the vertex lies past the lump's rim, and the
   * largest root below it is found by scanning.
   */
  #weightAt(x: number, y: number, z: number, drag: readonly [number, number, number]): number {
    const s = this.#strength;
    const [ax, ay, az] = [s * drag[0], s * drag[1], s * drag[2]];
    const c = this.#center;
    const length2 = ax * ax + ay * ay + az * az;
    if (length2 === 0) {
      return this.#weight(x, y, z);
    }
    const at = (u: number): number => this.#weight(x - u * ax, y - u * ay, z - u * az);
    const excess = (u: number): number => at(u) / s - u;
    const closest = ((x - c[0]) * ax + (y - c[1]) * ay + (z - c[2]) * az) / length2;
    const top = Math.min(1, Math.max(0, closest));
    let low = top;
    let high = 1;
    if (excess(top) < 0) {
      const STEPS = 16;
      high = top;
      low = 0;
      for (let k = STEPS - 1; k >= 0; k--) {
        const u = (top * k) / STEPS;
        if (excess(u) >= 0) {
          low = u;
          break;
        }
        high = u;
      }
    }
    for (let k = 0; k < 30; k++) {
      const mid = (low + high) / 2;
      if (excess(mid) >= 0) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return at(low);
  }

  /**
   * Where the grabbed vertices belong at the total `drag`, three numbers per id,
   * as `setPositions` takes them: the region works in the mesh's stored
   * positions, which are relative to its `origin`.
   */
  positionsAt(drag: readonly [number, number, number]): Float64Array {
    this.#drag = [drag[0], drag[1], drag[2]];
    const out = new Float64Array(this.#rest.length);
    for (let k = 0; k < this.ids.length; k++) {
      const w = this.#weights[k];
      out[3 * k] = this.#rest[3 * k] + w * drag[0] + this.#origin[0];
      out[3 * k + 1] = this.#rest[3 * k + 1] + w * drag[1] + this.#origin[1];
      out[3 * k + 2] = this.#rest[3 * k + 2] + w * drag[2] + this.#origin[2];
    }
    return out;
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
 *
 * The dabs lie every spacing along the pointer's path, whatever its event rate
 * and however long a dab takes: those that come due while one is in flight
 * wait, and go together when it lands, so a slow scene lags but leaves no gaps.
 */
export class SculptStroke {
  readonly #brush: () => SculptBrush;
  readonly #steps: StrokeSteps;
  readonly #apply: (ops: readonly RepairOp[]) => Promise<unknown>;
  readonly #blocked: (at: readonly [number, number, number], radius: number) => boolean;
  #last: [number, number, number] | null = null;
  #min: [number, number, number] | null = null;
  #max: [number, number, number] | null = null;
  #queue: RepairOp[] = [];
  #latest: (() => Promise<unknown>) | null = null;
  #inFlight = 0;
  /** A pass is out: nothing else goes until it settles. */
  #passing = false;
  /** What came due behind a pass, in order; a `true` entry is a pass itself. */
  #held: { run: () => Promise<unknown>; pass: boolean }[] = [];
  #begun = false;
  #landed = false;
  #ended = false;
  #closed = false;
  #dabs = 0;
  #close!: () => void;
  /** Resolves once the stroke is ended and its step closed. */
  readonly done: Promise<void>;

  /**
   * `brush` is read live, per dab: pass a getter and the brush inputs stay
   * adjustable mid-stroke (a plain brush is read as-is). `apply` gets the dabs
   * that came due, in order, one or more at a time. A dab `blocked` says no
   * to (the mask) is left out, and the path goes on past it.
   */
  constructor(
    brush: SculptBrush | (() => SculptBrush),
    steps: StrokeSteps,
    apply: (ops: readonly RepairOp[]) => Promise<unknown>,
    blocked: (at: readonly [number, number, number], radius: number) => boolean = () => false,
  ) {
    this.#brush = typeof brush === "function" ? brush : () => brush;
    this.#steps = steps;
    this.#apply = apply;
    this.#blocked = blocked;
    this.done = new Promise<void>((resolve) => {
      this.#close = resolve;
    });
  }

  /** Dabs attempted so far. */
  get dabs(): number {
    return this.#dabs;
  }

  /**
   * Moves the stroke to `center` (local coordinates), laying a dab every
   * spacing from the last one: `skipped` when it is under a spacing away,
   * `ended` once the stroke closed. With `force` (the hold-still buildup,
   * which goes by time and not by distance) one dab lands right there, or
   * `busy` while one is in flight. Past `TELEPORT_RADII` radii it starts over
   * at `center`, as after `lift`.
   */
  dab(center: readonly [number, number, number], force = false): "stamped" | "skipped" | "busy" | "ended" {
    if (this.#ended) {
      return "ended";
    }
    const brush = this.#brush();
    const at: [number, number, number] = [center[0], center[1], center[2]];
    if (force || this.#last === null) {
      if (force && this.#inFlight > 0) {
        return "busy";
      }
      this.#stamp(brush, at);
      this.#sendIfIdle();
      return "stamped";
    }
    const spacing = dabSpacing(brush.radius);
    const dx = at[0] - this.#last[0];
    const dy = at[1] - this.#last[1];
    const dz = at[2] - this.#last[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < spacing) {
      return "skipped";
    }
    if (dist > TELEPORT_RADII * brush.radius) {
      this.#stamp(brush, at);
      this.#sendIfIdle();
      return "stamped";
    }
    const [x, y, z] = this.#last;
    const steps = Math.floor(dist / spacing);
    for (let k = 1; k <= steps; k++) {
      const u = (k * spacing) / dist;
      this.#stamp(brush, [x + dx * u, y + dy * u, z + dz * u]);
    }
    this.#sendIfIdle();
    return "stamped";
  }

  /** The pointer left the surface: the next `dab` starts over where it lands. */
  lift(): void {
    this.#last = null;
  }

  /**
   * Writes a new displacement of a stroke that holds its own state (a grab):
   * `edit` runs once nothing is in flight, and a newer `move` replaces one that
   * has not started. `at` is where the brush is now, for `bounds`.
   */
  move(edit: () => Promise<unknown>, at: readonly [number, number, number]): "stamped" | "ended" {
    if (this.#ended) {
      return "ended";
    }
    this.#open();
    this.#dabs++;
    this.#cover(at);
    this.#latest = edit;
    this.#sendIfIdle();
    return "stamped";
  }

  #sendIfIdle(): void {
    if (this.#inFlight === 0) {
      this.flush();
    }
  }

  #open(): void {
    if (!this.#begun) {
      this.#begun = true;
      this.#steps.beginStep?.();
    }
  }

  #cover(at: readonly [number, number, number]): void {
    if (this.#min === null || this.#max === null) {
      this.#min = [at[0], at[1], at[2]];
      this.#max = [at[0], at[1], at[2]];
    } else {
      for (let k = 0; k < 3; k++) {
        this.#min[k] = Math.min(this.#min[k], at[k]);
        this.#max[k] = Math.max(this.#max[k], at[k]);
      }
    }
  }

  #stamp(brush: SculptBrush, at: [number, number, number]): void {
    this.#last = at;
    if (this.#blocked(at, brush.radius)) {
      return;
    }
    this.#open();
    this.#dabs++;
    this.#cover(at);
    this.#queue.push(dabOp(brush, at));
  }

  /**
   * Runs `work` in order with the stroke's own edits, after what is due and
   * before what comes due next (a remesh): the stroke does not close, and sends
   * nothing more, until it settles. A failure is the caller's to swallow.
   */
  pass(work: () => Promise<unknown>): Promise<void> {
    return new Promise<void>((resolve) => {
      const run = () => {
        const running = work();
        void running.then(
          () => resolve(),
          () => resolve(),
        );
        return running;
      };
      if (this.#passing) {
        this.#hold();
        this.#held.push({ run, pass: true });
        return;
      }
      this.flush();
      this.#startPass(run);
    });
  }

  #startPass(run: () => Promise<unknown>): void {
    this.#passing = true;
    this.#send(run, true);
  }

  /** Moves what is due behind the pass that is out. */
  #hold(): void {
    if (this.#queue.length > 0) {
      const ops = this.#queue;
      this.#queue = [];
      this.#held.push({ run: () => this.#apply(ops), pass: false });
    }
    if (this.#latest !== null) {
      this.#held.push({ run: this.#latest, pass: false });
      this.#latest = null;
    }
  }

  /** After a pass: what waited behind it, up to the next pass. */
  #release(): void {
    while (!this.#passing && this.#held.length > 0) {
      const next = this.#held.shift();
      if (next?.pass === true) {
        this.#startPass(next.run);
      } else if (next !== undefined) {
        this.#send(next.run);
      }
    }
  }

  /**
   * Sends what is due now, even with a dab still in flight: for a caller about
   * to send something else that has to land after it (a remesh). On its own
   * the stroke sends the next batch when the last one lands. A pass that is out
   * holds it: what is due goes when the pass settles.
   */
  flush(): void {
    if (this.#passing) {
      return;
    }
    if (this.#queue.length > 0) {
      const ops = this.#queue;
      this.#queue = [];
      this.#send(() => this.#apply(ops));
    }
    if (this.#latest !== null) {
      const edit = this.#latest;
      this.#latest = null;
      this.#send(edit);
    }
  }

  #send(run: () => Promise<unknown>, pass = false): void {
    this.#inFlight++;
    let settled: Promise<void>;
    try {
      settled = Promise.resolve(run()).then(
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
    void settled.finally(() => {
      this.#inFlight--;
      if (pass) {
        this.#passing = false;
        this.#release();
      }
      if (this.#inFlight === 0) {
        this.flush();
      }
      if (this.#inFlight === 0 && this.#ended && !this.#passing && this.#held.length === 0) {
        this.#finish();
      }
    });
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

  /** Closes the stroke: its step ends (or cancels, when nothing landed), once what it sent settles. */
  end(): void {
    if (this.#ended) {
      return;
    }
    this.#ended = true;
    this.flush();
    if (this.#inFlight === 0) {
      this.#finish();
    }
  }

  #finish(): void {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
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
// A painted protected region the deform brushes leave in place: the mask
// stores the painted vertex slots, and every dab carries the ones it can
// reach to the engine, which holds them byte-identical while the free
// vertices move (see `viewport/mask-protect.ts`).
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
// vertices alive, so a split stroke's mask carries over (the new vertices
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
 * Whether the mask still names live vertices of `mesh` over its current
 * buffers: every masked slot in range and used by a live triangle. Removed
 * slots read `0, 0, 0`, so triples of three zeros are skipped — mirroring the
 * engine's `buildTopology`, where a genuine all-zero triangle is degenerate
 * (zero area) and losing it is harmless. One pass over the triangles, then one
 * over the mask: painting calls this per dab. The stroke start and
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
  if (mask.vertices.length === 0) {
    return true;
  }
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a < slots) {
      used[a] = 1;
    }
    if (b < slots) {
      used[b] = 1;
    }
    if (c < slots) {
      used[c] = 1;
    }
  }
  for (let k = 0; k < mask.vertices.length; k++) {
    const id = mask.vertices[k];
    if (!Number.isInteger(id) || id < 0 || id >= slots || used[id] === 0) {
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
