// The sculpt stroke: what one dab applies, and one press-and-hold stroke
// over the meshes (the viewport's ring and pointer loop drive it).
import type { MeshHandle, RepairOp } from "fast-mesh-wasm";

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

/** Past this many spacings a dab jumps to the pointer instead of creeping (see `SculptStroke.dab`). */
export const CREEP_SPACINGS = 4;

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
