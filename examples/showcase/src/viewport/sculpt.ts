// The sculpt stroke: what one dab applies, and one press-and-hold stroke
// over the meshes (the viewport's ring and pointer loop drive it).
import type { MeshHandle, RepairOp } from "fast-mesh-wasm";

/** A sculpt brush armed on a mesh (app-state `sculpt`). */
export interface SculptBrush {
  mesh: MeshHandle;
  op: "inflate" | "smooth";
  radius: number;
  strength: number;
  falloff: "linear" | "smooth";
  cycles: number;
}

/** Dab spacing: a quarter of the radius (never zero, so dabs terminate). */
export function dabSpacing(radius: number): number {
  return Math.max(radius * 0.25, 1e-6);
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
  return brush.op === "smooth" ? { op: "smooth", ...shared, cycles: brush.cycles } : { op: "inflate", ...shared };
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
   * one runs, `ended` once the stroke closed.
   */
  dab(center: readonly [number, number, number], force = false): "stamped" | "skipped" | "busy" | "ended" {
    if (this.#ended) {
      return "ended";
    }
    if (this.#running !== null) {
      return "busy";
    }
    const brush = this.#brush();
    if (!force && this.#last !== null && !farEnough(this.#last, center, dabSpacing(brush.radius))) {
      return "skipped";
    }
    if (!this.#begun) {
      this.#begun = true;
      this.#steps.beginStep?.();
    }
    this.#dabs++;
    this.#last = [center[0], center[1], center[2]];
    let settled: Promise<void>;
    try {
      settled = Promise.resolve(this.#apply(dabOp(brush, center))).then(
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
