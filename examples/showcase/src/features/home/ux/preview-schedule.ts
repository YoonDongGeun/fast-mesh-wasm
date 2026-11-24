// When a mesh's picture in the mesh list is drawn: one picture at a time, a
// slice of a few milliseconds per idle callback, never under a press on the
// viewport. Edits only postpone it: a stroke pays for a timer per frame and
// the picture follows once the edits go quiet and the press has ended. A turn
// of the mesh redraws it; a move alone does not (the picture is framed).

/** How long edits must stay away before a picture is redrawn. */
export const PREVIEW_QUIET_MS = 400;
/** How long a new mesh waits, so its first frames in the viewport go first. */
export const PREVIEW_FIRST_MS = 150;
/** How often a held queue looks again. */
const HELD_RETRY_MS = 100;

/** A picture drawn in slices: each `step` works for about `budgetMs` and returns true once it is done. */
export interface PreviewSteps {
  step(budgetMs: number, now: () => number): boolean;
}

/** What the queue draws for one key. */
export interface PreviewJob<R extends PreviewSteps = PreviewSteps> {
  start(): R;
  done(render: R, stats: PreviewRun): void;
}

/** What one finished picture cost. */
export interface PreviewRun {
  /** From the first slice to the last, idle time between them included. */
  wallMs: number;
  /** Time on the main thread. */
  busyMs: number;
  slices: number;
  longestSliceMs: number;
}

/** The queue's clock, timers and idle callbacks (a fake in the tests). */
export interface PreviewHost {
  now(): number;
  setTimer(cb: () => void, ms: number): number;
  clearTimer(id: number): void;
  /** Calls back when the page is idle, with a slice's budget in milliseconds. */
  idle(cb: (budgetMs: number) => void): void;
  /** True while nothing may be drawn. */
  held(): boolean;
}

/**
 * The pointers pressed on the viewport. No picture is drawn under a press: on a
 * slow machine a stroke's edits can come further apart than the quiet time, and
 * a stroke or an orbit is when a frame matters most.
 */
const pressed = new Set<number>();
if (typeof window !== "undefined") {
  window.addEventListener(
    "pointerdown",
    (e) => {
      if (e.target instanceof Element && e.target.closest(".viewport canvas") !== null) {
        pressed.add(e.pointerId);
      }
    },
    { capture: true },
  );
  const release = (e: PointerEvent) => pressed.delete(e.pointerId);
  window.addEventListener("pointerup", release, { capture: true });
  window.addEventListener("pointercancel", release, { capture: true });
  window.addEventListener("blur", () => pressed.clear());
}

const browserHost: PreviewHost = {
  held: () => pressed.size > 0,
  now: () => performance.now(),
  setTimer: (cb, ms) => window.setTimeout(cb, ms),
  clearTimer: (id) => window.clearTimeout(id),
  idle: (cb) => {
    if (typeof requestIdleCallback === "function") {
      // The timeout keeps pictures coming while the viewport renders every frame.
      requestIdleCallback((d) => cb(Math.min(4, Math.max(2, d.timeRemaining()))), { timeout: 100 });
    } else {
      window.setTimeout(() => cb(3), 16); // Safari
    }
  },
};

interface ActiveRender<K, R extends PreviewSteps> {
  key: K;
  job: PreviewJob<R>;
  render: R;
  run: PreviewRun;
  started: number;
}

/**
 * One picture at a time, one slice per idle callback. {@link invalidate}
 * (re)arms a key: it starts once `quietMs` passed since its last invalidation,
 * and an invalidation drops that key's render in progress.
 */
export class PreviewQueue<K = unknown, R extends PreviewSteps = PreviewSteps> {
  readonly #host: PreviewHost;
  readonly #waiting = new Map<K, { job: PreviewJob<R>; due: number }>();
  #active: ActiveRender<K, R> | null = null;
  #timer: number | null = null;

  constructor(host: PreviewHost = browserHost) {
    this.#host = host;
  }

  invalidate(key: K, job: PreviewJob<R>, quietMs: number = PREVIEW_QUIET_MS): void {
    if (this.#active?.key === key) {
      this.#active = null;
    }
    this.#waiting.set(key, { job, due: this.#host.now() + quietMs });
    this.#pump();
  }

  cancel(key: K): void {
    this.#waiting.delete(key);
    if (this.#active?.key === key) {
      this.#active = null;
    }
    this.#pump();
  }

  #pump(): void {
    if (this.#active !== null) {
      return;
    }
    const now = this.#host.now();
    let soonest: K | undefined;
    let due = Infinity;
    for (const [key, w] of this.#waiting) {
      if (w.due < due) {
        soonest = key;
        due = w.due;
      }
    }
    if (this.#timer !== null) {
      this.#host.clearTimer(this.#timer);
      this.#timer = null;
    }
    if (due === Infinity) {
      return;
    }
    if (due > now) {
      this.#timer = this.#host.setTimer(() => {
        this.#timer = null;
        this.#pump();
      }, due - now);
      return;
    }
    if (this.#host.held()) {
      this.#timer = this.#host.setTimer(() => {
        this.#timer = null;
        this.#pump();
      }, HELD_RETRY_MS);
      return;
    }
    const { job } = this.#waiting.get(soonest as K) as { job: PreviewJob<R> };
    this.#waiting.delete(soonest as K);
    let render: R;
    try {
      render = job.start();
    } catch {
      this.#pump();
      return;
    }
    const active = { key: soonest as K, job, render, started: now, run: { wallMs: 0, busyMs: 0, slices: 0, longestSliceMs: 0 } };
    this.#active = active;
    this.#host.idle((budget) => this.#slice(active, budget));
  }

  #slice(active: ActiveRender<K, R>, budget: number): void {
    if (this.#active !== active) {
      return; // overtaken by an edit, or cancelled
    }
    if (this.#host.held()) {
      this.#host.setTimer(() => this.#slice(active, budget), HELD_RETRY_MS);
      return;
    }
    const t0 = this.#host.now();
    let finished: boolean;
    try {
      finished = active.render.step(budget, () => this.#host.now());
    } catch {
      // The mesh went away between its last change and this slice.
      this.#active = null;
      this.#pump();
      return;
    }
    const t1 = this.#host.now();
    const { run } = active;
    run.busyMs += t1 - t0;
    run.slices++;
    run.longestSliceMs = Math.max(run.longestSliceMs, t1 - t0);
    if (!finished) {
      this.#host.idle((b) => this.#slice(active, b));
      return;
    }
    run.wallMs = t1 - active.started;
    this.#active = null;
    active.job.done(active.render, run);
    this.#pump();
  }
}

/**
 * Whether two column-major placements differ in more than their move, which a
 * framed picture does not show. Rounding does not count: a move recomposes the
 * matrix from its rotation, so those numbers can change in their last digits.
 */
export function turnChanged(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  const axes = [0, 1, 2, 4, 5, 6, 8, 9, 10];
  const size = Math.max(...axes.map((i) => Math.abs(a[i])));
  return axes.some((i) => Math.abs(a[i] - b[i]) > 1e-6 * size);
}
