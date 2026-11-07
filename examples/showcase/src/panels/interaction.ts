// Keeps the heatmap live while the pointer changes a mesh: an engine
// interaction (`setFieldInteraction`) from the press of a gizmo drag or a
// sculpt stroke until the pointer stops, then the exact field. The engine's
// interaction tier does the rest; this only says when.
import type { Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect } from "react";
import { type AppState } from "../scene/app-state";
import { view } from "../viewport/view";

/**
 * Near vertices per field per colour update during an interaction: about a
 * third of a 60 Hz frame on the benchmark machine for the two fields of a pair
 * at 1M triangles (`scripts/showcase-heatmap-bench.mjs`). The cost follows this
 * budget, not the mesh size.
 */
export const INTERACTION_POINTS = 8192;

/** The pointer counts as stopped after this long without a move or a change of a mesh. */
export const SETTLE_MS = 200;

/** Whether an interaction runs, for what pauses during one (the readouts). */
export const interaction = {
  active: false,
  /** Told when it starts (`true`) and ends (`false`). */
  listeners: new Set<(active: boolean) => void>(),
};

/**
 * When an interaction runs: `press` (a gizmo grab or a press with the brush
 * armed) starts it, `release` ends it at once, and while pressed a `move` or a
 * `change` keeps it going, restarting it after a pause. `SETTLE_MS` without
 * either ends it: the pointer stopped, so the colours refine to the exact field
 * even with the button held.
 */
export class InteractionTimer {
  #active = false;
  #pressed = false;
  #timer: ReturnType<typeof setTimeout> | undefined;
  readonly #set: (active: boolean) => void;

  constructor(set: (active: boolean) => void) {
    this.#set = set;
  }

  press(): void {
    this.#pressed = true;
    this.#touch();
  }

  /** The pointer moved, or a mesh changed. Ignored unless pressed. */
  touch(): void {
    if (this.#pressed) {
      this.#touch();
    }
  }

  release(): void {
    this.#pressed = false;
    this.#stop();
  }

  /** Whether the pointer is down on something that changes a mesh. */
  get pressed(): boolean {
    return this.#pressed;
  }

  #touch(): void {
    if (!this.#active) {
      this.#active = true;
      this.#set(true);
    }
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => this.#stop(), SETTLE_MS);
  }

  #stop(): void {
    clearTimeout(this.#timer);
    this.#timer = undefined;
    if (this.#active) {
      this.#active = false;
      this.#set(false);
    }
  }
}

/** Runs the scene's field interaction during gizmo drags and sculpt strokes. */
export function useFieldInteraction(app: AppState, scene: Scene | WorkerScene): void {
  useEffect(() => {
    const timer = new InteractionTimer((active) => {
      scene.setFieldInteraction(active ? INTERACTION_POINTS : null);
      interaction.active = active;
      for (const l of interaction.listeners) {
        l(active);
      }
    });
    const onDrag = (dragging: boolean) => (dragging ? timer.press() : timer.release());
    // The capture phase: before a first dab or drag move leaves.
    const onDown = (e: PointerEvent) => {
      if (e.button === 0 && e.target === view.canvas && app.getSnapshot().sculpt !== null) {
        timer.press();
      }
    };
    const onMove = () => timer.touch();
    const onUp = () => {
      if (!view.dragging) {
        timer.release();
      }
    };
    const off = scene.onChanges((changes) => {
      if (timer.pressed && changes.some((c) => c.placementChanged || c.positions === "all" || c.positions.length > 0)) {
        timer.touch();
      }
    });
    view.dragListeners.add(onDrag);
    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      off();
      view.dragListeners.delete(onDrag);
      window.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      timer.release();
    };
  }, [app, scene]);
}
