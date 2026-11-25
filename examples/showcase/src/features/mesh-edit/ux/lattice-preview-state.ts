// The lattice preview the viewport draws (#265): the walls of a coarse
// lattice and the mesh they are in. One per app, kept beside the AppState as
// the Create dialog's preview is: the panel section sets it, and the viewport
// layer (which draws the walls) and the previewed mesh (which turns
// see-through) read it through `latticePreviewOf(app)`.
import type { LatticePreview, MeshHandle } from "fast-mesh-wasm";
import { useSyncExternalStore } from "react";

/** The walls of a lattice, in the local frame of the mesh they are in. */
export interface ShownLattice {
  mesh: MeshHandle;
  walls: LatticePreview;
}

export class LatticePreviewState {
  #shown: ShownLattice | null = null;
  readonly #listeners = new Set<() => void>();

  getSnapshot = (): ShownLattice | null => this.#shown;

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  /** Shows `shown` in the viewport, or nothing. */
  show(shown: ShownLattice | null): void {
    if (shown === this.#shown) {
      return;
    }
    this.#shown = shown;
    for (const l of this.#listeners) {
      l();
    }
  }
}

const states = new WeakMap<object, LatticePreviewState>();

/** The lattice preview of `app`, made on first use. */
export function latticePreviewOf(app: object): LatticePreviewState {
  let state = states.get(app);
  if (state === undefined) {
    state = new LatticePreviewState();
    states.set(app, state);
  }
  return state;
}

/** What `app` previews, re-rendering when it changes. */
export function useLatticePreview(app: object): ShownLattice | null {
  const state = latticePreviewOf(app);
  return useSyncExternalStore(state.subscribe, state.getSnapshot);
}
