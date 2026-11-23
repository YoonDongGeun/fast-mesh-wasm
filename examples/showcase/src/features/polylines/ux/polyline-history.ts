// Polyline edits in the scene's undo history (#268). The history is the
// engine's and knows meshes, not polylines, so each state of the polyline
// list has a marker in it: an empty mesh that is in the scene exactly while
// the list is in that state. An edit swaps the markers (an undoable removal
// and an undoable add) as one step, or as part of the open one; undo and redo
// swap them back, in order with every other edit, and the list follows the
// marker it finds. Markers are never listed or drawn. A state goes with its
// marker, once the history has dropped the step that could bring it back.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { Polyline } from "./polyline";

/** The scene calls the history makes. */
export type HistoryStore = Pick<SceneStore, "addMesh" | "removeMesh" | "beginStep" | "endStep" | "isRestorable" | "getSnapshot">;

export class PolylineHistory {
  /** The list's state by marker: the current one, and those undo or redo can bring back. */
  readonly #states = new Map<MeshHandle, readonly Polyline[]>();
  /** The marker of the list's current state (`null` before the first edit). */
  #current: MeshHandle | null = null;

  /** Makes the edit from `before` to `after` an undo step, or part of the open one. */
  record(store: HistoryStore, before: readonly Polyline[], after: readonly Polyline[]): void {
    if (this.#current === null || !store.getSnapshot().meshes.has(this.#current)) {
      // The state undo goes back to: not a step of its own.
      this.#current = marker(store, false);
    }
    const from = this.#current;
    this.#states.set(from, before);
    store.beginStep();
    try {
      store.removeMesh(from, { undoable: true });
      this.#current = marker(store, true);
      this.#states.set(this.#current, after);
    } finally {
      store.endStep();
    }
    this.#prune(store);
  }

  /** The list's state after an undo, a redo or a cancelled step, or `null` while it has not changed. */
  follow(store: HistoryStore): readonly Polyline[] | null {
    if (this.#current === null) {
      return null;
    }
    const { meshes } = store.getSnapshot();
    if (meshes.has(this.#current)) {
      return null;
    }
    for (const [m, state] of this.#states) {
      if (m !== this.#current && meshes.has(m)) {
        this.#current = m;
        this.#prune(store);
        return state;
      }
    }
    return null;
  }

  /** Forgets every state, and takes the marker out for good: the list is replaced, as by opening a project. */
  reset(store: HistoryStore): void {
    const current = this.#current;
    this.#current = null;
    this.#states.clear();
    if (current !== null && store.getSnapshot().meshes.has(current)) {
      store.removeMesh(current, { undoable: false });
    }
  }

  /** The states held (tests). */
  get size(): number {
    return this.#states.size;
  }

  /** Forgets the states nothing can bring back any more. */
  #prune(store: HistoryStore): void {
    for (const m of [...this.#states.keys()]) {
      if (m !== this.#current && !store.getSnapshot().meshes.has(m) && !store.isRestorable(m)) {
        this.#states.delete(m);
      }
    }
  }
}

/** Adds an empty mesh; with `undoable` as an undo step, or part of the open one. */
function marker(store: HistoryStore, undoable: boolean): MeshHandle {
  return store.addMesh(new Float64Array(0), new Uint32Array(0), undoable ? { undoable: true } : {});
}
