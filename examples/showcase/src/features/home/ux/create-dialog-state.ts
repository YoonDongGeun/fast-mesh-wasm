// The Create dialog's shared state (#259): whether it is open, the shape it
// opened on, and the live preview the viewport layer draws (the generated
// mesh and where it would land). One per app, kept beside the AppState
// rather than in it: the ribbon, the command palette, the dialog and the
// viewport layer reach it through `createDialogOf(app)`.
import type { PrimitiveMesh, PrimitiveShapeKind } from "fast-mesh-wasm";
import { useSyncExternalStore } from "react";
import type { Vec3 } from "../../../io/mesh-data";

/** What the viewport previews: the generated mesh at the dialog's centre and rotation. */
export interface CreatePreview {
  mesh: PrimitiveMesh;
  centre: Vec3;
  /** Euler XYZ degrees. */
  rotationDeg: Vec3;
}

export interface CreateDialogSnapshot {
  open: boolean;
  /** The shape the dialog opened on (it keeps its own choice after that). */
  kind: PrimitiveShapeKind;
  /** Counts the openings: the dialog starts afresh on each. */
  session: number;
  preview: CreatePreview | null;
}

export class CreateDialogState {
  #snap: CreateDialogSnapshot = { open: false, kind: "box", session: 0, preview: null };
  readonly #listeners = new Set<() => void>();

  getSnapshot = (): CreateDialogSnapshot => this.#snap;

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  #set(patch: Partial<CreateDialogSnapshot>): void {
    this.#snap = { ...this.#snap, ...patch };
    for (const l of this.#listeners) {
      l();
    }
  }

  /** Opens the dialog afresh on `kind` (default: the last one). */
  open(kind: PrimitiveShapeKind = this.#snap.kind): void {
    this.#set({ open: true, kind, session: this.#snap.session + 1, preview: null });
  }

  /** Closes the dialog and drops its preview. */
  close(): void {
    if (this.#snap.open || this.#snap.preview !== null) {
      this.#set({ open: false, preview: null });
    }
  }

  /** Remembers the dialog's current shape, for its next opening. */
  setKind(kind: PrimitiveShapeKind): void {
    if (this.#snap.kind !== kind) {
      this.#set({ kind });
    }
  }

  /** Shows `preview` (or nothing) in the viewport while the dialog is open. */
  setPreview(preview: CreatePreview | null): void {
    if (preview === this.#snap.preview || (preview !== null && !this.#snap.open)) {
      return;
    }
    this.#set({ preview });
  }
}

const states = new WeakMap<object, CreateDialogState>();

/** The Create dialog state of `app`, made on first use. */
export function createDialogOf(app: object): CreateDialogState {
  let state = states.get(app);
  if (state === undefined) {
    state = new CreateDialogState();
    states.set(app, state);
  }
  return state;
}

/** A slice of `app`'s Create dialog state, re-rendering when it changes. */
export function useCreateDialog<T>(app: object, select: (s: CreateDialogSnapshot) => T): T {
  const state = createDialogOf(app);
  return useSyncExternalStore(state.subscribe, () => select(state.getSnapshot()));
}
