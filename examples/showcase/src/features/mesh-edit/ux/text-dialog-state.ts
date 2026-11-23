// The Text dialog's shared state (#260): whether it is open, a count of its
// openings (each starts a fresh form), and the live preview the viewport
// layer draws (the text mesh and where it would land). One per app, kept
// beside the AppState as the Create dialog's is: the ribbon, the command
// palette, the dialog and the viewport layer reach it through
// `textDialogOf(app)`. Opening it closes the Create dialog, which floats in
// the same place.
import { useSyncExternalStore } from "react";
import type { Vec3 } from "../../../io/mesh-data";
import { createDialogOf } from "../../home/ux/create-dialog-state";

/** What the viewport previews: the text mesh at the dialog's centre and rotation. */
export interface TextPreview {
  mesh: { positions: Float32Array; indices: Uint32Array };
  centre: Vec3;
  /** Euler XYZ degrees. */
  rotationDeg: Vec3;
}

export interface TextDialogSnapshot {
  open: boolean;
  /** Counts the openings: the dialog starts afresh on each. */
  session: number;
  preview: TextPreview | null;
}

export class TextDialogState {
  #snap: TextDialogSnapshot = { open: false, session: 0, preview: null };
  readonly #listeners = new Set<() => void>();
  readonly #app: object;

  constructor(app: object) {
    this.#app = app;
  }

  getSnapshot = (): TextDialogSnapshot => this.#snap;

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  #set(patch: Partial<TextDialogSnapshot>): void {
    this.#snap = { ...this.#snap, ...patch };
    for (const l of this.#listeners) {
      l();
    }
  }

  /** Opens the dialog afresh, closing the Create dialog. */
  open(): void {
    createDialogOf(this.#app).close();
    this.#set({ open: true, session: this.#snap.session + 1, preview: null });
  }

  /** Closes the dialog and drops its preview. */
  close(): void {
    if (this.#snap.open || this.#snap.preview !== null) {
      this.#set({ open: false, preview: null });
    }
  }

  /** Shows `preview` (or nothing) in the viewport while the dialog is open. */
  setPreview(preview: TextPreview | null): void {
    if (preview === this.#snap.preview || (preview !== null && !this.#snap.open)) {
      return;
    }
    this.#set({ preview });
  }
}

const states = new WeakMap<object, TextDialogState>();

/** The Text dialog state of `app`, made on first use. */
export function textDialogOf(app: object): TextDialogState {
  let state = states.get(app);
  if (state === undefined) {
    state = new TextDialogState(app);
    states.set(app, state);
  }
  return state;
}

/** A slice of `app`'s Text dialog state, re-rendering when it changes. */
export function useTextDialog<T>(app: object, select: (s: TextDialogSnapshot) => T): T {
  const state = textDialogOf(app);
  return useSyncExternalStore(state.subscribe, () => select(state.getSnapshot()));
}
