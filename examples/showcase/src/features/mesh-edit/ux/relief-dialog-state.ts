// The Image to Relief dialog's shared state (#267): whether it is open, a
// count of its openings (each starts a fresh form), the files it was handed
// (dropped on the page, say), and the live preview the viewport layer draws.
// One per app, kept beside the AppState as the Create and Text dialogs' are.
// The three float in the same place: opening this one closes the others.
import { useSyncExternalStore } from "react";
import type { Vec3 } from "../../../io/mesh-data";
import { createDialogOf } from "../../home/ux/create-dialog-state";
import type { ReliefFiles } from "./relief";
import { textDialogOf } from "./text-dialog-state";

/** What the viewport previews: a coarse relief at the dialog's centre and rotation. */
export interface ReliefPreview {
  mesh: { positions: Float32Array; indices: Uint32Array };
  centre: Vec3;
  /** Euler XYZ degrees. */
  rotationDeg: Vec3;
}

export interface ReliefDialogSnapshot {
  open: boolean;
  /** Counts the openings: the dialog starts afresh on each. */
  session: number;
  /** The files last handed to the dialog, if any since it opened. */
  files: ReliefFiles | null;
  preview: ReliefPreview | null;
}

export class ReliefDialogState {
  #snap: ReliefDialogSnapshot = { open: false, session: 0, files: null, preview: null };
  readonly #listeners = new Set<() => void>();
  readonly #app: object;

  constructor(app: object) {
    this.#app = app;
  }

  getSnapshot = (): ReliefDialogSnapshot => this.#snap;

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  #set(patch: Partial<ReliefDialogSnapshot>): void {
    this.#snap = { ...this.#snap, ...patch };
    for (const l of this.#listeners) {
      l();
    }
  }

  /** Opens the dialog afresh, closing the dialogs in its place. */
  open(files: ReliefFiles | null = null): void {
    createDialogOf(this.#app).close();
    textDialogOf(this.#app).close();
    this.#set({ open: true, session: this.#snap.session + 1, files, preview: null });
  }

  /** Hands files to the dialog, opening it if need be. */
  take(files: ReliefFiles): void {
    if (this.#snap.open) {
      this.#set({ files });
    } else {
      this.open(files);
    }
  }

  /** Closes the dialog and drops its preview. */
  close(): void {
    if (this.#snap.open || this.#snap.preview !== null) {
      this.#set({ open: false, files: null, preview: null });
    }
  }

  /** Shows `preview` (or nothing) in the viewport while the dialog is open. */
  setPreview(preview: ReliefPreview | null): void {
    if (preview === this.#snap.preview || (preview !== null && !this.#snap.open)) {
      return;
    }
    this.#set({ preview });
  }
}

const states = new WeakMap<object, ReliefDialogState>();

/** The Image to Relief dialog state of `app`, made on first use. */
export function reliefDialogOf(app: object): ReliefDialogState {
  let state = states.get(app);
  if (state === undefined) {
    state = new ReliefDialogState(app);
    states.set(app, state);
  }
  return state;
}

/** A slice of `app`'s Image to Relief dialog state, re-rendering when it changes. */
export function useReliefDialog<T>(app: object, select: (s: ReliefDialogSnapshot) => T): T {
  const state = reliefDialogOf(app);
  return useSyncExternalStore(state.subscribe, () => select(state.getSnapshot()));
}
