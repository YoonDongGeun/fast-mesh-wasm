// Code fetched on demand (#285). A chunk is imported once and kept; a failed
// import (offline, a new deploy) can be tried again. Parts of the page that
// wait for their chunk are counted, so that what wants a part's controls (a
// command, the tour) can tell "not here yet" from "not there". No React.

export type ChunkState<M> =
  | { readonly status: "idle" }
  | { readonly status: "loading" }
  | { readonly status: "ready"; readonly module: M }
  | { readonly status: "failed"; readonly error: unknown };

export class LazyChunk<M = unknown> {
  readonly name: string;
  readonly #import: () => Promise<M>;
  #state: ChunkState<M> = { status: "idle" };
  #load: Promise<boolean> | null = null;
  readonly #listeners = new Set<() => void>();

  constructor(name: string, importer: () => Promise<M>) {
    this.name = name;
    this.#import = importer;
  }

  getSnapshot = (): ChunkState<M> => this.#state;

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  #set(state: ChunkState<M>): void {
    this.#state = state;
    for (const l of this.#listeners) {
      l();
    }
  }

  /** Imports the chunk: once, and again after a failure. Never rejects: whether the chunk is here. */
  load(): Promise<boolean> {
    if (this.#load === null) {
      this.#set({ status: "loading" });
      this.#load = this.#import().then(
        (module) => {
          this.#set({ status: "ready", module });
          return true;
        },
        (error: unknown) => {
          this.#load = null;
          console.warn(`The ${this.name} chunk did not load.`, error);
          this.#set({ status: "failed", error });
          return false;
        },
      );
    }
    return this.#load;
  }
}

/** The mounted parts whose content is not shown yet, by chunk. */
const waiting = new Map<LazyChunk, number>();

/** Counts a mounted part that waits for `chunk`. The result ends the wait: its content mounted, or it left the page. */
export function waitForChunk(chunk: LazyChunk): () => void {
  waiting.set(chunk, (waiting.get(chunk) ?? 0) + 1);
  let done = false;
  return () => {
    if (done) {
      return;
    }
    done = true;
    const left = (waiting.get(chunk) ?? 1) - 1;
    if (left > 0) {
      waiting.set(chunk, left);
    } else {
      waiting.delete(chunk);
    }
  };
}

/** Starts loading every chunk a mounted part waits for (failed ones again); `false` when none does. */
export function loadWaitingParts(): boolean {
  for (const chunk of waiting.keys()) {
    void chunk.load();
  }
  return waiting.size > 0;
}

/** Whether a mounted part still waits for a chunk that has not failed. */
export function partsArriving(): boolean {
  return [...waiting.keys()].some((chunk) => chunk.getSnapshot().status !== "failed");
}
