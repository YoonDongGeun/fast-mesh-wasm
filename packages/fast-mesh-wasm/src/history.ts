// The undo history's memory budget, the report of steps larger than it, and
// entries of the application's own, shared by `Scene`, `WorkerScene` and
// `SceneStore`. No WebAssembly here: the Worker client and the store load it
// without the glue.
import { FastMeshError, reportListenerError } from "./core.js";
import type { MeshHandle } from "./handles.js";

/**
 * The default history budget: 256 MiB. See {@link SceneOptions.historyBudget}.
 */
export const DEFAULT_HISTORY_BUDGET = 256 * 1024 * 1024;

/** The largest budget the engine counts with: its 32-bit address space. */
const MAX_HISTORY_BUDGET = 2 ** 32 - 1;

/**
 * A step the undo history could not keep, since it is larger than the whole
 * budget; from {@link Scene.onPermanent}.
 */
export interface PermanentStep {
  /**
   * What can no longer be done with it: `"undo"` for a step just made (or
   * redone), which is now permanent; `"redo"` for one just undone.
   */
  lost: "undo" | "redo";
  /**
   * The meshes the step edited, moved, added or removed, each once. Those in
   * `gone` are out of the scene, and calls on them give `UNKNOWN_MESH`.
   */
  meshes: readonly MeshHandle[];
  /** Of those, the meshes that left the scene with the step, for good. */
  gone: readonly MeshHandle[];
  /** The bytes the step counted, more than the history budget. */
  bytes: number;
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** Called with each step the history could not keep. */
export type PermanentListener = (step: PermanentStep) => void;

/**
 * An entry of the application's own in the undo history, for
 * {@link Scene.recordEntry}. The object itself can be the payload: its
 * functions are called as methods of it.
 */
export interface HistoryEntry {
  /**
   * Takes the application back to before the entry. Called when undo reaches
   * it, or its open step is cancelled, once the step's meshes have changed.
   */
  undo(): void;
  /** Takes the application to after the entry again. Called when redo reaches it. */
  redo(): void;
  /**
   * Called once when the history lets go of the entry: neither `undo` nor
   * `redo` is called any more, so what they need can be released. That is
   * when its step is evicted for the budget, its redo branch is dropped by a
   * newer step, its step was cancelled (after `undo`), the history is
   * cleared, or the scene is freed, terminated or restarted.
   */
  drop?(): void;
  /**
   * About the bytes the entry keeps alive, counted against
   * {@link SceneOptions.historyBudget} until it is dropped, with a few for
   * the entry itself. 0 by default.
   */
  readonly bytes?: number;
  /**
   * Within an open step, an entry recorded right after one with the same
   * `coalesce` merges with it, as the moves of a drag do: undo then calls the
   * first one's `undo`, redo the last one's `redo`, and those in between are
   * dropped. Anything else recorded in between starts a new run.
   */
  readonly coalesce?: string;
}

/** Checks an entry; its cost in bytes and its merge key. */
export function checkHistoryEntry(entry: HistoryEntry): { bytes: number; coalesce: string | undefined } {
  const { undo, redo, drop, bytes = 0, coalesce } = entry ?? ({} as Partial<HistoryEntry>);
  if (typeof undo !== "function" || typeof redo !== "function" || (drop !== undefined && typeof drop !== "function")) {
    throw invalid("a history entry needs undo and redo functions (and may have drop)");
  }
  if (typeof bytes !== "number" || Number.isNaN(bytes) || bytes < 0) {
    throw invalid(`a history entry's bytes must be a number, 0 or more (got ${String(bytes)})`);
  }
  if (coalesce !== undefined && typeof coalesce !== "string") {
    throw invalid("a history entry's coalesce must be a string");
  }
  return { bytes, coalesce };
}

/** What happened to an entry: `what` as `WasmScene.take_entry_events` codes it. */
export const ENTRY_EVENTS = ["undo", "redo", "drop"] as const;
/** See {@link ENTRY_EVENTS}. */
export type EntryEvent = (typeof ENTRY_EVENTS)[number];

/**
 * The application's entries a scene holds, by the ids the engine knows them
 * by. What a callback throws is reported, never thrown: the undo it is part
 * of has happened.
 */
export class EntryBook {
  readonly #entries = new Map<number, { entry: HistoryEntry; rev: number }>();
  #nextId = 1;
  #latest = 0;

  get size(): number {
    return this.#entries.size;
  }

  /** Keeps an entry recorded at revision `rev` (later ones have higher ids and revisions). Returns its id. */
  add(entry: HistoryEntry, rev = 0): number {
    const id = this.#nextId++;
    this.#entries.set(id, { entry, rev });
    this.#latest = rev;
    return id;
  }

  /** Forgets an entry that was never recorded, without a call. */
  forget(id: number): void {
    this.#entries.delete(id);
  }

  /** Calls what the event asks for; nothing for an entry already dropped. */
  tell(id: number, what: EntryEvent): void {
    const entry = this.#entries.get(id)?.entry;
    if (entry === undefined) {
      return;
    }
    if (what === "drop") {
      this.#entries.delete(id);
    }
    try {
      entry[what]?.();
    } catch (error) {
      reportListenerError(error);
    }
  }

  /** The entries recorded after revision `rev`, oldest first. */
  after(rev: number): number[] {
    if (this.#latest <= rev) {
      return [];
    }
    return [...this.#entries].filter(([, e]) => e.rev > rev).map(([id]) => id);
  }

  /** Drops every entry: the history is gone. Their `drop` is called on a microtask. */
  dropAll(): void {
    const all = [...this.#entries.values()];
    this.#entries.clear();
    if (all.some(({ entry }) => entry.drop !== undefined)) {
      queueMicrotask(() => {
        for (const { entry } of all) {
          try {
            entry.drop?.();
          } catch (error) {
            reportListenerError(error);
          }
        }
      });
    }
  }
}

/** Checks a history budget in bytes: rounded down, clamped to what the engine counts with. */
export function checkHistoryBudget(bytes: unknown): number {
  if (typeof bytes !== "number" || Number.isNaN(bytes) || bytes < 0) {
    throw invalid(`historyBudget must be a number of bytes, 0 or more (got ${String(bytes)})`);
  }
  return Math.min(Math.floor(bytes), MAX_HISTORY_BUDGET);
}

/** Decodes the message of `WasmScene.take_permanent` (see crates/fast-mesh-wasm/src/scene/history.rs). */
export function decodePermanent(
  msg: Float64Array,
  handle: (index: number, generation: number) => MeshHandle | undefined,
): PermanentStep[] {
  let at = 0;
  const meshes = (): MeshHandle[] => {
    const out: MeshHandle[] = [];
    for (let n = msg[at++]; n > 0; n--) {
      const h = handle(msg[at], msg[at + 1]);
      at += 2;
      if (h !== undefined) {
        out.push(h);
      }
    }
    return out;
  };
  const out: PermanentStep[] = [];
  while (at < msg.length) {
    const lost = msg[at++] === 0 ? "undo" : "redo";
    const bytes = msg[at++];
    out.push({ lost, bytes, meshes: meshes(), gone: meshes() });
  }
  return out;
}

/** Calls every listener with every step; errors are reported, not thrown. */
export function tellPermanent(listeners: ReadonlySet<PermanentListener>, steps: readonly PermanentStep[]): void {
  for (const step of steps) {
    for (const listener of [...listeners]) {
      try {
        listener(step);
      } catch (error) {
        reportListenerError(error);
      }
    }
  }
}
