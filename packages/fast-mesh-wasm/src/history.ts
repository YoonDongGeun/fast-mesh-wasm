// The undo history's memory budget and the report of steps larger than it,
// shared by `Scene`, `WorkerScene` and `SceneStore`. No WebAssembly here: the
// Worker client and the store load it without the glue.
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

/** Called with each step the history could not keep. */
export type PermanentListener = (step: PermanentStep) => void;

/** Checks a history budget in bytes: rounded down, clamped to what the engine counts with. */
export function checkHistoryBudget(bytes: unknown): number {
  if (typeof bytes !== "number" || Number.isNaN(bytes) || bytes < 0) {
    throw new FastMeshError("INVALID_ARGUMENT", `historyBudget must be a number of bytes, 0 or more (got ${String(bytes)})`);
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
