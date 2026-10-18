// Shared runtime state of the WebAssembly instance: its memory (for views
// over render buffers), error-code mapping and panic poisoning.

/**
 * Error codes of {@link FastMeshError}.
 * `INSTANCE_POISONED` means an earlier call panicked: the WebAssembly instance
 * is unusable and every later call throws it. Reload the module to recover.
 */
export type ErrorCode =
  | "UNKNOWN_MESH"
  | "UNKNOWN_FIELD"
  | "UNKNOWN_SECTION"
  | "INVALID_ARGUMENT"
  | "INVALID_INDEX"
  | "STALE_BVH"
  | "STEP_NOT_OPEN"
  | "CORRUPTED"
  | "OUT_OF_MEMORY"
  | "STALE_LOOPS"
  | "INSTANCE_POISONED";

/** An error thrown by fast-mesh, with a machine-readable `code`. */
export class FastMeshError extends Error {
  /** What went wrong; see {@link ErrorCode}. */
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = "FastMeshError";
    this.code = code;
  }
}

let poisoned = false;
let memory: WebAssembly.Memory | undefined;

/** Stores the instance's memory; called once the module is instantiated. */
export function setMemory(m: WebAssembly.Memory): void {
  memory = m;
}

/**
 * The instance's memory. Its `buffer` is replaced whenever the memory grows,
 * so build views from it right before use.
 */
export function wasmMemory(): WebAssembly.Memory {
  if (memory === undefined) {
    throw new FastMeshError("INVALID_ARGUMENT", "fast-mesh-wasm is not initialized");
  }
  return memory;
}

function poisonedError(): FastMeshError {
  return new FastMeshError(
    "INSTANCE_POISONED",
    "an earlier fast-mesh call panicked; the WebAssembly instance can no longer be used",
  );
}

// Messages from Rust read "<CODE>: <text>".
const CODED = /^([A-Z_]+): ([\s\S]*)$/;

/**
 * Runs a call into WebAssembly. A coded Rust error becomes a
 * {@link FastMeshError}; a trap (a Rust panic) poisons the instance, and from
 * then on every call throws `INSTANCE_POISONED` without entering WebAssembly.
 * Other errors pass through unchanged.
 */
export function call<T>(fn: () => T): T {
  if (poisoned) {
    throw poisonedError();
  }
  try {
    return fn();
  } catch (e) {
    if (e instanceof WebAssembly.RuntimeError) {
      poisoned = true;
      throw poisonedError();
    }
    const m = e instanceof Error ? CODED.exec(e.message) : null;
    if (m !== null) {
      throw new FastMeshError(m[1] as ErrorCode, m[2]);
    }
    throw e;
  }
}
