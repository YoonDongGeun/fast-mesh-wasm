// Shared runtime state of the WebAssembly instance: its memory (for views
// over render buffers), error-code mapping and panic poisoning.

/**
 * Error codes of {@link FastMeshError}.
 * `INSTANCE_POISONED` means an earlier call panicked: the WebAssembly instance
 * is unusable and every later call throws it. Reload the module to recover.
 */
export type FastMeshErrorCode =
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

const CODES: ReadonlySet<string> = new Set<FastMeshErrorCode>([
  "UNKNOWN_MESH",
  "UNKNOWN_FIELD",
  "UNKNOWN_SECTION",
  "INVALID_ARGUMENT",
  "INVALID_INDEX",
  "STALE_BVH",
  "STEP_NOT_OPEN",
  "CORRUPTED",
  "OUT_OF_MEMORY",
  "STALE_LOOPS",
  "INSTANCE_POISONED",
]);

/** An error thrown by fast-mesh, with a machine-readable `code`. */
export class FastMeshError extends Error {
  /** What went wrong; see {@link FastMeshErrorCode}. */
  readonly code: FastMeshErrorCode;

  constructor(code: FastMeshErrorCode, message: string) {
    super(message);
    this.name = "FastMeshError";
    this.code = code;
  }
}

/**
 * Internal (not exported from the entries). Reports a listener's error
 * without breaking the caller: `reportError`, else an async rethrow.
 */
export function reportListenerError(error: unknown): void {
  const report = (globalThis as { reportError?: (e: unknown) => void }).reportError;
  if (typeof report === "function") {
    report(error);
  } else {
    setTimeout(() => {
      throw error;
    });
  }
}

let poisoned = false;
let memory: WebAssembly.Memory | undefined;

/** Stores the instance's memory; called once the module is instantiated. */
export function setMemory(m: WebAssembly.Memory): void {
  memory = m;
}

/**
 * Throws the "call `await init()` first" error unless the module is
 * instantiated. `what` names the call, e.g. `"computeBounds()"`.
 */
export function ensureInitialized(what: string): void {
  if (memory === undefined) {
    throw new Error(`fast-mesh-wasm is not initialized: call \`await init()\` before ${what}.`);
  }
}

/**
 * The instance's memory. Its `buffer` is replaced whenever the memory grows,
 * so build views from it right before use.
 */
export function wasmMemory(): WebAssembly.Memory {
  ensureInitialized("reading render views");
  return memory as WebAssembly.Memory;
}

function poisonedError(): FastMeshError {
  return new FastMeshError(
    "INSTANCE_POISONED",
    "an earlier fast-mesh call panicked; the WebAssembly instance can no longer be used",
  );
}

// Messages from Rust read "<CODE>: <text>".
const CODED = /^([A-Z_]+): ([\s\S]*)$/;

// Stack exhaustion: V8 and JavaScriptCore throw a RangeError with this text;
// Firefox throws `InternalError: too much recursion` (a non-standard class,
// matched by name).
const STACK_EXHAUSTED = /call stack/i;

function stackExhausted(e: unknown): boolean {
  return (
    (e instanceof RangeError && STACK_EXHAUSTED.test(e.message)) ||
    (e instanceof Error && e.name === "InternalError")
  );
}

/**
 * Runs a call into WebAssembly. A coded Rust error becomes a
 * {@link FastMeshError}; a trap (a Rust panic) poisons the instance, and from
 * then on every call throws `INSTANCE_POISONED` without entering WebAssembly.
 *
 * Stack exhaustion (a `RangeError` about the call stack, or Firefox's
 * `InternalError`) poisons too: it can
 * unwind out of WebAssembly mid-call, leaving Rust's state half-updated.
 * Other errors, including messages with a `PREFIX: ` that is not a known
 * code, pass through unchanged, so a foreign error is never relabelled.
 */
export function call<T>(fn: () => T): T {
  if (poisoned) {
    throw poisonedError();
  }
  try {
    return fn();
  } catch (e) {
    if (e instanceof WebAssembly.RuntimeError || stackExhausted(e)) {
      poisoned = true;
      throw poisonedError();
    }
    const m = e instanceof Error ? CODED.exec(e.message) : null;
    if (m !== null && CODES.has(m[1])) {
      throw new FastMeshError(m[1] as FastMeshErrorCode, m[2]);
    }
    throw e;
  }
}

/** A wasm-bindgen object: `free()` enters WebAssembly, the rest is JS. */
interface Freeable {
  free(): void;
}

/**
 * Releases a wasm-bindgen object without ever entering a poisoned instance.
 * After poisoning only its JS side is detached (the glue's own finalizer is
 * unregistered, so it cannot enter WebAssembly later either); the memory is
 * lost with the instance anyway.
 */
export function release(obj: Freeable): void {
  if (poisoned) {
    (obj as Freeable & { __destroy_into_raw(): number }).__destroy_into_raw();
    return;
  }
  call(() => obj.free());
}

/**
 * Frees an object once its owner is garbage collected, through
 * {@link release}. Errors are swallowed: a finalizer has no caller to throw to.
 */
export const finalizer = new FinalizationRegistry<Freeable>((obj) => {
  try {
    release(obj);
  } catch {
    // Nothing to report to.
  }
});
