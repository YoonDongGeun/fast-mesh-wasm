// A `worker_threads` entry running the Worker-side engine on `parentPort`,
// standing in for the browser Worker script in tests. It loads the built
// package (dist/): the node entry instantiates the wasm on import, and the
// engine shares its modules, so the engine starts with the wasm ready.
//
// workerData `{ testPanic: true }` makes the `analyze` binding trap through
// `__test_panic` (the test build of the wasm only), to exercise poisoning;
// `{ packThrow: true }` makes `can_undo` (read while packing a result) throw
// a plain error; `{ undoThrow: true }` makes `undo` fail with a coded
// `INVALID_ARGUMENT` error (a failed history request). `{ serve: true }` runs it as the browser Worker script does
// (`serveEngine`: a `ready` message first) with the test-only `debugDump`
// query, for `WorkerScene` through its `createWorker` option.
// `{ yieldMs }` sets the engine's yield budget (`0`: yield after every chunk);
// `{ stepPoints }` and `{ staleGuardMs }` set the field step options, and
// `{ reportSteps: true }` posts `{ kind: "chunk", chunk }` for every field step
// and abandoned unit (`fieldStep`, `fieldStepAbandon`), besides the results.
//
// Served, the engine starts on the scene's init message through the Worker's
// artifact selection (`startEngine`), here not cross-origin isolated (Node has
// no `crossOriginIsolated`): single-threaded, no warnings. `{ isolated }` poses
// as a page reporting that `crossOriginIsolated`, and `{ threaded }` replaces
// the threaded glue: `"fake"` starts (the single-threaded glue, with an
// `initThreadPool` that posts `{ kind: "pool", threads }`), `"real"` is the
// built dist/wasm-threads glue (it instantiates, but its pool cannot start: no
// nested `Worker` in Node), `"memFail"` is that glue failing to allocate its
// shared memory (`new WebAssembly.Memory` throws a RangeError), and
// `"importFail"` an import that rejects.
import { readFileSync } from "node:fs";
import { parentPort, workerData } from "node:worker_threads";
import "../../dist/node.js";
import * as raw from "../../dist/wasm/fast_mesh_wasm.js";
import { runEngine, serveEngine, startEngine } from "../../dist/worker-engine.js";

if (workerData?.testPanic) {
  raw.WasmScene.prototype.analyze = () => raw.__test_panic();
}
if (workerData?.packThrow) {
  raw.WasmScene.prototype.can_undo = () => {
    throw new Error("boom");
  };
}

if (workerData?.undoThrow) {
  raw.WasmScene.prototype.undo = () => {
    throw new Error("INVALID_ARGUMENT: undo failed (test)");
  };
}

const port = {
  post: (msg, transfer) => parentPort.postMessage(msg, transfer),
  onMessage: (cb) => parentPort.on("message", cb),
};
const options = {
  yieldMs: workerData?.yieldMs,
  stepPoints: workerData?.stepPoints,
  staleGuardMs: workerData?.staleGuardMs,
  onChunk: workerData?.reportSteps
    ? (chunk) => {
        if (chunk.kind === "fieldStep" || chunk.kind === "fieldStepAbandon") {
          parentPort.postMessage({ kind: "chunk", chunk });
        }
      }
    : undefined,
};
/** The threaded glue's loader for `workerData.threaded` (see above). */
function threadedLoader(kind) {
  const threadedUrl = new URL("../../dist/wasm-threads/", import.meta.url);
  const real = async () => {
    // The rayon helper listens on the Worker's `self` when it loads; Node has
    // none. A stand-in that never fires lets the glue load and instantiate.
    globalThis.self ??= { addEventListener() {}, removeEventListener() {} };
    const glue = await import(new URL("fast_mesh_wasm.js", threadedUrl).href);
    const bytes = readFileSync(new URL("fast_mesh_wasm_bg.wasm", threadedUrl));
    // Node's fetch has no file: URLs: hand the glue its bytes.
    return { ...glue, default: () => glue.default({ module_or_path: bytes }) };
  };
  switch (kind) {
    case "fake":
      return async () => ({
        default: raw.default,
        WasmScene: raw.WasmScene,
        initThreadPool: async (threads) => {
          parentPort.postMessage({ kind: "pool", threads });
        },
      });
    case "real":
      return real;
    case "memFail":
      return async () => {
        const Memory = WebAssembly.Memory;
        WebAssembly.Memory = function (desc) {
          if (desc?.shared) {
            throw new RangeError("WebAssembly.Memory(): could not allocate memory (test)");
          }
          return new Memory(desc);
        };
        return real();
      };
    default:
      return async () => {
        throw new TypeError("Failed to fetch dynamically imported module (test)");
      };
  }
}

if (workerData?.serve) {
  serveEngine(
    port,
    (init) =>
      startEngine(
        init,
        {
          importThreaded: threadedLoader(workerData.threaded),
          // Already instantiated by dist/node.js: its init returns that instance.
          importSingle: async () => raw,
        },
        workerData.isolated,
      ),
    { debug: true, ...options },
  );
} else {
  runEngine(port, options);
}
