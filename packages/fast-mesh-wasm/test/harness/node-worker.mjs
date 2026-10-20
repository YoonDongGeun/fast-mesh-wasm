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
// `{ yieldMs }` sets the engine's yield budget (`0`: yield after every chunk).
import { parentPort, workerData } from "node:worker_threads";
import "../../dist/node.js";
import * as raw from "../../dist/wasm/fast_mesh_wasm.js";
import { runEngine, serveEngine } from "../../dist/worker-engine.js";

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
if (workerData?.serve) {
  serveEngine(port, Promise.resolve(), { debug: true, yieldMs: workerData?.yieldMs });
} else {
  runEngine(port, { yieldMs: workerData?.yieldMs });
}
