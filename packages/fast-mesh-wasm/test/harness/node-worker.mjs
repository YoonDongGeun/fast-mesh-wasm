// A `worker_threads` entry running the Worker-side engine on `parentPort`,
// standing in for the browser Worker script in tests. It loads the built
// package (dist/): the node entry instantiates the wasm on import, and the
// engine shares its modules, so the engine starts with the wasm ready.
//
// workerData `{ testPanic: true }` makes the `analyze` binding trap through
// `__test_panic` (the test build of the wasm only), to exercise poisoning;
// `{ packThrow: true }` makes `can_undo` (read while packing a result) throw
// a plain error.
import { parentPort, workerData } from "node:worker_threads";
import "../../dist/node.js";
import * as raw from "../../dist/wasm/fast_mesh_wasm.js";
import { runEngine } from "../../dist/worker-engine.js";

if (workerData?.testPanic) {
  raw.WasmScene.prototype.analyze = () => raw.__test_panic();
}
if (workerData?.packThrow) {
  raw.WasmScene.prototype.can_undo = () => {
    throw new Error("boom");
  };
}

runEngine({
  post: (msg, transfer) => parentPort.postMessage(msg, transfer),
  onMessage: (cb) => parentPort.on("message", cb),
});
