// The browser Worker script (`dist/worker-entry.js`): loads the WebAssembly
// module, then serves the engine on the Worker's message port. Batches sent
// while the module loads are kept and run once it is ready.
import { init } from "../web.js";
import { serveEngine } from "./engine.js";

/** The parts of a dedicated Worker's global scope used here. */
interface WorkerScope {
  postMessage(msg: unknown, transfer: Transferable[]): void;
  addEventListener(type: "message", cb: (e: MessageEvent) => void): void;
}

const scope = globalThis as unknown as WorkerScope;

serveEngine(
  {
    post: (msg, transfer) => scope.postMessage(msg, transfer),
    onMessage: (cb) => scope.addEventListener("message", (e) => cb(e.data)),
  },
  init(),
);
