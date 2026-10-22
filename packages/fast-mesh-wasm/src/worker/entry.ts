// The browser Worker script (`dist/worker-entry.js`): waits for the scene's
// init message, loads the WebAssembly module (the threaded artifact on cross-
// origin isolated pages, see load.ts), then serves the engine on the Worker's
// message port. Batches sent while the module loads are kept and run once it
// is ready.
import { serveEngine } from "./engine.js";
import { startEngine, type ThreadedGlue } from "./load.js";

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
  (init) =>
    startEngine(init, {
      // Literal relative specifiers (dist/wasm-threads and dist/wasm next to
      // this script), so bundlers see both artifacts. The threaded one is the
      // only import of dist/wasm-threads in the package, and is reached only
      // when the threaded engine is tried. It is absent from a local build
      // without the pinned nightly; the import then fails and the engine
      // falls back.
      // @ts-ignore: src/wasm-threads exists only once built.
      importThreaded: () => import("../wasm-threads/fast_mesh_wasm.js") as Promise<ThreadedGlue>,
      importSingle: () => import("../wasm/fast_mesh_wasm.js"),
    }),
);
