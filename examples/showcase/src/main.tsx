// The fast-mesh-wasm showcase: load meshes (STL, OBJ, PLY), move them, undo.
// The engine is a `WorkerScene` behind a `SceneStore` (threaded on this
// cross-origin isolated page); `?local` runs the synchronous `Scene` on the
// main thread instead. `?big` makes the demo meshes A and B about a million
// triangles each; `?threads=N` asks for N engine threads.
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createWorkerScene, type WorkerScene } from "fast-mesh-wasm/worker";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App, type Engine } from "./App";
import { Loader } from "./load/loader";
import { AppState } from "./scene/app-state";
import { installTestHook } from "./test-hook";
import "./styles.css";

const params = new URLSearchParams(location.search);
const local = params.has("local");
const threadsParam = params.get("threads");

async function createEngine(): Promise<{ store: SceneStore; engine: Engine }> {
  const isolated = globalThis.crossOriginIsolated === true;
  if (local) {
    // Loaded only in this mode, so the Worker mode never fetches the wasm on the main thread.
    const { init, Scene } = await import("fast-mesh-wasm");
    await init();
    return { store: createSceneStore(new Scene()), engine: { mode: "local", threads: () => 1, isolated } };
  }
  const worker: WorkerScene = await createWorkerScene(
    threadsParam === null ? {} : { threads: Math.max(1, Math.round(Number(threadsParam)) || 1) },
  );
  return { store: createSceneStore(worker), engine: { mode: "worker", threads: () => worker.threads, isolated } };
}

const root = createRoot(document.getElementById("root")!);

try {
  const { store, engine } = await createEngine();
  const app = new AppState();
  const loader = new Loader(store, app);
  root.render(
    <StrictMode>
      <SceneProvider store={store}>
        <App engine={engine} app={app} loader={loader} />
      </SceneProvider>
    </StrictMode>,
  );
  installTestHook(store, engine, app);
  // Seeded here, outside React: StrictMode runs effects twice in development,
  // which would add every mesh twice.
  void loader.loadDemo(params.has("big"));
} catch (error) {
  console.error(error);
  root.render(<div className="fatal">Could not start the scene: {String(error)}</div>);
}
