// The fast-mesh-wasm showcase: load meshes (STL, OBJ, PLY), move them, undo.
// The engine is a `WorkerScene` behind a `SceneStore` (threaded on this
// cross-origin isolated page); `?local` runs the synchronous `Scene` on the
// main thread instead. `?threads=N` asks for N engine threads. The app
// boots onto the vessel tree; the demo meshes load from their button.
// No top-level await here: the chunks that load on demand (#285), the engine of
// `?local` among them, import from this one and would wait for it.
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createWorkerScene, type WorkerScene } from "fast-mesh-wasm/worker";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App, type Engine } from "./app/App";
import { Loader } from "./io/loader";
import { loadSettings, saveSettings, settingsFromSnapshot } from "./shared/ux/settings";
import { AppState } from "./app/state/app-state";
import { preloadTab } from "./app/ux/feature-chunks";
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
    const scene = new Scene();
    return { store: createSceneStore(scene), engine: { mode: "local", scene, threads: () => 1, isolated } };
  }
  const worker: WorkerScene = await createWorkerScene(
    threadsParam === null ? {} : { threads: Math.max(1, Math.round(Number(threadsParam)) || 1) },
  );
  return {
    store: createSceneStore(worker),
    engine: { mode: "worker", scene: worker, threads: () => worker.threads, isolated },
  };
}

const root = createRoot(document.getElementById("root")!);

async function start(): Promise<void> {
  try {
    const starting = createEngine();
    // The first screen's panels, and the volume panel of the mesh it boots
    // onto, arrive while the engine starts.
    preloadTab("home");
    preloadTab("volume");
    const { store, engine } = await starting;
    const app = new AppState(loadSettings());
    // `?decimate=0` opts out of decimate-on-load (for testing; see load/big-file.ts).
    // An explicit param wins over the stored prefs; otherwise they stand.
    try {
      const decimate = new URLSearchParams(location.search).get("decimate");
      if (decimate !== null) {
        app.setDecimateOnLoad(decimate !== "0");
      }
    } catch {
      // The stored prefs stand.
    }
    // Every display change persists (prefs only: meshes stay file-based).
    app.subscribe(() => {
      saveSettings(settingsFromSnapshot(app.getSnapshot()));
    });
    // The list follows undone and redone adds and removals.
    app.follow(store);
    const loader = new Loader(store, app);
    root.render(
      <StrictMode>
        <SceneProvider store={store}>
          <App engine={engine} app={app} loader={loader} />
        </SceneProvider>
      </StrictMode>,
    );
    // Read by the checks, not by the page: after the first render, in a chunk of its own.
    import("./app/testing/test-hook").then(
      ({ installTestHook }) => installTestHook(store, engine, app, loader),
      (error: unknown) => console.warn("The test hook did not load.", error),
    );
  } catch (error) {
    console.error(error);
    root.render(<div className="fatal">Could not start the scene: {String(error)}</div>);
  }
}

void start();
