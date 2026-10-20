// fast-mesh-wasm with React and React Three Fiber, on a `SceneStore`
// (`fast-mesh-wasm/store`). The scene runs in a Web Worker by default
// (`createWorkerScene` from `fast-mesh-wasm/worker`); `?local` in the URL
// runs it on the main thread with the synchronous `Scene` instead. The
// components are the same in both modes: they only talk to the store.
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createWorkerScene } from "fast-mesh-wasm/worker";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { type Item, makeItem, SECTION_NORMAL } from "./items";
import { sphere, wavyPlate } from "./meshes";

const local = new URLSearchParams(location.search).has("local");

async function createStore(): Promise<SceneStore> {
  if (local) {
    // Loaded only in this mode, so the Worker mode never fetches the wasm on the main thread.
    const { init, Scene } = await import("fast-mesh-wasm");
    await init();
    return createSceneStore(new Scene());
  }
  // The default Worker URL (`new URL("./worker-entry.js", import.meta.url)`
  // inside the package) is picked up by Vite as a Worker entry: no Vite
  // configuration is needed.
  return createSceneStore(await createWorkerScene());
}

const root = createRoot(document.getElementById("root")!);

try {
  const store = await createStore();
  // The scene is seeded here, outside React: StrictMode runs initialisers
  // twice in development, which would add every mesh twice.
  const items: Item[] = [
    makeItem(store, "sphere", sphere([0, 1.1, 0], 1), 0xd9735f),
    makeItem(store, "wavy plate", wavyPlate([0, 0, 0], 4, 0.25), 0x8fc46e),
  ];
  // One section for the app's lifetime: off while it cuts no meshes.
  const section = store.addSection({ plane: { origin: [0, 0, 0], normal: SECTION_NORMAL }, meshes: [] });
  root.render(
    <StrictMode>
      <SceneProvider store={store}>
        <App initialItems={items} section={section} />
      </SceneProvider>
    </StrictMode>,
  );
} catch (error) {
  console.error(error);
  root.render(<div className="panel">Could not start the scene: {String(error)}</div>);
}
