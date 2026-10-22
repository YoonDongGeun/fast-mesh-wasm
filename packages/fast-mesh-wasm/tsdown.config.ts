import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    web: "src/web.ts",
    node: "src/node.ts",
    three: "src/three/index.ts",
    // `fast-mesh-wasm/worker` (main thread) and the browser Worker script it
    // loads by URL (`new URL("./worker-entry.js", import.meta.url)`), which
    // must stay a file next to it. Both import the wasm glue in dist/wasm.
    worker: "src/worker/index.ts",
    "worker-entry": "src/worker/entry.ts",
    // `fast-mesh-wasm/store`: free of React, three.js and the wasm glue.
    store: "src/store/index.ts",
    // `fast-mesh-wasm/react`: hooks over the store; React is a peer.
    react: "src/react/index.ts",
    // `fast-mesh-wasm/r3f`: `<FastMesh>` and the frame driver; R3F is a peer.
    r3f: "src/r3f/index.ts",
    // Internal (not in `exports`): the Worker-side engine, for the Node
    // `worker_threads` test harness.
    "worker-engine": "src/worker/engine.ts",
  },
  format: "esm",
  platform: "neutral",
  dts: true,
  deps: {
    // Keep the wasm-bindgen glue as a separate file so its `new URL(..., import.meta.url)`
    // keeps pointing at the .wasm next to it. It is copied into dist/wasm below.
    // `three`, `react` and `@react-three/fiber` are optional peers: imported, never inlined.
    // The threaded glue (dist/wasm-threads) is never bundled either: only the
    // Worker entry may load it, and its rayon worker helpers import it by
    // relative path.
    neverBundle: [/fast_mesh_wasm\.js$/, /wasm-threads\//, /^node:/, /^three($|\/)/, /^react($|\/)/, /^@react-three\/fiber($|\/)/],
  },
  // src/wasm-threads is the threaded artifact (`build-wasm.mjs --threads`), with its
  // rayon worker helpers under snippets/ and the package.json they resolve it by.
  copy: [
    { from: "src/wasm", to: "dist" },
    { from: "src/wasm-threads", to: "dist" },
  ],
});
