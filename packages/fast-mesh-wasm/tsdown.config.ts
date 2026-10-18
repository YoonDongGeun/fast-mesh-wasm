import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    web: "src/web.ts",
    node: "src/node.ts",
    three: "src/three/index.ts",
    // Internal (not in `exports`): the Worker-side engine, for the browser
    // Worker script and the Node `worker_threads` test harness.
    "worker-engine": "src/worker/engine.ts",
  },
  format: "esm",
  platform: "neutral",
  dts: true,
  deps: {
    // Keep the wasm-bindgen glue as a separate file so its `new URL(..., import.meta.url)`
    // keeps pointing at the .wasm next to it. It is copied into dist/wasm below.
    // `three` is the adapter's optional peer: imported, never inlined.
    neverBundle: [/fast_mesh_wasm\.js$/, /^node:/, /^three($|\/)/],
  },
  copy: [{ from: "src/wasm", to: "dist" }],
});
