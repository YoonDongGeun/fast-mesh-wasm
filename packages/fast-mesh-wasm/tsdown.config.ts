import { defineConfig } from "tsdown";

export default defineConfig({
  entry: { web: "src/web.ts", node: "src/node.ts" },
  format: "esm",
  platform: "neutral",
  dts: true,
  deps: {
    // Keep the wasm-bindgen glue as a separate file so its `new URL(..., import.meta.url)`
    // keeps pointing at the .wasm next to it. It is copied into dist/wasm below.
    neverBundle: [/fast_mesh_wasm\.js$/, /^node:/],
  },
  copy: [{ from: "src/wasm", to: "dist" }],
});
