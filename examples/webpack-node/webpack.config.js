// Bundles server-side code the way Next.js and other webpack-based frameworks do.
// Production mode turns on `sideEffects`-based module pruning, which is what this
// example guards against: the Node entry must still run and load the .wasm file.
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export default {
  mode: "production",
  target: "node22",
  entry: "./src/main.js",
  output: {
    path: resolve(here, "dist"),
    filename: "main.cjs",
    clean: true,
  },
};
