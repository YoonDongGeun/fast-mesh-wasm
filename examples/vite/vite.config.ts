import { defineConfig } from "vite";

export default defineConfig({
  // Top-level await and `using` in src/main.ts need a modern target.
  build: { target: "es2022" },
});
