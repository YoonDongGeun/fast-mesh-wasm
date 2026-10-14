import { defineConfig } from "vite";

export default defineConfig({
  // Top-level await in src/main.ts needs a modern target.
  build: { target: "es2022" },
});
