import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Top-level await in src/main.tsx needs a modern target.
  // three.js and React Three Fiber make one large chunk; that is expected here.
  build: { target: "es2022", chunkSizeWarningLimit: 1600 },
});
