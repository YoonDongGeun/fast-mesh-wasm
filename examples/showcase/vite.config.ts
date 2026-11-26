import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

/**
 * Cross-origin isolation (COOP/COEP): with these headers the Worker scene runs
 * its multithreaded engine; without them it runs single-threaded. Set by a
 * middleware on every response of the dev and preview servers, not with
 * `server.headers`/`preview.headers`: Vite leaves those off its 304 (Not
 * Modified) responses, and Safari (WebKit) then refuses to start a Worker
 * whose script it revalidated. (The same plugin as examples/react.)
 */
function crossOriginIsolation(): Plugin {
  const isolate = (_req: unknown, res: { setHeader(name: string, value: string): void }, next: () => void) => {
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    next();
  };
  return {
    name: "cross-origin-isolation",
    configureServer: (server) => void server.middlewares.use(isolate),
    configurePreviewServer: (server) => void server.middlewares.use(isolate),
  };
}

export default defineConfig({
  // The public path: `/` by default; the GitHub Pages build sets
  // SHOWCASE_BASE=/fast-mesh-wasm/. Every asset URL (the Workers, the wasm
  // files the library finds through `import.meta.url`, public/coi-sw.js) is
  // resolved against it.
  base: process.env.SHOWCASE_BASE ?? "/",
  plugins: [react(), crossOriginIsolation()],
  // A modern target: browsers with wasm threads take it. three.js is one stable vendor
  // chunk (the page and the feature chunks share it). Everything else the page
  // imports statically, React Three Fiber included, is the one index chunk, so
  // what it shares with the feature chunks (src/app/ux/feature-chunks.ts) is not
  // split into many small ones. A build warns when index passes 1,000 kB.
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: "three", test: /node_modules[\\/]three[\\/]/, priority: 2 },
            { name: "index", tags: ["$initial"], priority: 1 },
          ],
        },
      },
    },
  },
  worker: { format: "es" },
});
