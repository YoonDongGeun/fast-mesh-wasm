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
  plugins: [react(), crossOriginIsolation()],
  // Top-level await in src/main.tsx needs a modern target, in the page and in
  // the parser Worker. three.js and React Three Fiber make one large chunk;
  // that is expected here.
  build: { target: "es2022", chunkSizeWarningLimit: 1600 },
  worker: { format: "es" },
});
