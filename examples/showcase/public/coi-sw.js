// Cross-origin isolation for static hosts that cannot set headers (GitHub
// Pages): this service worker re-serves the page's same-origin responses with
// COOP/COEP, so after one reload the page is cross-origin isolated and the
// Worker scene runs its multithreaded engine. Registered by the snippet in
// index.html, which skips it when the server already sends the headers (the
// Vite dev and preview servers). The same idea as the coi-serviceworker
// project, written here without a dependency.
//
// COEP `require-corp` (not `credentialless`): the showcase loads only its own
// files (scripts, Workers, wasm; demo meshes are generated, user files are
// blobs), so nothing needs a CORP header, and `require-corp` is what the dev
// and preview servers send and what every browser with COEP supports.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  // Cross-origin requests pass through untouched (the showcase makes none).
  if (new URL(request.url).origin !== self.location.origin) {
    return;
  }
  // A DevTools quirk: fetch() rejects this combination.
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") {
    return;
  }
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) {
        return response; // opaque: headers cannot be changed
      }
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
