// Checks the GitHub Pages build of the showcase (examples/showcase) in
// headless Chrome: the dist built with SHOWCASE_BASE=/fast-mesh-wasm/, served
// under that subpath by a plain static server that sends NO COOP/COEP headers,
// as GitHub Pages does. Cross-origin isolation then comes from the service
// worker (examples/showcase/public/coi-sw.js), registered by index.html.
//
//   node scripts/showcase-pages-check.mjs               # build, then every check
//   node scripts/showcase-pages-check.mjs --no-build    # reuse examples/showcase/dist (built with the base)
//   node scripts/showcase-pages-check.mjs --swiftshader # software WebGL (the default on Linux and under CI)
//   node scripts/showcase-pages-check.mjs --verbose     # echo the page's console
//
// Checks (each prints PASS or FAIL; the exit code is 1 on any FAIL):
// - sw: a fresh profile loads the page, the service worker installs, the page
//   reloads once and is then cross-origin isolated, with the Worker scene
//   threaded (threads > 1); two navigations, no more.
// - scene: the demo meshes load and are drawn, and A and B are coloured
//   against each other (the contact heatmap, with readouts).
// - stable: a reload in the same profile stays isolated and threaded with no
//   extra reload (one navigation).
// - blocked: with the service worker script unavailable (registration
//   fails), the page loads once, is not isolated, and the scene runs
//   single-threaded with the demo meshes and the heatmap.
// - guard: with a service worker that adds no headers (controlled but still
//   not isolated), the page reloads once and stops there: two navigations,
//   single-threaded, the scene works.
// - console: no uncaught exception, in the page or its Workers.
import { spawn } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChrome } from "./chrome-cdp.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const example = join(root, "examples", "showcase");
const dist = join(example, "dist");
const vite = join(example, "node_modules", "vite", "bin", "vite.js");
const BASE = "/fast-mesh-wasm/";

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const verbose = flag("--verbose");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
function verdict(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
  if (!ok) {
    failures++;
  }
}

if (!flag("--no-build")) {
  await new Promise((done, fail) => {
    const child = spawn(process.execPath, [vite, "build"], {
      cwd: example,
      stdio: "inherit",
      env: { ...process.env, SHOWCASE_BASE: BASE },
    });
    child.on("exit", (code) => (code === 0 ? done() : fail(new Error(`vite build exited ${code}`))));
  });
}

// ---- A plain static server under BASE, without isolation headers ----

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".css": "text/css",
  ".json": "application/json",
};
/** How the server answers for `coi-sw.js`: the real file, 404, or a worker that adds no headers. */
let swMode = "real";
let navigations = 0;
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (!path.startsWith(BASE)) {
    res.writeHead(404).end("not found");
    return;
  }
  const rel = path.slice(BASE.length);
  if (rel === "coi-sw.js" && swMode !== "real") {
    if (swMode === "missing") {
      res.writeHead(404).end("not found");
    } else {
      res.writeHead(200, { "Content-Type": "text/javascript" });
      res.end(`self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => e.respondWith(fetch(e.request)));`);
    }
    return;
  }
  let file = normalize(join(dist, rel));
  if (!file.startsWith(resolve(dist))) {
    res.writeHead(403).end();
    return;
  }
  if (existsSync(file) && statSync(file).isDirectory()) {
    file = join(file, "index.html");
  }
  if (!existsSync(file)) {
    res.writeHead(404).end("not found");
    return;
  }
  if (file.endsWith("index.html")) {
    navigations++;
  }
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream", "Cache-Control": "no-cache" });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}${BASE}`;

const swiftshader = flag("--swiftshader") || process.platform === "linux" || process.env.CI !== undefined;
const chromeArgs = ["--window-size=1400,900", ...(swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [])];

const A = "A · blobby sphere";
const B = "B · rounded box";
const q = JSON.stringify;

/** Polls `expr` in the page until it is truthy; returns its value, or `undefined` after `ms`. */
async function waitFor(browser, expr, ms = 20_000) {
  const t = Date.now();
  for (;;) {
    const v = await browser.evaluate(expr).catch(() => undefined);
    if (v) {
      return v;
    }
    if (Date.now() - t > ms) {
      return undefined;
    }
    await sleep(100);
  }
}

/** Waits until the page has stopped navigating (no new index.html for `quiet` ms) and the app is up. */
async function settle(browser, quiet = 3000) {
  let seen = -1;
  const t = Date.now();
  while (seen !== navigations && Date.now() - t < 30_000) {
    seen = navigations;
    await sleep(quiet);
  }
  return waitFor(browser, "window.__showcase !== undefined && document.readyState === 'complete'", 30_000);
}

/** The page's isolation and thread count, then the demo meshes and the A-B heatmap. */
async function scene(browser) {
  const info = await browser.evaluate(`({
    isolated: crossOriginIsolated,
    controlled: navigator.serviceWorker?.controller != null,
    mode: __showcase.mode,
    threads: __showcase.threads(),
  })`);
  const demo = await waitFor(
    browser,
    "(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn) && i; })()",
    60_000,
  );
  const heat = await waitFor(
    browser,
    `(() => { const c = __showcase.collision();
      return c.coloring[${q(A)}]?.to === ${q(B)} && c.coloring[${q(A)}].status === "ready"
        && c.coloring[${q(B)}]?.to === ${q(A)} && c.coloring[${q(B)}].status === "ready"
        && c.settled && c.readouts?.penetration?.depth > 0 && c; })()`,
    60_000,
  );
  const colors = heat === undefined ? null : await browser.evaluate(`__showcase.colorCounts(${q(A)})`);
  const ok = demo !== undefined && heat !== undefined && colors?.red > 0 && colors?.gap > 0;
  const detail = `meshes ${demo === undefined ? "not drawn" : demo.length}, heatmap ${
    heat === undefined ? "not ready" : `penetration ${heat.readouts.penetration.depth.toFixed(3)}, A's vertices ${q(colors)}`
  }`;
  return { ...info, ok, detail };
}

/** One fresh Chrome profile per scenario: no service worker left from another. */
async function session(mode, body) {
  swMode = mode;
  navigations = 0;
  const browser = await launchChrome({ verbose, args: chromeArgs });
  try {
    await body(browser);
  } catch (e) {
    console.error(e);
    console.error(browser.consoleLines.join("\n"));
    failures++;
  } finally {
    const problems = browser.consoleLines.filter((l) => /^(page|worker): exception: /.test(l));
    for (const p of problems) {
      console.log(`  ${p}`);
    }
    verdict(`console (${mode})`, problems.length === 0, `${browser.consoleLines.length} console line(s), ${problems.length} exception(s)`);
    await browser.close();
  }
}

console.log(`serving ${dist} at ${url} without COOP/COEP`);
try {
  await session("real", async (browser) => {
    await browser.goto(url, "document.readyState !== 'loading'");
    await settle(browser);
    const s = await scene(browser);
    verdict(
      "sw",
      s.isolated && s.controlled && s.mode === "worker" && s.threads > 1 && navigations === 2,
      `crossOriginIsolated ${s.isolated}, controlled ${s.controlled}, ${s.mode} scene, threads ${s.threads}, ${navigations} navigation(s)`,
    );
    verdict("scene", s.ok, s.detail);

    navigations = 0;
    await browser.goto(url, "document.readyState !== 'loading'");
    await settle(browser);
    const again = await scene(browser);
    verdict(
      "stable",
      again.isolated && again.threads > 1 && again.ok && navigations === 1,
      `crossOriginIsolated ${again.isolated}, threads ${again.threads}, ${navigations} navigation(s); ${again.detail}`,
    );
  });

  await session("missing", async (browser) => {
    await browser.goto(url, "document.readyState !== 'loading'");
    await settle(browser);
    const s = await scene(browser);
    verdict(
      "blocked",
      !s.isolated && !s.controlled && s.mode === "worker" && s.threads === 1 && s.ok && navigations === 1,
      `crossOriginIsolated ${s.isolated}, controlled ${s.controlled}, threads ${s.threads}, ${navigations} navigation(s); ${s.detail}`,
    );
  });

  await session("inert", async (browser) => {
    await browser.goto(url, "document.readyState !== 'loading'");
    await settle(browser, 4000);
    const s = await scene(browser);
    verdict(
      "guard",
      !s.isolated && s.controlled && s.threads === 1 && s.ok && navigations === 2,
      `crossOriginIsolated ${s.isolated}, controlled ${s.controlled}, threads ${s.threads}, ${navigations} navigation(s); ${s.detail}`,
    );
  });
} finally {
  await new Promise((r) => server.close(r));
}
if (failures > 0) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
