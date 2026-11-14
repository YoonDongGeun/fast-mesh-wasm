// Headless Chrome benchmark & regression check for Issue #198:
// Verifies 1M-triangle STL imports decimate off the main thread with smooth
// `"decimating"` progress, fast wall-clock completion, responsive main thread
// in both default Worker mode (`/`) and sync mode (`/?local`), and clean
// mid-decimation cancellation.
//
//   node scripts/load-decimate-bench.mjs
//   node scripts/load-decimate-bench.mjs --verbose
//   node scripts/load-decimate-bench.mjs --swiftshader
import { createServer } from "node:net";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChrome, startServer } from "./chrome-cdp.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const example = join(root, "examples", "showcase");
const exampleVite = join(example, "node_modules", "vite", "bin", "vite.js");
const vite = existsSync(exampleVite) ? exampleVite : join(root, "node_modules", "vite", "bin", "vite.js");

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const verbose = flag("--verbose");
const swiftshader = flag("--swiftshader") || process.platform === "linux" || process.env.CI !== undefined;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));

let failures = 0;
function verdict(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
  if (!ok) {
    failures++;
  }
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 4195;
      srv.close(() => resolve(port));
    });
  });
}

/** Writes a 1,000,000-triangle closed UV sphere binary STL (`1000 x 501` grid). */
function writeSphereStl(outPath, segments = 1000, rings = 501) {
  const count = 2 * segments * (rings - 1);
  const buf = Buffer.alloc(84 + 50 * count);
  buf.write("fast-mesh load-decimate-bench 1M sphere", 0, "latin1");
  buf.writeUInt32LE(count, 80);

  const verts = new Float32Array((rings - 1) * segments * 3 + 6);
  verts[0] = 0;
  verts[1] = 2;
  verts[2] = 0;
  for (let i = 1; i < rings; i++) {
    const t = (Math.PI * i) / rings;
    const st = Math.sin(t);
    const ct = Math.cos(t);
    for (let j = 0; j < segments; j++) {
      const f = (2 * Math.PI * j) / segments;
      const base = (1 + (i - 1) * segments + j) * 3;
      verts[base] = 2 * st * Math.cos(f);
      verts[base + 1] = 2 * ct;
      verts[base + 2] = 2 * st * Math.sin(f);
    }
  }
  const south = 1 + (rings - 1) * segments;
  verts[south * 3] = 0;
  verts[south * 3 + 1] = -2;
  verts[south * 3 + 2] = 0;

  let at = 84;
  const ringIdx = (i, j) => 1 + (i - 1) * segments + (j % segments);
  const tri = (ia, ib, ic) => {
    at += 12;
    for (const idx of [ia, ib, ic]) {
      const o = idx * 3;
      buf.writeFloatLE(verts[o], at);
      buf.writeFloatLE(verts[o + 1], at + 4);
      buf.writeFloatLE(verts[o + 2], at + 8);
      at += 12;
    }
    at += 2;
  };
  for (let j = 0; j < segments; j++) {
    tri(0, ringIdx(1, j + 1), ringIdx(1, j));
    tri(south, ringIdx(rings - 1, j), ringIdx(rings - 1, j + 1));
  }
  for (let i = 1; i < rings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      tri(ringIdx(i, j), ringIdx(i, j + 1), ringIdx(i + 1, j + 1));
      tri(ringIdx(i, j), ringIdx(i + 1, j + 1), ringIdx(i + 1, j));
    }
  }
  writeFileSync(outPath, buf);
  return count;
}

/** Writes a 1,000,000-triangle irregular terrain scan binary STL (`1001 x 501` grid). */
function writeTerrainStl(outPath, nx = 1001, ny = 501) {
  const count = 2 * (nx - 1) * (ny - 1);
  const buf = Buffer.alloc(84 + 50 * count);
  buf.write("fast-mesh load-decimate-bench 1M terrain", 0, "latin1");
  buf.writeUInt32LE(count, 80);

  const verts = new Float32Array(nx * ny * 3);
  for (let iy = 0; iy < ny; iy++) {
    const v = iy / (ny - 1);
    for (let ix = 0; ix < nx; ix++) {
      const u = ix / (nx - 1);
      const wx = (u - 0.5) * 100 + 0.03 * Math.sin(v * 23.0) + 0.015 * Math.cos(u * 31.0);
      const wy = (v - 0.5) * 100 + 0.03 * Math.cos(u * 19.0) + 0.015 * Math.sin(v * 29.0);
      const wz =
        4.0 * Math.sin(wx * 0.17) * Math.cos(wy * 0.13) +
        1.5 * Math.sin((wx + wy) * 0.43) +
        0.35 * Math.cos(wx * 1.1 - wy * 0.8);
      const o = (iy * nx + ix) * 3;
      verts[o] = wx;
      verts[o + 1] = wy;
      verts[o + 2] = wz;
    }
  }

  let at = 84;
  const tri = (ia, ib, ic) => {
    at += 12;
    for (const idx of [ia, ib, ic]) {
      const o = idx * 3;
      buf.writeFloatLE(verts[o], at);
      buf.writeFloatLE(verts[o + 1], at + 4);
      buf.writeFloatLE(verts[o + 2], at + 8);
      at += 12;
    }
    at += 2;
  };
  for (let iy = 0; iy < ny - 1; iy++) {
    for (let ix = 0; ix < nx - 1; ix++) {
      const p00 = iy * nx + ix;
      const p10 = p00 + 1;
      const p01 = p00 + nx;
      const p11 = p01 + 1;
      tri(p00, p10, p11);
      tri(p00, p11, p01);
    }
  }
  writeFileSync(outPath, buf);
  return count;
}

const fixtures = mkdtempSync(join(tmpdir(), "fast-mesh-load-bench-"));
const port = await getFreePort();
const baseUrl = `http://127.0.0.1:${port}`;
const server = await startServer(
  process.execPath,
  [vite, "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"],
  { cwd: example, url: `${baseUrl}/` },
);
const browser = await launchChrome({
  verbose,
  args: ["--window-size=1400,900", ...(swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [])],
});

const page = (expr) => browser.evaluate(expr);

async function waitFor(expr, ms = 60_000) {
  const t = Date.now();
  for (;;) {
    const v = await page(expr).catch(() => undefined);
    if (v) {
      return v;
    }
    if (Date.now() - t > ms) {
      return undefined;
    }
    await sleep(25);
  }
}

async function upload(paths) {
  const { root: doc } = await browser.send("DOM.getDocument", { depth: 0 });
  const { nodeId } = await browser.send("DOM.querySelector", { nodeId: doc.nodeId, selector: "#file-input" });
  await browser.send("DOM.setFileInputFiles", { nodeId, files: paths });
}

async function warmUpShaders() {
  // Compile Three.js / FastMesh unselected-mesh WebGLProgram once in the Chrome
  // GPU process before timed loads (Windows ANGLE D3D11 FXC compiles on first draw).
  await browser.goto(`${baseUrl}/?decimate-target=100000`, "window.__showcase !== undefined");
  await waitFor(
    '__showcase.items().length > 0 && __showcase.items().every((i) => i.drawn) && __showcase.jobs().every((j) => j.phase === "done")',
    30_000,
  );
  await page('document.querySelector(\'[data-action="demo-meshes"]\')?.click()');
  await waitFor(
    '__showcase.items().length >= 4 && __showcase.items().every((i) => i.drawn) && __showcase.jobs().every((j) => j.phase === "done")',
    30_000,
  );
  await sleep(150);
}

async function prepareCleanPage(query) {
  await browser.goto(`${baseUrl}/${query}`, "window.__showcase !== undefined");
  await waitFor(
    '__showcase.items().length > 0 && __showcase.items().every((i) => i.drawn) && __showcase.jobs().every((j) => j.phase === "done") && __showcase.selection() !== null && document.querySelector(".meshes li button.danger") !== null',
    30_000,
  );
  await sleep(250);
  await page('document.querySelector(".meshes li button.danger")?.click()');
  await waitFor("__showcase.items().length === 0", 10_000);
  await sleep(250);
}

async function runBenchCase(label, query, filePath, filename) {
  await prepareCleanPage(query);
  const decimateEnabled = await page("__showcase.bigFile().decimateOnLoad === true");
  await page(`(() => {
    const tracker = {
      start: performance.now(),
      tasks: [],
      maxRafDelta: 0,
      lastRaf: performance.now(),
      on: true,
      phases: [],
      phaseTimes: [],
      decimatingProgress: [],
    };
    const obs = new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        if (e.startTime >= tracker.start) {
          tracker.tasks.push({ start: e.startTime, duration: e.duration });
        }
      }
    });
    obs.observe({ type: "longtask" });
    const raf = (t) => {
      tracker.maxRafDelta = Math.max(tracker.maxRafDelta, t - tracker.lastRaf);
      tracker.lastRaf = t;
      if (tracker.on) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
    const sample = () => {
      const job = __showcase.jobs().find((j) => j.name === ${JSON.stringify(filename)});
      if (job) {
        if (tracker.phases.at(-1) !== job.phase) {
          tracker.phases.push(job.phase);
          tracker.phaseTimes.push(job.phase + "@" + Math.round(performance.now() - tracker.start));
        }
        if (job.phase === "decimating" && typeof job.progress === "number") {
          if (tracker.decimatingProgress.at(-1) !== job.progress) {
            tracker.decimatingProgress.push(job.progress);
          }
        }
      }
    };
    const mutObs = new MutationObserver(sample);
    mutObs.observe(document.body, { childList: true, subtree: true, attributes: true });
    const timer = setInterval(sample, 10);
    window.__bench = { tracker, obs, mutObs, timer, sample };
    return true;
  })()`);

  await sleep(250);
  const t0 = performance.now();
  await upload([filePath]);
  const done = await waitFor(
    `(() => {
      window.__bench.sample();
      const item = __showcase.items().find((i) => i.name === ${JSON.stringify(filename)});
      const jobDone = __showcase.jobs().every((j) => j.name !== ${JSON.stringify(filename)} || j.phase === "done");
      return item !== undefined && item.drawn && jobDone ? { triangles: item.triangles } : false;
    })()`,
    60_000,
  );
  const wallMs = performance.now() - t0;
  await sleep(100);
  const stats = await page(`(() => {
    const b = window.__bench;
    b.sample();
    b.tracker.on = false;
    b.obs.disconnect();
    b.mutObs.disconnect();
    clearInterval(b.timer);
    return {
      tasks: b.tracker.tasks.map((t) => ({ at: Math.round(t.start - b.tracker.start), duration: Math.round(t.duration) })),
      maxRafDelta: b.tracker.maxRafDelta,
      phases: b.tracker.phases,
      phaseTimes: b.tracker.phaseTimes,
      decimatingProgress: b.tracker.decimatingProgress,
      summary: __showcase.jobs().find((j) => j.name === ${JSON.stringify(filename)})?.summary ?? null,
    };
  })()`);

  const maxTask = stats.tasks.reduce((m, t) => Math.max(m, t.duration), 0);
  const visitedDecimating = stats.phases.includes("decimating");
  const monotonic =
    stats.decimatingProgress.length >= 2 &&
    stats.decimatingProgress.every((p, i) => i === 0 || p > stats.decimatingProgress[i - 1]);
  const trisOk = done !== undefined && done.triangles > 0 && done.triangles <= 100_000;
  const mainThreadOk = maxTask < 200;

  verdict(
    label,
    decimateEnabled && trisOk && visitedDecimating && monotonic && mainThreadOk,
    `tris=${done?.triangles ?? "none"}, wall=${fmt(wallMs)} ms, maxLongTask=${fmt(maxTask)} ms (${stats.tasks.length} >=50ms${verbose && stats.tasks.length ? ` ${JSON.stringify(stats.tasks)} phases=${JSON.stringify(stats.phaseTimes)}` : ""}), maxRafGap=${fmt(stats.maxRafDelta)} ms, phases=[${stats.phases.join("->")}], decimateSteps=[${stats.decimatingProgress.map((p) => p.toFixed(2)).join(",")}]${stats.summary ? `, summary=${JSON.stringify(stats.summary)}` : ""}`,
  );
}

async function runCancelCase(filePath, filename) {
  await prepareCleanPage("?decimate-target=100000");
  await upload([filePath]);
  const sawDecimating = await waitFor(
    `(() => {
      const btn = document.querySelector('li[data-phase="decimating"] [data-action="cancel-load"]');
      if (!btn) return false;
      btn.click();
      return true;
    })()`,
    30_000,
  );
  const settled = await waitFor(
    `__showcase.jobs().every((j) => j.name !== ${JSON.stringify(filename)} || j.phase === "error")`,
    15_000,
  );
  await sleep(800);
  const after = await page(`({
    items: __showcase.items().length,
    jobs: __showcase.jobs(),
  })`);
  const noOrphans = after.items === 0 && after.jobs.every((j) => j.phase === "error" || j.phase === "done");
  const hasCancelledMsg = after.jobs.some((j) => j.name === filename && j.phase === "error" && j.message === "cancelled");
  verdict(
    "cancel-mid-decimating",
    sawDecimating === true && settled === true && noOrphans && hasCancelledMsg,
    `clickedDuringDecimating=${sawDecimating}, itemsAfter=${after.items}, jobs=${JSON.stringify(after.jobs)}`,
  );
}

try {
  const spherePath = join(fixtures, "sphere-1m.stl");
  const terrainPath = join(fixtures, "terrain-1m.stl");
  const sphereTris = writeSphereStl(spherePath);
  const terrainTris = writeTerrainStl(terrainPath);
  console.log(`Generated fixtures: sphere=${sphereTris.toLocaleString("en")} tris, terrain=${terrainTris.toLocaleString("en")} tris`);

  await warmUpShaders();
  await runBenchCase("worker-sphere-1m", "?decimate-target=100000", spherePath, "sphere-1m.stl");
  await runBenchCase("worker-terrain-1m", "?decimate-target=100000", terrainPath, "terrain-1m.stl");
  await runBenchCase("local-sphere-1m", "?local&decimate-target=100000", spherePath, "sphere-1m.stl");
  await runBenchCase("local-terrain-1m", "?local&decimate-target=100000", terrainPath, "terrain-1m.stl");
  await runCancelCase(spherePath, "sphere-1m.stl");

  const consoleErrors = browser.consoleLines.filter((l) => l.includes(": error:") || l.includes(": exception:"));
  verdict("console-clean", consoleErrors.length === 0, `${consoleErrors.length} error(s)`);
} finally {
  await browser.close();
  server.stop();
  rmSync(fixtures, { recursive: true, force: true });
}

if (failures > 0) {
  process.exitCode = 1;
}
