// The contact heatmap under a gizmo drag and a sculpt stroke, in the showcase
// on a real page: per-update time (between colour deliveries) and frame time
// (between animation frames), on demo pairs of each size. Issue #173.
//
//   pnpm build && pnpm --filter fast-mesh-wasm-example-showcase build
//   node scripts/showcase-heatmap-bench.mjs [--sizes 100000,1000000] [--seconds 4] [--runs 3]
//     [--fast] [--no-sculpt] [--no-drag] [--threads N] [--json out.json] [--port 4273] [--swiftshader] [--verbose]
//     [--shots dir]
//
// `--fast` ticks Fast mode first (the default settings otherwise). The numbers
// are medians (and p95) over the runs' samples. A page runs on the machine's
// GPU unless `--swiftshader`; the WebGL renderer is printed. After each case the
// colours at rest are compared with a new field of the pair, computed from scratch,
// and the script fails when they differ. `--shots` saves a screenshot mid-gesture.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChrome, startServer } from "./chrome-cdp.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const example = join(root, "examples", "showcase");
const vite = join(example, "node_modules", "vite", "bin", "vite.js");

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i === -1 ? fallback : argv[i + 1];
};
const sizes = option("--sizes", "100000,1000000").split(",").map(Number);
const seconds = Number(option("--seconds", 4));
const runs = Number(option("--runs", 3));
const port = Number(option("--port", 4273));
const threads = option("--threads", undefined);
const jsonPath = option("--json", undefined);
const bound = option("--bound", undefined);
const shots = option("--shots", undefined);

const A = "A · blobby sphere";
const B = "B · rounded box";
const q = JSON.stringify;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x) => (x === null || x === undefined ? "-" : x >= 100 ? x.toFixed(0) : x.toFixed(1));

function stats(samples) {
  const s = Float64Array.from(samples).sort();
  const at = (p) => (s.length === 0 ? null : s[Math.min(s.length - 1, Math.floor(p * s.length))]);
  return { n: s.length, median: at(0.5), p95: at(0.95), max: at(1) };
}

const server = await startServer(process.execPath, [vite, "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"], {
  cwd: example,
  url: `http://127.0.0.1:${port}/`,
});
const browser = await launchChrome({
  verbose: flag("--verbose"),
  args: [
    "--window-size=1400,900",
    ...(flag("--swiftshader") ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : ["--use-angle=d3d11", "--ignore-gpu-blocklist"]),
  ],
});
const page = (expr) => browser.evaluate(expr);
const hook = (expr) => page(`window.__showcase.${expr}`);
const mouse = (type, x, y, extra = {}) =>
  browser.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });
const clickOn = (selector) => page(`(() => { const el = document.querySelector(${q(selector)}); el?.click(); return el !== null; })()`);

async function waitFor(expr, ms = 120_000) {
  const t = Date.now();
  for (;;) {
    const v = await page(expr).catch(() => undefined);
    if (v) {
      return v;
    }
    if (Date.now() - t > ms) {
      throw new Error(`timed out: ${expr}\n${browser.consoleLines.slice(-20).join("\n")}`);
    }
    await sleep(50);
  }
}

/** Starts recording in the page: animation frames, colour deliveries, long tasks. */
const startProbe = () =>
  page(`(() => {
    const p = { frames: [], updates: [], vertices: [], long: [], on: true, off: null };
    const tick = (t) => { if (p.on) { p.frames.push(t); requestAnimationFrame(tick); } };
    requestAnimationFrame(tick);
    p.off = __showcase.store.scene.onChanges((changes) => {
      let n = 0;
      for (const c of changes) {
        if (c.colors === "all") { n = -1; break; }
        for (const r of c.colors) { n += r[1]; }
      }
      if (n !== 0) { p.updates.push(performance.now()); p.vertices.push(n); }
    });
    p.observer = new PerformanceObserver((l) => { for (const e of l.getEntries()) p.long.push(e.duration); });
    p.observer.observe({ type: "longtask" });
    window.__heat = p;
    return true;
  })()`);

/** Stops recording: the samples inside `[from, to]` (page clock). */
const readProbe = (from, to) =>
  page(`(() => {
    const p = window.__heat;
    p.on = false; p.off(); p.observer.disconnect();
    const inside = (t) => t >= ${from} && t <= ${to};
    const gaps = (a) => a.slice(1).map((t, i) => [a[i], t - a[i]]).filter(([t]) => inside(t)).map(([, d]) => d);
    return {
      frames: gaps(p.frames),
      updates: gaps(p.updates),
      vertices: p.vertices.filter((_, i) => inside(p.updates[i])),
      longTasks: p.long,
      lastUpdate: p.updates.at(-1) ?? null,
    };
  })()`);

const nowInPage = () => page("performance.now()");

/**
 * Waits for the colours to come to rest: a query answered after every update
 * still queued (a stepped recompute delivers nothing until it is done), and no
 * delivery for `quiet` ms after it.
 */
async function settled(quiet = 500, ms = 120_000) {
  const t = Date.now();
  for (;;) {
    await page(`(async () => {
      const s = __showcase.store;
      for (const [m, st] of s.getSnapshot().meshes) {
        if (st.display.coloring !== null) { await s.fetchColoringValues(m).catch(() => null); }
      }
      return true;
    })()`);
    const state = await page(`({ status: __showcase.status().status, last: window.__heat.updates.at(-1) ?? 0, now: performance.now() })`);
    if (state.status === "idle" && state.now - state.last > quiet) {
      return;
    }
    if (Date.now() - t > ms) {
      throw new Error("the colours never settled");
    }
    await sleep(100);
  }
}

/** The vertices of each coloured mesh whose settled value differs from a new field's, computed from scratch. */
const inexact = () =>
  page(`(async () => {
    const s = __showcase.store;
    const scene = s.scene;
    let wrong = 0;
    let n = 0;
    for (const [m, st] of s.getSnapshot().meshes) {
      const c = st.display.coloring;
      if (c === null) { continue; }
      const shown = await s.fetchColoringValues(m);
      const field = scene.addDistanceField({ name: "bench:" + (n++), on: m, to: c.distanceTo, maxDistance: c.maxDistance, sign: c.sign });
      const exact = await scene.fetchFieldValues(field);
      for (let v = 0; v < exact.length; v++) {
        if (!Object.is(exact[v], shown[v])) { wrong++; }
      }
      scene.removeField(field);
    }
    return wrong;
  })()`);

async function shot(name) {
  if (shots !== undefined) {
    mkdirSync(shots, { recursive: true });
    const { data } = await browser.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(join(shots, `${name}.png`), Buffer.from(data, "base64"));
  }
}

/** Presses at `from` and moves the pointer on a small loop for `seconds`; returns when it stopped (page clock). */
async function gesture(from, radius, name) {
  const t0 = Date.now();
  let shown = false;
  await mouse("mousePressed", from.x, from.y, { buttons: 1 });
  while (Date.now() - t0 < seconds * 1000) {
    const a = ((Date.now() - t0) / 1000) * 2 * Math.PI * 0.5;
    await mouse("mouseMoved", from.x + radius * Math.sin(a), from.y + radius * (1 - Math.cos(a)) * 0.5, { buttons: 1 });
    await sleep(16);
    if (!shown && Date.now() - t0 > (seconds * 1000) / 2) {
      shown = true;
      await shot(`${name}-moving`);
    }
  }
  const end = await nowInPage();
  await mouse("mouseReleased", from.x, from.y, { buttons: 0 });
  return end;
}

async function dragRun() {
  const from = await hook(`screenPoint(${q(A)})`);
  await waitFor(`__showcase.gizmo()?.mesh === ${q(A)}`, 10_000);
  for (let axis = null, t = Date.now(); axis === null && Date.now() - t < 5000; ) {
    await mouse("mouseMoved", from.x, from.y, { button: "none" });
    axis = await hook("gizmo()?.axis ?? null");
    await sleep(50);
  }
  await startProbe();
  const start = await nowInPage();
  const end = await gesture(from, 60, "drag");
  await settled();
  await shot("drag-settled");
  const r = await readProbe(start + 200, end);
  return { ...r, settle: r.lastUpdate === null ? null : r.lastUpdate - end };
}

async function strokeRun() {
  const from = await hook(`screenPoint(${q(A)})`);
  await mouse("mouseMoved", from.x, from.y, { button: "none" });
  await sleep(200);
  const dabs0 = (await hook("sculpt()")).dabs;
  await startProbe();
  const start = await nowInPage();
  const end = await gesture(from, 60, "stroke");
  const dabs = (await hook("sculpt()")).dabs - dabs0;
  await settled();
  await shot("stroke-settled");
  const r = await readProbe(start + 200, end);
  return { ...r, settle: r.lastUpdate === null ? null : r.lastUpdate - end, dabsPerSecond: dabs / ((end - start) / 1000) };
}

function summarize(name, size, samples, extra = {}) {
  const pool = (k) => samples.flatMap((s) => s[k]);
  return {
    case: name,
    size,
    frames: stats(pool("frames")),
    updates: stats(pool("updates")),
    settle: stats(samples.map((s) => s.settle ?? 0)),
    longTasks: stats(pool("longTasks")),
    vertices: stats(pool("vertices")),
    ...extra,
  };
}

const results = [];
try {
  let info;
  for (const size of sizes) {
    const search = new URLSearchParams({ big: String(size), decimate: "0" });
    if (threads !== undefined) {
      search.set("threads", threads);
    }
    await browser.goto(`http://127.0.0.1:${port}/?${search}`, "window.__showcase !== undefined");
    info ??= await page(`({
      threads: __showcase.threads(),
      cores: navigator.hardwareConcurrency,
      gl: (() => {
        const gl = document.createElement("canvas").getContext("webgl2");
        const ext = gl?.getExtension("WEBGL_debug_renderer_info");
        return gl === null ? "no WebGL2" : ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
      })(),
    })`);
    console.log(`${browser.name}; ${info.cores} logical cores; ${info.threads} engine threads; WebGL: ${info.gl}`);
    const boot = await waitFor(`__showcase.items().find((m) => m.name.includes("headsq") && m.drawn)?.name`);
    await clickOn("[data-action=demo-meshes]");
    await waitFor(`(() => { const want = [${q(A)}, ${q(B)}];
      const got = want.map((n) => __showcase.items().find((m) => m.name === n)).filter((m) => m !== undefined && m.drawn);
      return got.length === 2 && __showcase.status().status === "idle"; })()`);
    const items = await hook("items()");
    const triangles = Object.fromEntries(items.map((i) => [i.name, i.triangles]));
    await clickOn(`[data-mesh=${q(A)}] .mesh-name`);
    await clickOn(`[data-mesh=${q(boot)}] button.danger`);
    await waitFor(`!__showcase.items().some((m) => m.name === ${q(boot)})`);
    await clickOn("[data-action=view-reset]");
    await sleep(500);
    if (flag("--fast")) {
      await clickOn("[data-control=fast]");
    }
    for (const [side, name] of [
      ["a", A],
      ["b", B],
    ]) {
      await page(`(() => {
        const sel = document.querySelector('[data-pick=${side}]');
        const opt = [...sel.options].find((o) => o.text === ${q(name)});
        sel.value = opt.value;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      })()`);
    }
    await waitFor(
      `(() => { const c = __showcase.collision(); return c.coloring[${q(A)}]?.status === "ready" && c.coloring[${q(B)}]?.status === "ready" && c.settled; })()`,
    );
    if (bound !== undefined) {
      await page(`(() => {
        const s = __showcase.store;
        for (const [m, st] of s.getSnapshot().meshes) {
          const c = st.display.coloring;
          if (c !== null) { s.setDisplay(m, { coloring: { ...c, maxDistance: ${Number(bound)} } }); }
        }
        return true;
      })()`);
      await sleep(2000);
      await waitFor(`__showcase.status().status === "idle"`);
    }
    const label = `A ${triangles[A]} + B ${triangles[B]} triangles`;
    const report = async (r) => {
      r.inexact = await inexact();
      results.push(r);
      if (r.inexact !== 0) {
        process.exitCode = 1;
      }
      console.log(
        `${label}, ${r.case}: frame ${fmt(r.frames.median)} ms (p95 ${fmt(r.frames.p95)}, max ${fmt(r.frames.max)}); colour update every ${fmt(r.updates.median)} ms (p95 ${fmt(r.updates.p95)}, ${r.updates.n} updates), ${fmt(r.vertices.median)} vertices each; settles ${fmt(r.settle.median)} ms after release; long tasks ${r.longTasks.n} (max ${fmt(r.longTasks.max)})${r.dabsPerSecond === undefined ? "" : `; ${r.dabsPerSecond.toFixed(1)} dabs/s`}; ${r.inexact} vertices off the exact field at rest`,
      );
    };
    if (!flag("--no-drag")) {
      const samples = [];
      for (let i = 0; i < runs; i++) {
        samples.push(await dragRun());
        await sleep(300);
      }
      await report(summarize("gizmo drag", size, samples));
    }
    if (!flag("--no-sculpt")) {
      await clickOn("[data-action=sculpt-tool-inflate]");
      await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);
      await sleep(300);
      const samples = [];
      for (let i = 0; i < runs; i++) {
        samples.push(await strokeRun());
        await sleep(300);
      }
      await report(
        summarize("sculpt stroke", size, samples, {
          dabsPerSecond: samples.reduce((a, s) => a + s.dabsPerSecond, 0) / samples.length,
        }),
      );
    }
  }
  if (jsonPath !== undefined) {
    writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  }
} finally {
  await browser.close();
  server.stop?.();
}
