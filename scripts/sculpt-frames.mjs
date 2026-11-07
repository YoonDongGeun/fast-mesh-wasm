// Frame times during a scripted sculpt stroke in the real showcase, in headless
// Chrome (scripts/chrome-cdp.mjs), on the sync scene (`?local`) and the Worker
// scene. A UV sphere STL is uploaded (`?decimate=0`), the tool armed through the
// toolbox, and the pointer dragged along an arc at about 120 events per second,
// with the stroke remeshes off (they stand down past 100k triangles anyway).
//
//   pnpm build && pnpm --filter fast-mesh-wasm-example-showcase build
//   node scripts/sculpt-frames.mjs [--tris 100000,1000000] [--tools inflate,grab]
//                                  [--radius 0.1] [--dist <dir>] [--swiftshader] [--profile]
//   node scripts/sculpt-frames.mjs --verify [--dist <dir>]
//
// Prints, per case: the animation-frame gaps (median, p95, longest), the longest
// main-thread task, the dabs applied, and the WebGL renderer. Numbers go in
// docs/benchmarks.md. Hand feel is not something this measures.
//
// `--verify` checks instead, on 20k triangles with the default stroke remeshes on,
// through the canvas on both scenes: a stroke changes the surface, a stroke takes one
// Ctrl+Z on the sync scene, the live remesh adds triangles during a stroke, a grab drags
// the surface by the pointer's drag, and dabs that touch a painted mask are dropped and
// leave its vertices where they were, and a release before the press's raycast returns
// opens no stroke. It exits 1 on any failure.
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { ISOLATION, launchChrome, serveStatic } from "./chrome-cdp.mjs";

const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const k = argv.indexOf(`--${name}`);
  return k === -1 ? fallback : argv[k + 1];
};
const triangleCounts = opt("tris", "100000,1000000").split(",").map(Number);
const tools = opt("tools", "inflate,grab").split(",");
const radius = Number(opt("radius", "0.1"));
const dist = resolve(opt("dist", "examples/showcase/dist"));
const swiftshader = argv.includes("--swiftshader") || process.env.CI !== undefined;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const EVENTS = 240;
const EVENT_MS = 8;
const NAME = "sculpt-sphere.stl";

/** A UV sphere of radius 2 as a binary STL, `2 * segments * (rings - 1)` triangles. */
function sphereStl(triangles) {
  const rings = Math.max(4, Math.round(Math.sqrt(triangles / 4)));
  const segments = Math.max(4, Math.round(triangles / (2 * (rings - 1))));
  const count = 2 * segments * (rings - 1);
  const buf = Buffer.alloc(84 + 50 * count);
  buf.writeUInt32LE(count, 80);
  const pt = (i, j) => {
    if (i === 0) return [0, 2, 0];
    if (i === rings) return [0, -2, 0];
    const t = (Math.PI * i) / rings;
    const f = (2 * Math.PI * (j % segments)) / segments;
    return [2 * Math.sin(t) * Math.cos(f), 2 * Math.cos(t), 2 * Math.sin(t) * Math.sin(f)];
  };
  let at = 84;
  const tri = (a, b, c) => {
    at += 12;
    for (const p of [a, b, c]) {
      for (const x of p) {
        buf.writeFloatLE(x, at);
        at += 4;
      }
    }
    at += 2;
  };
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      if (i > 0) tri(pt(i, j), pt(i, j + 1), pt(i + 1, j));
      if (i < rings - 1) tri(pt(i, j + 1), pt(i + 1, j + 1), pt(i + 1, j));
    }
  }
  return { buf, count };
}

const percentile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
const ms = (x) => (x === undefined ? "-" : x.toFixed(1));

const server = await serveStatic(dist, ISOLATION);
const browser = await launchChrome({
  args: ["--window-size=1400,900", ...(swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [])],
});
const fixtures = mkdtempSync(join(tmpdir(), "fast-mesh-sculpt-"));
const page = (expr) => browser.evaluate(expr);
async function waitFor(expr, timeout = 180_000) {
  const t = Date.now();
  for (;;) {
    const v = await page(expr).catch(() => undefined);
    if (v) return v;
    if (Date.now() - t > timeout) throw new Error(`timed out: ${expr}\n${browser.consoleLines.slice(-20).join("\n")}`);
    await sleep(100);
  }
}
const mouse = (type, x, y, extra = {}) =>
  browser.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });
const clickOn = (selector) => page(`(document.querySelector(${JSON.stringify(selector)}).click(), true)`);
const setValue = (selector, value) =>
  page(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, ${JSON.stringify(String(value))});
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
const uncheck = (selector) =>
  page(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (el.checked) el.click(); return true; })()`);

/** Drags over the canvas on an arc, recording the frames in the page. */
async function stroke() {
  const { x: cx, y: cy } = await page(`__showcase.screenPoint(${JSON.stringify(NAME)})`);
  const reach = 60;
  const at = (k) => {
    const a = (k / EVENTS) * Math.PI * 1.5;
    return [cx + reach * Math.cos(a), cy + reach * Math.sin(a)];
  };
  await page(`(() => {
    const probe = { gaps: [], longest: 0, last: performance.now(), on: true };
    new PerformanceObserver((l) => { for (const e of l.getEntries()) probe.longest = Math.max(probe.longest, e.duration); }).observe({ type: "longtask" });
    const frame = (t) => { probe.gaps.push(t - probe.last); probe.last = t; if (probe.on) requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
    window.__frames = probe;
    return true;
  })()`);
  const [x0, y0] = at(0);
  await mouse("mouseMoved", x0, y0, { button: "none" });
  await sleep(150);
  const before = await page("window.__showcase.sculpt()");
  if (argv.includes("--profile")) {
    await browser.send("Profiler.enable");
    await browser.send("Profiler.setSamplingInterval", { interval: 500 });
    await browser.send("Profiler.start");
  }
  await mouse("mousePressed", x0, y0, { buttons: 1 });
  await sleep(30);
  const t0 = Date.now();
  for (let k = 1; k <= EVENTS; k++) {
    const [x, y] = at(k);
    await mouse("mouseMoved", x, y, { buttons: 1 });
    const wait = t0 + k * EVENT_MS - Date.now();
    if (wait > 0) await sleep(wait);
  }
  const elapsed = Date.now() - t0;
  const [xe, ye] = at(EVENTS);
  await mouse("mouseReleased", xe, ye, { buttons: 0 });
  if (argv.includes("--profile")) {
    printProfile((await browser.send("Profiler.stop")).profile);
  }
  await sleep(300);
  const probe = await page("(window.__frames.on = false, { gaps: window.__frames.gaps, longest: window.__frames.longest })");
  const after = await page("window.__showcase.sculpt()");
  const gaps = probe.gaps.slice(2).sort((a, b) => a - b);
  return { elapsed, gaps, longest: probe.longest, dabs: after.dabs - before.dabs, strokes: after.strokes - before.strokes };
}

/** The main thread's top functions by self time. */
function printProfile(profile) {
  const self = new Map();
  const byId = new Map(profile.nodes.map((n) => [n.id, n]));
  profile.samples.forEach((id, k) => {
    const n = byId.get(id).callFrame;
    const key = `${n.functionName || "(anon)"} ${n.url.split("/").pop()}:${n.lineNumber}`;
    self.set(key, (self.get(key) ?? 0) + profile.timeDeltas[k] / 1000);
  });
  for (const [key, t] of [...self].sort((a, b) => b[1] - a[1]).slice(0, 18)) {
    console.log(`    ${t.toFixed(0).padStart(7)} ms  ${key}`);
  }
}

/** Loads the sphere in `mode` and frames it, with the boot head gone. */
async function load(mode, path) {
  await browser.goto(`${server.url}/?decimate=0${mode === "sync" ? "&local" : ""}`, "window.__showcase !== undefined");
  const { root: doc } = await browser.send("DOM.getDocument", { depth: 0 });
  const { nodeId } = await browser.send("DOM.querySelector", { nodeId: doc.nodeId, selector: "#file-input" });
  await browser.send("DOM.setFileInputFiles", { nodeId, files: [path] });
  await waitFor(`__showcase.items().find((i) => i.name === ${JSON.stringify(NAME)})?.drawn === true`);
  // Drop the boot head: its volume is by far the costliest thing to draw, and it is in the way.
  await waitFor(`__showcase.items().some((m) => m.name.includes("headsq") && m.drawn)`);
  await page(`(document.querySelector('[data-mesh*="headsq"] button.danger')?.click(), true)`);
  await waitFor(`!__showcase.items().some((m) => m.name.includes("headsq"))`);
  await clickOn(`[data-mesh=${JSON.stringify(NAME)}] .mesh-name`);
  await sleep(300);
  await page(`(document.activeElement?.blur?.(), true)`);
  await browser.send("Input.dispatchKeyEvent", { type: "keyDown", key: "f", code: "KeyF", text: "f", windowsVirtualKeyCode: 70 });
  await browser.send("Input.dispatchKeyEvent", { type: "keyUp", key: "f", code: "KeyF", windowsVirtualKeyCode: 70 });
  await sleep(1000);
}

async function verify() {
  const { buf } = sphereStl(20_000);
  const path = join(fixtures, NAME);
  writeFileSync(path, buf);
  let failed = false;
  const check = (name, ok, detail) => {
    console.log(`${ok ? "PASS" : "FAIL"} ${name}: ${detail}`);
    failed ||= !ok;
  };
  const q = JSON.stringify(NAME);
  const positions = () => page(`__showcase.positions(${q})`);
  let original = 0;
  const farthest = (a, b) => {
    let worst = 0;
    // The sphere's own slots: later ones are the remeshes', and get reused.
    for (let k = 0; k + 2 < original; k += 3) {
      worst = Math.max(worst, Math.hypot(a[k] - b[k], a[k + 1] - b[k + 1], a[k + 2] - b[k + 2]));
    }
    return worst;
  };
  const drag = async (mx, my, steps = 40, offset = 0, across = 0) => {
    const center = await page(`__showcase.screenPoint(${q})`);
    const cx = center.x + across;
    const cy = center.y + offset;
    await mouse("mouseMoved", cx, cy, { button: "none" });
    await sleep(150);
    await mouse("mousePressed", cx, cy, { buttons: 1 });
    await sleep(60);
    for (let k = 1; k <= steps; k++) {
      await mouse("mouseMoved", cx + (mx * k) / steps, cy + (my * k) / steps, { buttons: 1 });
      await sleep(12);
    }
    await mouse("mouseReleased", cx + mx, cy + my, { buttons: 0 });
    await sleep(800);
  };
  const ctrlZ = async () => {
    const key = { key: "z", code: "KeyZ", windowsVirtualKeyCode: 90, modifiers: 2 };
    await browser.send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...key });
    await browser.send("Input.dispatchKeyEvent", { type: "keyUp", ...key });
    await sleep(300);
  };
  const arm = async (tool) => {
    await clickOn(`[data-action="sculpt-tool-${tool}"]`);
    await waitFor(`document.querySelector('[data-action="sculpt-radius"]') !== null`);
    await setValue('[data-action="sculpt-radius"]', 0.4);
    await sleep(150);
  };
  const disarm = async (tool) => {
    await clickOn(`[data-action="sculpt-tool-${tool}"]`);
    await sleep(100);
  };
  for (const mode of ["sync", "worker"]) {
    await load(mode, path);
    const start = await positions();
    original = start.length;

    await arm("inflate");
    const before = await page("__showcase.sculpt()");
    await drag(80, 0);
    const painted = await positions();
    const after = await page("__showcase.sculpt()");
    check(`${mode} inflate`, farthest(start, painted) > 0.05 && after.strokes === before.strokes + 1, `moved ${farthest(start, painted).toFixed(3)}, ${after.dabs - before.dabs} dabs`);
    check(`${mode} live remesh`, after.liveRemeshes > before.liveRemeshes && painted.length > start.length, `${after.liveRemeshes - before.liveRemeshes} pass(es), ${(painted.length - start.length) / 3} new vertices`);
    await disarm("inflate");
    if (mode === "sync") {
      await page(`(document.activeElement?.blur?.(), true)`);
      await ctrlZ();
      const undone = await positions();
      check("sync one undo per stroke", farthest(start, undone) === 0, `max difference ${farthest(start, undone)} after one Ctrl+Z`);
    }

    const ahead = await positions();
    await arm("grab");
    // Below the inflated lump, where the sphere's own vertices are the surface.
    await drag(60, 0, 40, 110);
    const grabbed = await positions();
    const expected = 60 * 2 * Math.hypot(...(await page("__showcase.cameraPose().position"))) * Math.tan((Math.PI * 45) / 360) / (await page(`document.querySelector("canvas").clientHeight`));
    check(`${mode} grab`, Math.abs(farthest(ahead, grabbed) - expected) < 0.45 * expected, `moved ${farthest(ahead, grabbed).toFixed(3)} for a drag of about ${expected.toFixed(3)}`);
    await disarm("grab");

    // A painted mask: dabs touching it drop, and it does not move.
    const atMask = await positions();
    const toggleMask = () => page(`(document.querySelector('[data-action="sculpt-mask-paint"]').click(), true)`);
    await arm("inflate");
    await disarm("inflate");
    await toggleMask();
    await sleep(150);
    await drag(1, 0, 1, -90, -90); // Clear of the gizmo, which stays up for a mask paint.
    await toggleMask();
    await sleep(150);
    await arm("inflate");
    const maskBefore = await page("__showcase.sculpt()");
    await drag(1, 0, 1, -90, -90);
    const maskAfter = await page("__showcase.sculpt()");
    const still = await positions();
    check(`${mode} mask`, maskAfter.maskedDabs > maskBefore.maskedDabs && farthest(atMask, still) === 0, `${maskAfter.maskedDabs - maskBefore.maskedDabs} dropped, moved ${farthest(atMask, still)}`);
    await disarm("inflate");
    // A release before the press's raycast returns (a busy scene): no stroke opens.
    await arm("inflate");
    const raceBefore = await page("__showcase.sculpt()");
    const raceStart = await positions();
    await page(`(() => {
      const scene = __showcase.store.scene;
      const raycast = scene.raycast;
      scene.raycast = function (...args) {
        return new Promise((r) => setTimeout(r, 400)).then(() => raycast.apply(this, args));
      };
      window.__fastRaycast = () => delete scene.raycast;
      return true;
    })()`);
    const raceAt = await page(`__showcase.screenPoint(${q})`);
    const [rx, ry] = [raceAt.x + 90, raceAt.y + 110];
    await mouse("mouseMoved", rx, ry, { button: "none" });
    await mouse("mousePressed", rx, ry, { buttons: 1 });
    await mouse("mouseReleased", rx, ry, { buttons: 0 });
    await sleep(700);
    await page("(window.__fastRaycast(), true)");
    // The button is up: moving now paints only if a stroke was left open.
    for (let k = 1; k <= 10; k++) {
      await mouse("mouseMoved", rx + 4 * k, ry, { button: "none" });
      await sleep(30);
    }
    await sleep(400);
    const raceAfter = await page("__showcase.sculpt()");
    const raceEnd = await positions();
    check(
      `${mode} release before the press lands`,
      raceAfter.dabs === raceBefore.dabs && raceAfter.strokes === raceBefore.strokes && farthest(raceStart, raceEnd) === 0,
      `${raceAfter.dabs - raceBefore.dabs} dabs, moved ${farthest(raceStart, raceEnd)}`,
    );
    await drag(30, 0, 10, 110, 90);
    const raceLater = await page("__showcase.sculpt()");
    check(`${mode} the next press still paints`, raceLater.strokes === raceAfter.strokes + 1 && raceLater.dabs > raceAfter.dabs, `${raceLater.dabs - raceAfter.dabs} dabs`);
    await disarm("inflate");

    const errors = browser.consoleLines.filter((l) => /exception|error:/i.test(l));
    check(`${mode} console`, errors.length === 0, errors.slice(0, 3).join(" | ") || "clean");
  }
  return failed;
}

if (argv.includes("--verify")) {
  const failed = await verify();
  await browser.close();
  await server.close();
  rmSync(fixtures, { recursive: true, force: true });
  process.exit(failed ? 1 : 0);
}

const rows = [];
for (const triangles of triangleCounts) {
  const { buf, count } = sphereStl(triangles);
  const path = join(fixtures, NAME);
  writeFileSync(path, buf);
  for (const mode of ["sync", "worker"]) {
    await load(mode, path);
    for (const tool of tools) {
      await waitFor(`document.querySelector('[data-action="sculpt-tool-${tool}"]') !== null`);
      await clickOn(`[data-action="sculpt-tool-${tool}"]`);
      await waitFor(`document.querySelector('[data-action="sculpt-radius"]') !== null`);
      await uncheck('[data-action="sculpt-remesh"]');
      await uncheck('[data-action="sculpt-live-remesh"]');
      await setValue('[data-action="sculpt-radius"]', radius);
      await sleep(200);
      const r = await stroke();
      const row = {
        mode,
        triangles: count,
        tool,
        median: percentile(r.gaps, 0.5),
        p95: percentile(r.gaps, 0.95),
        max: r.gaps.at(-1),
        longTask: r.longest,
        dabs: r.dabs,
        strokes: r.strokes,
      };
      rows.push(row);
      console.log(
        `${mode.padEnd(6)} ${String(count).padStart(8)} ${tool.padEnd(8)} frame median ${ms(row.median)} p95 ${ms(row.p95)} max ${ms(row.max)} ms; long task ${ms(row.longTask)} ms; ${row.dabs} dabs in ${r.elapsed} ms, ${r.strokes} stroke(s)`,
      );
      await clickOn(`[data-action="sculpt-tool-${tool}"]`);
      await sleep(100);
    }
  }
}
await browser.close();
await server.close();
rmSync(fixtures, { recursive: true, force: true });
const out = opt("json", null);
if (out !== null) {
  writeFileSync(out, JSON.stringify(rows, null, 2));
}
