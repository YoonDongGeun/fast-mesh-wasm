// Checks the showcase app (examples/showcase) in headless Chrome, driven over
// the DevTools protocol (scripts/chrome-cdp.mjs) against `vite preview`:
// a scripted session through the UI (file input, mouse on the canvas and the
// gizmo, toolbar buttons, keyboard), read back through `window.__showcase`.
//
//   node scripts/showcase-check.mjs               # every check
//   node scripts/showcase-check.mjs --build       # build the example first
//   node scripts/showcase-check.mjs --threads 4   # ?threads=4, and expect exactly 4
//   node scripts/showcase-check.mjs --big         # also the 1M-triangle upload (RF4)
//   node scripts/showcase-check.mjs --swiftshader # software WebGL (the default on Linux and under CI)
//   node scripts/showcase-check.mjs --verbose     # echo the page's console
//   node scripts/showcase-check.mjs --screenshot out.png   # the page at the end, as a PNG
//
// Checks (each prints PASS or FAIL; the exit code is 1 on any FAIL):
// - page: cross-origin isolated, and the Worker scene runs threaded
//   (threads > 1, or exactly --threads).
// - demo: the demo meshes A, B and C are listed and drawn; the scene is idle.
// - select: a click on A in the canvas selects it.
// - drag: dragging the gizmo's centre handle moves A (one undo step).
// - undo: the Undo button puts A back; Ctrl+Shift+Z redoes, Ctrl+Z undoes again.
// - upload: three fixtures (binary STL, OBJ with quads, ASCII PLY) through
//   the file input are listed with their triangle counts and drawn; a broken
//   file shows a failed load.
// - big (--big): a generated 1M-triangle binary STL through the file input;
//   the page's longest main-thread task while it loads stays under 50 ms
//   (a PerformanceObserver long-task probe: no `longtask` entry at all).
// - console: no console error or uncaught exception, in the page or its Workers.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
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
const verbose = flag("--verbose");
const threadsOpt = option("--threads", undefined);
const threads = threadsOpt === undefined ? undefined : Number(threadsOpt);
const port = Number(option("--port", 4193));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
let failures = 0;
function verdict(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
  if (!ok) {
    failures++;
  }
}

function run(cmd, args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, stdio: "inherit" });
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} ${args.join(" ")} exited ${code}`))));
  });
}

// ---- Fixtures ----

/** A binary STL of `tris` (arrays of 9 numbers). */
function binaryStl(tris) {
  const buf = Buffer.alloc(84 + 50 * tris.length);
  buf.write("fast-mesh showcase-check", 0, "latin1");
  buf.writeUInt32LE(tris.length, 80);
  tris.forEach((t, i) => t.forEach((x, k) => buf.writeFloatLE(x, 84 + 50 * i + 12 + 4 * k)));
  return buf;
}

/** An octahedron around `c` (8 triangles), outward. */
function octahedron([cx, cy, cz], r) {
  const p = [
    [r, 0, 0],
    [-r, 0, 0],
    [0, r, 0],
    [0, -r, 0],
    [0, 0, r],
    [0, 0, -r],
  ].map(([x, y, z]) => [cx + x, cy + y, cz + z]);
  const faces = [
    [0, 2, 4],
    [2, 1, 4],
    [1, 3, 4],
    [3, 0, 4],
    [2, 0, 5],
    [1, 2, 5],
    [3, 1, 5],
    [0, 3, 5],
  ];
  return faces.map((f) => f.flatMap((i) => p[i]));
}

const CUBE_OBJ = `# a unit cube of quads, below the demo meshes
v -0.5 -1.5 -0.5
v 0.5 -1.5 -0.5
v 0.5 -0.5 -0.5
v -0.5 -0.5 -0.5
v -0.5 -1.5 0.5
v 0.5 -1.5 0.5
v 0.5 -0.5 0.5
v -0.5 -0.5 0.5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
`;

const TETRA_PLY = `ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
2 -1.5 0
3 -1.5 0
2.5 -1.5 1
2.5 -0.7 0.4
3 0 2 1
3 0 1 3
3 1 2 3
3 2 0 3
`;

/** A UV sphere as a binary STL: 2 · segments · (rings − 1) triangles (1,000,000 for 1000 x 501). */
function bigSphereStl(segments, rings) {
  const count = 2 * segments * (rings - 1);
  const buf = Buffer.alloc(84 + 50 * count);
  buf.write("fast-mesh showcase-check big sphere", 0, "latin1");
  buf.writeUInt32LE(count, 80);
  const pt = (i, j) => {
    if (i === 0) return [0, 1, 0];
    if (i === rings) return [0, -1, 0];
    const t = (Math.PI * i) / rings;
    const f = (2 * Math.PI * (j % segments)) / segments;
    return [Math.sin(t) * Math.cos(f), Math.cos(t), Math.sin(t) * Math.sin(f)];
  };
  let at = 84;
  const tri = (a, b, c) => {
    at += 12;
    for (const p of [a, b, c]) {
      for (const x of p) {
        buf.writeFloatLE(x * 2, at);
        at += 4;
      }
    }
    at += 2;
  };
  for (let j = 0; j < segments; j++) {
    tri(pt(0, 0), pt(1, j + 1), pt(1, j));
    tri(pt(rings, 0), pt(rings - 1, j), pt(rings - 1, j + 1));
  }
  for (let i = 1; i < rings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      tri(pt(i, j), pt(i, j + 1), pt(i + 1, j + 1));
      tri(pt(i, j), pt(i + 1, j + 1), pt(i + 1, j));
    }
  }
  return { buf, count };
}

// ---- Session ----

if (flag("--build")) {
  await run(process.execPath, [vite, "build"], example);
}
const fixtures = mkdtempSync(join(tmpdir(), "fast-mesh-showcase-"));
const server = await startServer(
  process.execPath,
  [vite, "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"],
  { cwd: example, url: `http://127.0.0.1:${port}/` },
);
// WebGL without a GPU (CI runners): SwiftShader through ANGLE. On by default
// on Linux and under CI, or with --swiftshader. `args` add to launchChrome's own.
const swiftshader = flag("--swiftshader") || process.platform === "linux" || process.env.CI !== undefined;
const browser = await launchChrome({
  verbose,
  args: ["--window-size=1400,900", ...(swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [])],
});
const page = (expr) => browser.evaluate(expr);
const hook = (expr) => page(`window.__showcase.${expr}`);

/** Polls `expr` in the page until it is truthy; returns its value, or `undefined` after `ms`. */
async function waitFor(expr, ms = 20_000) {
  const t = Date.now();
  for (;;) {
    const v = await page(expr).catch(() => undefined);
    if (v) {
      return v;
    }
    if (Date.now() - t > ms) {
      return undefined;
    }
    await sleep(50);
  }
}

const mouse = (type, x, y, extra = {}) =>
  browser.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });

async function click(x, y) {
  await mouse("mouseMoved", x, y, { button: "none" });
  await mouse("mousePressed", x, y, { buttons: 1 });
  await mouse("mouseReleased", x, y, { buttons: 0 });
}

/** Ctrl(+Shift)+Z on the focused element. */
async function ctrlZ(shift) {
  const modifiers = 2 | (shift ? 8 : 0);
  const key = { key: shift ? "Z" : "z", code: "KeyZ", windowsVirtualKeyCode: 90, modifiers };
  await browser.send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...key });
  await browser.send("Input.dispatchKeyEvent", { type: "keyUp", ...key });
}

/** Sets the app's file input to `paths` (fires its change event). */
async function upload(paths) {
  const { root: doc } = await browser.send("DOM.getDocument", { depth: 0 });
  const { nodeId } = await browser.send("DOM.querySelector", { nodeId: doc.nodeId, selector: "#file-input" });
  await browser.send("DOM.setFileInputFiles", { nodeId, files: paths });
}

const length = (v) => Math.hypot(...v);
const A = "A · blobby sphere";

try {
  const query = threads === undefined ? "" : `?threads=${threads}`;
  await browser.goto(`http://127.0.0.1:${port}/${query}`, "window.__showcase !== undefined");

  // ---- page ----
  const info = await page(`({
    isolated: crossOriginIsolated,
    mode: __showcase.mode,
    threads: __showcase.threads(),
    cores: navigator.hardwareConcurrency,
    gl: (() => {
      const gl = document.createElement("canvas").getContext("webgl2");
      const ext = gl?.getExtension("WEBGL_debug_renderer_info");
      return gl === null ? "no WebGL2" : ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    })(),
  })`);

  console.log(`${browser.name} (vite preview); ${info.cores} logical cores; WebGL: ${info.gl}`);
  const threadsOk = threads === undefined ? info.threads > 1 : info.threads === threads;
  verdict("page", info.isolated && info.mode === "worker" && threadsOk, `crossOriginIsolated ${info.isolated}, ${info.mode} scene, threads ${info.threads}`);

  // ---- demo ----
  const demo = await waitFor(
    "(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn) && __showcase.status().status === 'idle' && i; })()",
    60_000,
  );
  verdict(
    "demo",
    demo !== undefined && demo.map((m) => m.name[0]).join("") === "ABC",
    demo === undefined ? `items ${JSON.stringify(await hook("items()"))}` : demo.map((m) => `${m.name} (${m.triangles})`).join(", "),
  );

  // ---- select: click A in the canvas ----
  const at = await hook(`screenPoint(${JSON.stringify(A)})`);
  await click(at.x, at.y);
  const selected = await waitFor(`__showcase.selected() === ${JSON.stringify(A)}`, 5000);
  verdict("select", selected === true, `clicked (${fmt(at.x)}, ${fmt(at.y)}); selected ${await hook("selected()")}`);

  // ---- drag: the gizmo's centre handle (free move in the view plane) ----
  // The gizmo is attached to A, then hovered until it reports a handle under
  // the pointer (it picks against its own matrices, set when it is drawn).
  const attached = await waitFor(`__showcase.gizmo()?.mesh === ${JSON.stringify(A)}`, 5000);
  const from = await hook(`screenPoint(${JSON.stringify(A)})`);
  let axis = null;
  for (const t = Date.now(); attached && axis === null && Date.now() - t < 5000; ) {
    await mouse("mouseMoved", from.x, from.y, { button: "none" });
    axis = await hook("gizmo()?.axis ?? null");
    if (axis === null) {
      await sleep(50);
    }
  }
  await mouse("mousePressed", from.x, from.y, { buttons: 1 });
  const wasDragging = await waitFor("__showcase.dragging()", 2000);
  for (let i = 1; i <= 12; i++) {
    await mouse("mouseMoved", from.x + 10 * i, from.y - 3 * i, { buttons: 1 });
    await sleep(16);
  }
  await mouse("mouseReleased", from.x + 120, from.y - 36, { buttons: 0 });
  const moved = await waitFor(
    `(() => { const t = __showcase.translation(${JSON.stringify(A)}); const s = __showcase.status(); return !__showcase.dragging() && Math.hypot(...t) > 0.05 && s.canUndo && t; })()`,
    5000,
  );
  verdict(
    "drag",
    attached === true && axis !== null && wasDragging === true && moved !== undefined,
    `gizmo on A ${attached === true}, hovered handle ${axis}, dragging ${wasDragging === true}; A moved by ${moved === undefined ? "nothing" : fmt(length(moved))}`,
  );

  // ---- undo / redo ----
  await page("document.querySelector('[data-action=undo]').click()");
  const undone = await waitFor(
    `(() => { const t = __showcase.translation(${JSON.stringify(A)}); return Math.hypot(...t) < 1e-9 && __showcase.status().canRedo; })()`,
    5000,
  );
  await ctrlZ(true);
  const redone = await waitFor(
    `(() => { const t = __showcase.translation(${JSON.stringify(A)}); return Math.hypot(...t) > 0.05; })()`,
    5000,
  );
  await ctrlZ(false);
  const undoneAgain = await waitFor(`Math.hypot(...__showcase.translation(${JSON.stringify(A)})) < 1e-9`, 5000);
  verdict(
    "undo",
    undone === true && redone === true && undoneAgain === true,
    `Undo button ${undone === true}, Ctrl+Shift+Z ${redone === true}, Ctrl+Z ${undoneAgain === true}`,
  );

  // ---- upload ----
  const files = {
    "octahedron.stl": binaryStl(octahedron([-2, -1, 0], 0.5)),
    "cube.obj": CUBE_OBJ,
    "tetra.ply": TETRA_PLY,
    "broken.stl": "this is not a mesh",
  };
  for (const [name, data] of Object.entries(files)) {
    writeFileSync(join(fixtures, name), data);
  }
  await upload(Object.keys(files).map((n) => join(fixtures, n)));
  const expected = { "octahedron.stl": 8, "cube.obj": 12, "tetra.ply": 4 };
  const loaded = await waitFor(
    `(() => { const i = __showcase.items(); const want = ${JSON.stringify(expected)};
      const got = i.filter((m) => m.name in want);
      return got.length === 3 && got.every((m) => m.drawn && m.triangles === want[m.name]) && i; })()`,
    20_000,
  );
  const broken = await waitFor("__showcase.jobs().find((j) => j.name === 'broken.stl' && j.phase === 'error')", 10_000);
  verdict(
    "upload",
    loaded !== undefined && broken !== undefined,
    `${loaded === undefined ? `items ${JSON.stringify(await hook("items()"))}` : loaded.slice(3).map((m) => `${m.name} (${m.triangles})`).join(", ")}; broken.stl ${broken === undefined ? "not reported" : `failed: ${broken.message}`}`,
  );

  // ---- big: a 1M-triangle STL upload keeps the main thread responsive (RF4) ----
  if (flag("--big")) {
    const t0 = Date.now();
    const { buf, count } = bigSphereStl(1000, 501);
    const path = join(fixtures, "big-sphere.stl");
    writeFileSync(path, buf);
    console.log(`  generated ${count.toLocaleString("en")} triangles (${fmt(buf.length / 2 ** 20)} MB) in ${fmt((Date.now() - t0) / 1e3)} s`);
    // Long tasks (>= 50 ms) on the page's main thread, and the longest gap between animation frames.
    await page(`(() => {
      const probe = { start: performance.now(), tasks: [], maxGap: 0, last: performance.now(), on: true };
      new PerformanceObserver((list) => { for (const e of list.getEntries()) probe.tasks.push({ start: e.startTime, duration: e.duration }); })
        .observe({ type: "longtask" });
      const frame = (t) => { probe.maxGap = Math.max(probe.maxGap, t - probe.last); probe.last = t; if (probe.on) requestAnimationFrame(frame); };
      requestAnimationFrame(frame);
      window.__probe = probe;
    })()`);
    await sleep(500); // a quiet baseline before the upload
    await page("(__probe.start = performance.now(), __probe.maxGap = 0, true)");
    await upload([path]);
    const big = await waitFor(
      `(() => { const m = __showcase.items().find((i) => i.name === "big-sphere.stl");
        return m !== undefined && m.drawn && __showcase.jobs().every((j) => j.name !== "big-sphere.stl" || j.phase === "done") && performance.now() - __probe.start; })()`,
      180_000,
    );
    await sleep(1000); // a few more frames with the mesh drawn
    const probe = await page(
      "(__probe.on = false, { tasks: __probe.tasks.filter((t) => t.start >= __probe.start), maxGap: __probe.maxGap, triangles: __showcase.items().find((i) => i.name === 'big-sphere.stl')?.triangles })",
    );
    const longest = probe.tasks.reduce((m, t) => Math.max(m, t.duration), 0);
    // The probe itself: an 80 ms task on the main thread must show up as a long task.
    const calibrated = await page(`new Promise((resolve) => {
      const mark = performance.now();
      new PerformanceObserver((list, obs) => {
        if (list.getEntries().some((e) => e.startTime >= mark)) { obs.disconnect(); resolve(true); }
      }).observe({ type: "longtask" });
      setTimeout(() => { const t = performance.now(); while (performance.now() - t < 80) {} }, 0);
      setTimeout(() => resolve(false), 3000);
    })`);
    console.log(
      `  loaded in ${big === undefined ? "-" : `${fmt(big)} ms`}; ${probe.tasks.length} long task(s)${probe.tasks.length > 0 ? ` (longest ${fmt(longest)} ms: ${probe.tasks.map((t) => fmt(t.duration)).join(", ")})` : ""}; longest frame gap ${fmt(probe.maxGap)} ms`,
    );
    verdict(
      "big",
      big !== undefined && probe.triangles === count && probe.tasks.length === 0 && calibrated,
      `${count.toLocaleString("en")} triangles; longest main-thread task ${probe.tasks.length === 0 ? "under 50 ms (no long task)" : `${fmt(longest)} ms`}; the probe sees an 80 ms task: ${calibrated}`,
    );
  }

  if (option("--screenshot", undefined) !== undefined) {
    const { data } = await browser.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(option("--screenshot"), Buffer.from(data, "base64"));
  }

  // ---- console ----
  const problems = browser.consoleLines.filter((l) => /^(page|worker): (error|assert): |^(page|worker): exception: /.test(l));
  for (const p of problems) {
    console.log(`  ${p}`);
  }
  if (verbose) {
    for (const l of browser.consoleLines) {
      console.log(`  ${l}`);
    }
  }
  verdict("console", problems.length === 0, `${browser.consoleLines.length} console line(s), ${problems.length} error(s)`);
} catch (e) {
  console.error(e);
  console.error(browser.consoleLines.join("\n"));
  failures++;
} finally {
  await browser.close();
  server.stop();
  rmSync(fixtures, { recursive: true, force: true });
}
if (failures > 0) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
