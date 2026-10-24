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
// - glsl: bindMesh's fragment shading run on the GPU (a float render target,
//   read back) for constant quads at a contact heatmap's knots, range ends,
//   +-tolerance, far, -far, NaN and random values, and for built-in maps with
//   a centre and both out-of-range modes, at softness 0 and above, against
//   the library's JavaScript reference (packages/fast-mesh-wasm/src/colormap.ts,
//   loaded with Node's type stripping): every colour within one 16-bit level
//   at softness 0 (the exact path), within four with softness (the filter
//   sums in f32 on the GPU).
// - select: a click on A in the canvas selects it.
// - drag: dragging the gizmo's centre handle moves A (one undo step).
// - undo: the Undo button puts A back; Ctrl+Shift+Z redoes, Ctrl+Z undoes again.
// - heatmap: the collision pair defaults to A and B, both coloured against
//   each other (reds where they penetrate, green to blue across the gap);
//   the readouts show the penetration, intersecting, a gap of 0 and contact
//   triangles.
// - softness: the meshes are drawn with fragment shading, and the
//   "Boundary softness" slider changes the boundary pixels: two screenshots
//   of the canvas at softness 0 are identical, one at the slider's maximum
//   differs (decoded and compared in the page).
// - colour-from: "Colour from" 0 colours penetration only (no gap colours);
//   the readouts stay up while the slider moves.
// - readouts: dragging A away from B pauses the readouts: no query starts
//   while the gizmo drags (reading field values would force stale poses
//   current), and a result held across the drag's start (the test hook
//   holds it) lands mid-drag and is dropped. The drag's end refreshes once,
//   to the final pose: no longer intersecting, a gap, as queried now. Undo
//   puts A back.
// - open-target (RF1): picking C (open) as B still colours A and C, and
//   the panel warns that C is open (from `analyze` of C).
// - repair (RF5): with C selected and coloured, the repair panel lists its
//   two holes and two components; Fill holes (up to 64 edges, the default)
//   leaves none, C stays coloured (over its new vertices too) and the
//   warning goes; Undo brings both back and clears the stale report.
// - group: the Group mode with A, B and C (the default members, all pairs):
//   each is coloured by its distance to the other two (one field measured
//   to both), with colours on all three; the readouts list the three pairs
//   (A and B intersecting, a penetration depth; a gap between A and C)
//   and the warning that C is open;
//   hovering A shows the partner nearest to the vertex under the pointer.
//   Unticking "All pairs" and then the A-C cell of the pair matrix leaves
//   A and C each coloured against B only (B against both); "All pairs"
//   again restores them. Removing C keeps A and B coloured, each against the other, with one
//   pair's readouts and no error. Back to the pair mode.
// - remove (RF3): removing B while A and B are coloured resets B's picker,
//   stops A's colouring and the readouts, with no error.
// - upload: three fixtures (binary STL, OBJ with quads, ASCII PLY) through
//   the file input are listed with their triangle counts and drawn; a broken
//   file shows a failed load.
// - big (--big): a generated 1M-triangle binary STL through the file input;
//   the page's longest main-thread task while it loads stays under 50 ms
//   (a PerformanceObserver long-task probe: no `longtask` entry at all).
// - local: a short pass with ?local (the synchronous Scene on the main
//   thread): the demo meshes, the heatmap readouts, the Group mode's
//   per-pair readouts (depths from `fieldNearest`), Fill holes on C and undo.
// - console: no console error or uncaught exception, in the page or its Workers.
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
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

/** Sets a range input (`data-control`) to `value`, as a user drag would (React reads `input` events). */
async function setRange(control, value) {
  await page(`(() => {
    const el = document.querySelector('[data-control=${control}]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, ${JSON.stringify(String(value))});
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
}

/** Picks the mesh named `name` in the collision panel's `side` picker. */
async function pick(side, name) {
  await page(`(() => {
    const sel = document.querySelector('[data-pick=${side}]');
    const opt = [...sel.options].find((o) => o.text === ${JSON.stringify(name)});
    sel.value = opt.value;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
}

/** Clicks the first element matching `selector`; false when there is none. */
const clickOn = (selector) =>
  page(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); el?.click(); return el !== null; })()`);

/**
 * Drags the gizmo's centre handle on `name` (it must be selected) by
 * `(dx, dy)` pixels in `steps` moves `ms` apart. The gizmo is hovered until
 * it reports a handle under the pointer (it picks against its own matrices,
 * set when it is drawn).
 */
async function dragGizmo(name, dx, dy, steps, ms, { midway = async () => {}, beforeRelease = async () => {} } = {}) {
  const attached = await waitFor(`__showcase.gizmo()?.mesh === ${JSON.stringify(name)}`, 5000);
  const from = await hook(`screenPoint(${JSON.stringify(name)})`);
  let axis = null;
  for (const t = Date.now(); attached && axis === null && Date.now() - t < 5000; ) {
    await mouse("mouseMoved", from.x, from.y, { button: "none" });
    axis = await hook("gizmo()?.axis ?? null");
    if (axis === null) {
      await sleep(50);
    }
  }
  await mouse("mousePressed", from.x, from.y, { buttons: 1 });
  const dragging = await waitFor("__showcase.dragging()", 2000);
  for (let i = 1; i <= steps; i++) {
    await mouse("mouseMoved", from.x + (dx * i) / steps, from.y + (dy * i) / steps, { buttons: 1 });
    await sleep(ms);
    if (i === Math.floor(steps / 2)) {
      await midway();
    }
  }
  await beforeRelease();
  await mouse("mouseReleased", from.x + dx, from.y + dy, { buttons: 0 });
  return { attached: attached === true, axis, dragging: dragging === true };
}

/** `extra` (an expression over `c`, the collision panel's state) in the page: `c` when it holds. */
const collision = (extra) => `(() => { const c = __showcase.collision(); return (${extra}) && c; })()`;
/** The vertices `colorCounts` counted. */
const used = (counts) => (counts === null ? 0 : counts.red + counts.white + counts.gap + counts.other);

const length = (v) => Math.hypot(...v);
const A = "A · blobby sphere";
const B = "B · rounded box";
const C = "C · open shell + piece";
const q = JSON.stringify;

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

  // ---- glsl: the fragment shading on the GPU against its reference ----
  {
    const { shadingOf, shadeReference } = await import(
      pathToFileURL(join(root, "packages", "fast-mesh-wasm", "src", "colormap.ts")).href
    );
    const heatRule = await hook("heatmapRule({ depth: 1, gap: 2, tolerance: 0.25 })");
    const rules = [
      heatRule,
      { map: "turbo", range: [-1, 3], center: 0.5 },
      { map: "viridis", range: [0, 1], below: "base", above: "clamp" },
      { map: "coolwarm", range: [-2, 1], center: 0, base: [10, 20, 30, 40], nan: [1, 2, 3, 4] },
    ];
    let seed = 11;
    const random = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    // Worst difference (and where) at softness 0 (exact path) and above (the filter).
    const worst = [0, 0];
    const bad = ["", ""];
    let count = 0;
    for (const rule of rules) {
      const [lo, hi] = rule.range;
      const knots = typeof rule.map === "string" ? [] : rule.map.stops.map(([t]) => lo + t * (hi - lo));
      const values = [lo, hi, -0.25, 0.25, 0, ...knots, Math.fround(lo - 1e-3), hi + 1, "Infinity", "-Infinity", "NaN"];
      for (let k = 0; k < 40; k++) {
        values.push(Math.fround(lo - 0.1 * (hi - lo) + random() * 1.2 * (hi - lo)));
      }
      for (const softness of [0, 0.05 * (hi - lo)]) {
        const list = `[${values.map((v) => (typeof v === "string" ? v : String(v))).join(",")}]`;
        const probe = await hook(`shadeProbe(${q(rule)}, ${list}, ${softness})`);
        const s = shadingOf(rule);
        probe.rgba.forEach((got, i) => {
          const want = shadeReference(s, probe.attribute[i], softness);
          const err = Math.max(...got.map((x, c) => Math.abs(x - want[c])));
          const path = softness > 0 ? 1 : 0;
          count++;
          if (err > worst[path]) {
            worst[path] = err;
            bad[path] = ` (value ${values[i]}: GPU ${got.map((x) => x.toFixed(6))} vs ${want.map((x) => x.toFixed(6))})`;
          }
        });
      }
    }
    // Exact path: one 16-bit level, plus float slack. The softness filter sums
    // in f32 on the GPU: within 4 levels (1/64 of an 8-bit level).
    const levels = (x) => (x * 65535).toFixed(3);
    verdict(
      "glsl",
      worst[0] <= 1.01 / 65535 && worst[1] <= 4 / 65535,
      `${count} colours; worst difference ${levels(worst[0])} 16-bit levels at softness 0${bad[0]}, ${levels(worst[1])} with softness${bad[1]}`,
    );
  }

  // ---- select: click A in the canvas ----
  const at = await hook(`screenPoint(${JSON.stringify(A)})`);
  await click(at.x, at.y);
  const selected = await waitFor(`__showcase.selected() === ${JSON.stringify(A)}`, 5000);
  verdict("select", selected === true, `clicked (${fmt(at.x)}, ${fmt(at.y)}); selected ${await hook("selected()")}`);

  // ---- drag: the gizmo's centre handle (free move in the view plane) ----
  const { attached, axis, dragging: wasDragging } = await dragGizmo(A, 120, -36, 12, 16);
  const moved = await waitFor(
    `(() => { const t = __showcase.translation(${JSON.stringify(A)}); const s = __showcase.status(); return !__showcase.dragging() && Math.hypot(...t) > 0.05 && s.canUndo && t; })()`,
    5000,
  );
  verdict(
    "drag",
    attached && axis !== null && wasDragging && moved !== undefined,
    `gizmo on A ${attached}, hovered handle ${axis}, dragging ${wasDragging}; A moved by ${moved === undefined ? "nothing" : fmt(length(moved))}`,
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

  // ---- heatmap: A and B, each coloured against the other ----
  await sleep(300); // the undo's change records reach the panel, then its readouts settle
  const heat = await waitFor(
    collision(`c.a === ${q(A)} && c.b === ${q(B)}
      && c.coloring[${q(A)}]?.to === ${q(B)} && c.coloring[${q(A)}].status === "ready"
      && c.coloring[${q(B)}]?.to === ${q(A)} && c.coloring[${q(B)}].status === "ready"
      && c.settled && c.readouts?.penetration?.depth > 0`),
    30_000,
  );
  const colorsA = await hook(`colorCounts(${q(A)})`);
  const r0 = heat?.readouts;
  verdict(
    "heatmap",
    heat !== undefined && r0.intersects && r0.gap === 0 && r0.contacts.a > 0 && r0.contacts.b > 0 && colorsA?.red > 0 && colorsA?.gap > 0,
    heat === undefined
      ? `collision ${q(await hook("collision()"))}`
      : `A and B coloured; penetration ${fmt(r0.penetration.depth)}, intersecting ${r0.intersects}, gap ${r0.gap}, contact triangles ${r0.contacts.a} · ${r0.contacts.b}; A's vertices ${q(colorsA)}`,
  );

  // ---- softness: fragment shading; the slider moves the boundary pixels ----
  const rect = await page(
    `(() => { const r = document.querySelector("canvas").getBoundingClientRect(); return { x: r.left, y: r.top, width: r.width, height: r.height }; })()`,
  );
  const shot = async () => (await browser.send("Page.captureScreenshot", { format: "png", clip: { ...rect, scale: 1 } })).data;
  /** Pixels that differ by more than 8 in a channel between two PNG screenshots, decoded in the page. */
  const pixelDiff = (a, b) =>
    page(`(async () => {
      const load = async (data) => {
        const img = new Image();
        img.src = "data:image/png;base64," + data;
        await img.decode();
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        const g = c.getContext("2d");
        g.drawImage(img, 0, 0);
        return g.getImageData(0, 0, c.width, c.height).data;
      };
      const [x, y] = await Promise.all([load(${q(a)}), load(${q(b)})]);
      let n = 0;
      for (let i = 0; i < x.length; i += 4) {
        if (Math.abs(x[i] - y[i]) > 8 || Math.abs(x[i + 1] - y[i + 1]) > 8 || Math.abs(x[i + 2] - y[i + 2]) > 8) n++;
      }
      return { n, total: x.length / 4 };
    })()`);
  const shadingA = await hook(`shading(${q(A)})`);
  await setRange("softness", 0);
  await sleep(300);
  const hard = await shot();
  await sleep(200);
  const hardAgain = await shot();
  const maxSoftness = await page("Number(document.querySelector('[data-control=softness]').max)");
  await setRange("softness", maxSoftness);
  await waitFor(`__showcase.shading(${q(A)}).edgeSoftness > 0`, 2000);
  await sleep(300);
  const soft = await shot();
  const still = await pixelDiff(hard, hardAgain);
  const softened = await pixelDiff(hard, soft);
  await setRange("softness", 0);
  verdict(
    "softness",
    shadingA.shading === "fragment" && still.n === 0 && softened.n > 100,
    `A drawn with ${shadingA.shading} shading; softness 0 twice: ${still.n} pixel(s) differ; softness ${fmt(maxSoftness)}: ${softened.n} of ${softened.total} differ`,
  );

  // ---- colour-from: 0 colours penetration only (the white band ends the range) ----
  // The readouts stay up while a slider moves (no flash to none), also in
  // fast mode, where the depth slider changes the fields' bound.
  await clickOn("[data-control=fast]");
  const kept = [];
  for (const v of [0.8, 0.6, 1]) {
    await setRange("depth", v);
    await sleep(40);
    kept.push(await page("__showcase.collision().readouts !== null"));
  }
  await clickOn("[data-control=fast]");
  await setRange("gap", 0);
  const penOnly = await waitFor(
    collision(`c.coloring[${q(A)}]?.range?.[1] < 0.1 && c.coloring[${q(A)}].status === "ready"`),
    10_000,
  );
  await sleep(100); // the colours of that colouring, committed
  const colorsPen = await hook(`colorCounts(${q(A)})`);
  verdict(
    "colour-from",
    penOnly !== undefined && colorsPen?.gap === 0 && colorsPen?.red > 0 && kept.every((k) => k),
    `range ${q(penOnly?.coloring[A]?.range)}; A's vertices ${q(colorsPen)}; readouts kept while the slider moved: ${kept.join(", ")}`,
  );
  await setRange("gap", 1);
  await waitFor(collision(`c.coloring[${q(A)}]?.range?.[1] === 1 && c.coloring[${q(A)}].status === "ready"`), 10_000);

  // ---- readouts: paused while A is dragged away from B, then refreshed once ----
  // A query held across the drag's start (the test hook holds its result)
  // lands mid-drag however fast or slow the renderer (SwiftShader): it
  // describes the pre-drag pose, so it is dropped. No query starts while
  // the gizmo drags. The mouse is released once the scene has caught up with
  // the last move (the pose is final when the drag ends): the drag's end
  // then refreshes exactly once, to that pose.
  await waitFor(collision("c.settled"), 10_000);
  const before = await hook("collision()");
  const tolerance = await page("(() => { const el = document.querySelector('[data-control=tolerance]'); return { value: Number(el.value), max: Number(el.max) }; })()");
  await hook(`holdReadouts("collision")`);
  await setRange("tolerance", tolerance.value === 0 ? tolerance.max : 0); // a refresh, held
  const held = await waitFor(collision(`c.starts > ${before.starts} && !c.settled`), 10_000);
  let landed;
  let caughtUp;
  const away = await dragGizmo(A, -260, 0, 40, 25, {
    midway: async () => {
      await hook(`releaseReadouts("collision")`);
      landed = await waitFor(collision(`__showcase.dragging() && c.dropped > ${before.dropped}`), 10_000);
    },
    beforeRelease: async () => {
      const ready = `__showcase.dragging() && __showcase.status().status === "idle"
        && c.coloring[${q(A)}]?.status === "ready" && c.coloring[${q(B)}]?.status === "ready"`;
      for (const t = Date.now(); caughtUp === undefined && Date.now() - t < 20_000; ) {
        const first = await waitFor(collision(`(${ready}) && __showcase.translation(${q(A)})`), 20_000);
        await sleep(150);
        const second = await waitFor(collision(`(${ready}) && __showcase.translation(${q(A)})`), 20_000);
        if (first !== undefined && q(first) === q(second)) {
          caughtUp = first;
        }
      }
    },
  });
  await hook(`releaseReadouts("collision")`); // if the drag never got midway
  const apart = await waitFor(
    collision(`!__showcase.dragging() && c.settled && c.readouts !== null && !c.readouts.intersects && c.readouts.gap > 0 && c.readouts.penetration?.depth === 0`),
    10_000,
  );
  // A later refresh would be a second one after the drag.
  await sleep(600);
  const final = await hook("collision()");
  const now = await hook(`pairNow(${q(A)}, ${q(B)})`);
  const startedDuring = final.startsWhileDragging - before.startsWhileDragging;
  const dropped = final.dropped - before.dropped;
  verdict(
    "readouts",
    away.dragging &&
      held !== undefined &&
      landed !== undefined &&
      caughtUp !== undefined &&
      apart !== undefined &&
      startedDuring === 0 &&
      dropped === 1 &&
      final.updatesAfterDrag === 1 &&
      final.settled &&
      final.readouts !== null &&
      final.readouts.gap === now.gap &&
      final.readouts.intersects === now.intersects,
    `${startedDuring} query(ies) started while dragging (paused); ${dropped} result(s) landed mid-drag and dropped${held === undefined ? " (no query held before the drag)" : ""}${caughtUp === undefined ? "; the scene did not catch up before the release" : ""}; ${final.updatesAfterDrag} refresh(es) after the drag; readouts gap ${final.readouts === null ? "none" : `${fmt(final.readouts.gap ?? NaN)}, intersecting ${final.readouts.intersects}`} vs the final pose's gap ${now.gap === null ? "none" : fmt(now.gap)}, intersecting ${now.intersects}`,
  );
  await setRange("tolerance", tolerance.value);
  await clickOn("[data-action=undo]");
  await waitFor(collision(`Math.hypot(...__showcase.translation(${q(A)})) < 1e-9 && c.readouts?.intersects === true`), 10_000);

  // ---- open-target (RF1): C, open, as B (about 1 from A: colour from 2) ----
  await setRange("gap", 2);
  await pick("b", C);
  const open = await waitFor(
    collision(`c.b === ${q(C)}
      && c.coloring[${q(A)}]?.to === ${q(C)} && c.coloring[${q(A)}].status === "ready"
      && c.coloring[${q(C)}]?.to === ${q(A)} && c.coloring[${q(C)}].status === "ready"
      && c.coloring[${q(B)}]?.to === null
      && c.warnings.some((w) => w.startsWith(${q(`${C} is open`)}))`),
    20_000,
  );
  const colorsOpen = { A: await hook(`colorCounts(${q(A)})`), C: await hook(`colorCounts(${q(C)})`) };
  verdict(
    "open-target",
    open !== undefined && colorsOpen.A?.gap > 0 && colorsOpen.C?.gap > 0,
    open === undefined
      ? `collision ${q(await hook("collision()"))}`
      : `A and C coloured (gap-coloured vertices: A ${colorsOpen.A.gap}, C ${colorsOpen.C.gap}); warning "${open.warnings.join(" / ")}"`,
  );

  // ---- repair (RF5): fill C's holes while it is coloured, then undo ----
  await clickOn(`[data-mesh="${C}"] .mesh-name`);
  const analysed = await waitFor(`(() => { const r = __showcase.repair(); return r.analysis?.mesh === ${q(C)} && r; })()`, 10_000);
  const holesBefore = analysed?.analysis.summary.holes.length;
  const componentsBefore = analysed?.analysis.summary.components.length;
  const verticesBefore = used(await hook(`colorCounts(${q(C)})`));
  const analysesBefore = analysed?.analyses ?? 0;
  await clickOn("[data-repair-op=fillHoles]");
  const filled = await waitFor(
    `(() => { const r = __showcase.repair(); const c = __showcase.collision();
      return r.busy === null && r.report?.startsWith("Fill holes: filled") && r.analyses > ${analysesBefore} && r.analysis?.mesh === ${q(C)} && r.analysis.summary.holes.length === 0
        && c.coloring[${q(C)}]?.to === ${q(A)} && c.coloring[${q(C)}].status === "ready"
        && c.readouts !== null && !c.warnings.some((w) => w.startsWith(${q(`${C} is open`)})) && { r, c }; })()`,
    30_000,
  );
  const verticesFilled = used(await hook(`colorCounts(${q(C)})`));
  await clickOn("[data-action=undo]");
  const restored = await waitFor(
    `(() => { const r = __showcase.repair(); const c = __showcase.collision();
      return r.report === null && r.analysis?.mesh === ${q(C)} && r.analysis.summary.holes.length === ${holesBefore ?? 2}
        && c.coloring[${q(C)}]?.to === ${q(A)} && c.coloring[${q(C)}].status === "ready"
        && c.warnings.some((w) => w.startsWith(${q(`${C} is open`)})) && { r, c }; })()`,
    30_000,
  );
  const verticesRestored = used(await hook(`colorCounts(${q(C)})`));
  verdict(
    "repair",
    holesBefore === 2 &&
      componentsBefore === 2 &&
      filled !== undefined &&
      verticesFilled > verticesBefore &&
      restored !== undefined &&
      verticesRestored === verticesBefore,
    `C: ${holesBefore} holes, ${componentsBefore} components; ${filled === undefined ? `after Fill holes ${q(await hook("repair()"))}` : `"${filled.r.report}"`}; coloured vertices ${verticesBefore} -> ${verticesFilled} -> ${verticesRestored} after undo; undo ${restored === undefined ? `did not restore the holes, the colouring and the warning, or clear the report (${q(await hook("repair()"))})` : "restored the holes, the colouring and the warning, and cleared the report"}`,
  );

  // ---- group: A, B and C coloured as a contact group, then C removed ----
  // `to(name, others)`: the mesh is coloured, ready, against exactly `others`.
  const to = (name, others) =>
    `(Array.isArray(c.coloring[${q(name)}]?.to) && c.coloring[${q(name)}].status === "ready"
      && c.coloring[${q(name)}].to.length === ${others.length} && ${q(others)}.every((m) => c.coloring[${q(name)}].to.includes(m)))`;
  const groupState = (extra) =>
    `(() => { const c = __showcase.collision(); const g = __showcase.group(); return c.mode === "group" && (${extra}) && { c, g }; })()`;
  const pairOf = (x, y) => `g.readouts?.pairs.find((p) => p.a === ${q(x)} && p.b === ${q(y)})`;
  await clickOn("[data-mode=group]");
  const grouped = await waitFor(
    groupState(`g.members.length === 3 && g.pairs === "all" && ${to(A, [B, C])} && ${to(B, [A, C])} && ${to(C, [A, B])}
      && g.settled && g.readouts?.pairs.length === 3 && ${pairOf(A, B)}?.intersects && ${pairOf(A, B)}.depth > 0
      && ${pairOf(A, C)}?.gap > 0 && g.warnings.some((w) => w.startsWith(${q(`${C} is open`)}))`),
    30_000,
  );
  await sleep(100); // the colours of those colourings, committed
  const colorsGroup = { A: await hook(`colorCounts(${q(A)})`), B: await hook(`colorCounts(${q(B)})`), C: await hook(`colorCounts(${q(C)})`) };
  // Hover A where it is projected (its front surface): the tip names a partner.
  const overA = await hook(`screenPoint(${q(A)})`);
  let hovered;
  for (const t = Date.now(); hovered === undefined && Date.now() - t < 10_000; ) {
    await mouse("mouseMoved", overA.x + ((Date.now() / 50) % 3), overA.y, { button: "none" });
    hovered = await waitFor(
      `(() => { const h = __showcase.group().hover; return h?.mesh === ${q(A)} && [${q(B)}, ${q(C)}].includes(h.partner) && h; })()`,
      500,
    );
  }
  await mouse("mouseMoved", 5, 5, { button: "none" });
  // The pair matrix: every pair ticked when "All pairs" goes off; untick A-C.
  await clickOn("[data-control=all-pairs]");
  const matrix = await page(`document.querySelectorAll("[data-pair]:checked").length`);
  await clickOn(`[data-pair="${A}|${C}"]`);
  const edited = await waitFor(
    groupState(`Array.isArray(g.pairs) && g.pairs.length === 2 && ${to(A, [B])} && ${to(C, [B])} && ${to(B, [A, C])}
      && g.settled && g.readouts?.pairs.length === 2`),
    20_000,
  );
  await clickOn("[data-control=all-pairs]");
  const restored3 = await waitFor(groupState(`g.pairs === "all" && ${to(A, [B, C])} && ${to(C, [A, B])}`), 20_000);
  const pairsGroup = grouped?.g.readouts.pairs.map((p) => `${p.a.split(" ")[0]}-${p.b.split(" ")[0]} gap ${fmt(p.gap)} hit ${p.intersects} depth ${p.depth === null ? "-" : fmt(p.depth)}`);
  await clickOn(`[data-mesh="${C}"] button.danger`);
  const shrunk = await waitFor(
    groupState(`!(${q(C)} in c.coloring) && g.members.length === 2 && ${to(A, [B])} && ${to(B, [A])}
      && g.settled && g.readouts?.pairs.length === 1 && ${pairOf(A, B)}?.intersects && __showcase.status().status !== "error"`),
    20_000,
  );
  await sleep(100);
  const colorsShrunk = { A: await hook(`colorCounts(${q(A)})`), B: await hook(`colorCounts(${q(B)})`) };
  const groupStatus = await hook("status()");
  verdict(
    "group",
    grouped !== undefined &&
      colorsGroup.A?.red > 0 &&
      colorsGroup.B?.red > 0 &&
      colorsGroup.C?.gap > 0 &&
      hovered !== undefined &&
      matrix === 3 &&
      edited !== undefined &&
      restored3 !== undefined &&
      shrunk !== undefined &&
      colorsShrunk.A?.red > 0 &&
      colorsShrunk.B?.red > 0 &&
      groupStatus.status !== "error",
    `${grouped === undefined ? `group ${q(await hook("group()"))}; collision ${q((await hook("collision()")).coloring)}` : `A, B and C coloured (red/gap vertices: A ${colorsGroup.A?.red}/${colorsGroup.A?.gap}, B ${colorsGroup.B?.red}/${colorsGroup.B?.gap}, C ${colorsGroup.C?.red}/${colorsGroup.C?.gap}); pairs ${pairsGroup.join(", ")}`}; hover over A ${hovered === undefined ? "showed no partner" : `nearest ${hovered.partner} at ${fmt(hovered.distance)}`}; pair matrix ${matrix} ticked, without A-C ${edited === undefined ? q(await hook("group()")) : "A and C against B only"}, all pairs again ${restored3 !== undefined}; after removing C ${shrunk === undefined ? q(await hook("group()")) : `A and B coloured against each other (red vertices ${colorsShrunk.A?.red}, ${colorsShrunk.B?.red}), one pair`}; scene ${groupStatus.status}`,
  );
  await clickOn("[data-mode=pair]");

  // ---- remove (RF3): B removed while A and B are coloured ----
  await pick("b", B);
  const pairAB = await waitFor(
    collision(`c.mode === "pair" && c.coloring[${q(A)}]?.to === ${q(B)} && c.coloring[${q(B)}]?.to === ${q(A)}
      && c.coloring[${q(B)}].status === "ready" && c.readouts?.intersects === true`),
    20_000,
  );
  await clickOn(`[data-mesh="${B}"] button.danger`);
  const removed = await waitFor(
    collision(`c.b === null && c.a === ${q(A)} && !(${q(B)} in c.coloring)
      && c.coloring[${q(A)}]?.to === null && c.readouts === null && __showcase.status().status !== "error"`),
    10_000,
  );
  const settledUpdates = (await hook("collision()")).updates;
  await sleep(1000);
  const after = await hook("collision()");
  const status = await hook("status()");
  verdict(
    "remove",
    pairAB !== undefined && removed !== undefined && after.updates === settledUpdates && after.readouts === null && status.status !== "error",
    `A and B coloured ${pairAB !== undefined}; B's picker ${after.b === null ? "reset" : "kept"}, A's colouring ${after.coloring[A]?.to === null ? "stopped" : "kept"}, readouts ${after.readouts === null && after.updates === settledUpdates ? "stopped" : "still running"}, scene ${status.status}${status.error === undefined ? "" : ` (${status.error})`}`,
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
    `${loaded === undefined ? `items ${JSON.stringify(await hook("items()"))}` : loaded.filter((m) => m.name in expected).map((m) => `${m.name} (${m.triangles})`).join(", ")}; broken.stl ${broken === undefined ? "not reported" : `failed: ${broken.message}`}`,
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

  // ---- local: ?local, the synchronous Scene on the main thread ----
  await browser.goto(`http://127.0.0.1:${port}/?local`, "window.__showcase !== undefined");
  const localDemo = await waitFor(
    `(() => { const i = __showcase.items(); return __showcase.mode === "local" && i.length === 3 && i.every((m) => m.drawn) && i; })()`,
    60_000,
  );
  const localHeat = await waitFor(
    collision(`c.a === ${q(A)} && c.b === ${q(B)} && c.coloring[${q(A)}]?.status === "ready"
      && c.readouts?.penetration?.depth > 0 && c.readouts.intersects`),
    30_000,
  );
  await clickOn("[data-mode=group]");
  const localGroup = await waitFor(
    groupState(`${to(A, [B, C])} && g.settled && g.readouts?.pairs.length === 3 && ${pairOf(A, B)}?.depth > 0`),
    30_000,
  );
  await clickOn("[data-mode=pair]");
  await clickOn(`[data-mesh="${C}"] .mesh-name`);
  const localHoles = await waitFor(
    `(() => { const r = __showcase.repair(); return r.analysis?.mesh === ${q(C)} && r.analysis.summary.holes.length; })()`,
    10_000,
  );
  await clickOn("[data-repair-op=fillHoles]");
  const localFilled = await waitFor(
    `(() => { const r = __showcase.repair(); return r.busy === null && r.analysis?.mesh === ${q(C)} && r.analysis.summary.holes.length === 0; })()`,
    30_000,
  );
  await clickOn("[data-action=undo]");
  const localUndone = await waitFor(
    `(() => { const r = __showcase.repair(); return r.analysis?.mesh === ${q(C)} && r.analysis.summary.holes.length === 2; })()`,
    30_000,
  );
  verdict(
    "local",
    localDemo !== undefined && localHeat !== undefined && localGroup !== undefined && localHoles === 2 && localFilled === true && localUndone === true,
    `local scene ${localDemo !== undefined}; heatmap ${localHeat === undefined ? "not ready" : `penetration ${fmt(localHeat.readouts.penetration.depth)}`}; group ${localGroup === undefined ? "not ready" : `A-B depth ${fmt(localGroup.g.readouts.pairs[0].depth)}`}; C ${localHoles} holes, filled ${localFilled === true}, undone ${localUndone === true}`,
  );

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
