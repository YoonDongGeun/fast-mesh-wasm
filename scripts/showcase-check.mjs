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
// - boot: the CT head loads, selected, its Vessels volume linked into the
//   MPR state and the scene overlay, its surface hidden for the volume.
// - demo: the demo button loads A, B and C, listed and drawn; the scene is
//   idle. (The boot head is removed after, for a pristine A/B/C stage.)
// - glsl: bindMesh's fragment shading run on the GPU (a float render target,
//   read back) for constant quads at a contact heatmap's knots, range ends,
//   +-tolerance, far, -far, NaN and random values, and for built-in maps with
//   a centre and both out-of-range modes, at softness 0 and above, against
//   the library's JavaScript reference (packages/fast-mesh-wasm/src/colormap.ts,
//   loaded with Node's type stripping): every colour within one 16-bit level
//   at softness 0 (the exact path), within four with softness (the filter
//   sums in f32 on the GPU).
// - select: a click on A in the canvas selects it.
// - multiselect: shift-clicking B adds it (the primary stays A), and the
//   Repair panel seeds its Join source from the queue.
// - drag: dragging the gizmo's centre handle moves A (one undo step).
// - undo: the Undo button puts A back; Ctrl+Shift+Z redoes, Ctrl+Z undoes again.
// - heatmap: the collision pair picks A and B, both coloured against
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
//   holds it) lands mid-drag and is dropped. After the drag they settle on
//   the final pose: no longer intersecting, a gap, as queried now. Undo
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
// - drop: a synthetic file drop on the canvas (dragover + drop with a PLY)
//   loads and draws the mesh, like an OS drag-and-drop would.
// - dicom: a synthetic 8-slice sphere series dropped together stacks and
//   meshes its isosurface (Otsu's threshold), listed and drawn.
// - dicom-demo: the Vessel tree phantom button meshes through the same
//   pipeline, listed and drawn.
// - sample: the bundled LIDC2 chest CT button meshes through the same
//   pipeline, listed and drawn (boot covers the headsq button).
// - volume: selecting the phantom's mesh renders its source volume,
//   with a canvas, the Bone preset, and the 3 slice planes.
// - volume-preset: the Vessels preset plus MIP switch the rendering.
// - volume-planes: the Planes checkbox hides and shows the slice planes.
// - volume-slices: the Slices tab renders axial/coronal/sagittal views.
// - volume-overlay: the floating viewport toolbar steers the same preset,
//   tab and Scene switch (the main scene's volume follows the preset),
//   and hides for a mesh without a volume.
// - volume-slice-move: the axial slider and the window preset move the
//   slice views.
// - volume-navigate: clicking the axial view moves the coronal and
//   sagittal slices there.
// - volume-scene: the volume and its slices float in the main scene (the
//   mesh surface hides); the Scene checkbox hides and shows them.
// - scene-drag: dragging an axial corner ball in the scene moves the
//   slice, and the Slices tab follows.
// - volume-source: the source picker switches the viewed volume, and the
//   scene overlay follows.
// - deselect: a tap on empty canvas clears the selection, and the scene
//   volume's mesh surface comes back.
// - sculpt-surface: arming a sculpt tool on a volume mesh shows its
//   surface (quads and volume stand down); disarming restores the volume.
// - remesh-gate: the remesh checkboxes disable past 100 k triangles (the
//   chest) and stay enabled below it (the vessel).
// - big (--big): a generated 1M-triangle binary STL through the file input;
//   the page's longest main-thread task while it loads stays under 50 ms
//   (a PerformanceObserver long-task probe: no `longtask` entry at all),
//   its list picture included (its cost is printed). The frame gaps while
//   zooming are printed with and without redrawing the picture throughout.
// - units-*: on a fresh page, 1 in cubes loaded with "Assume files in" inches
//   span 25.4 once (units-import); the gap and the sliders read in both units
//   and switching moves no stored value (units-collision), the group table,
//   a hole's perimeter and a typed radius too (units-repair); area and volume
//   convert squared and cubed (units-measure), the ruler by the factor
//   (units-ruler); a project round trip keeps the geometry bit-exact, an older
//   file loads on the default (units-project); a DICOM series ignores the
//   import scale (units-dicom); the default survives a reload (units-default).
// - previews: on a fresh page, every built-in model's button shows its committed
//   picture (right size, alt text, not blank, loaded before any click), and each
//   picture matches that model once loaded better than any other model does.
//   Each row's picture has the mesh's silhouette and alt text.
// - previews-edit: a sculpt stroke on A starts no redraw while it runs, and A's
//   picture changes after it; Undo brings the picture back.
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

/** A minimal explicit-VR-LE DICOM slice (8x8 16-bit axial, 1 mm voxels). */
function dicomSlice({ seriesUid, instanceUid, position, pixels }) {
  const el = (into, group, element, vr, value) => {
    into.push(group & 0xff, group >> 8, element & 0xff, element >> 8, vr.charCodeAt(0), vr.charCodeAt(1));
    if (["OB", "OW", "SQ", "UN"].includes(vr)) {
      into.push(0, 0, value.length & 0xff, (value.length >> 8) & 0xff, 0, 0);
    } else {
      into.push(value.length & 0xff, value.length >> 8);
    }
    for (const b of value) into.push(b);
  };
  const str = (s, pad = 0x20) => {
    const raw = [...Buffer.from(s, "latin1")];
    if (raw.length % 2 !== 0) raw.push(pad);
    return raw;
  };
  const us = (v) => [v & 0xff, v >> 8];
  const out = new Array(128).fill(0);
  out.push(0x44, 0x49, 0x43, 0x4d);
  const meta = [];
  el(meta, 0x0002, 0x0001, "OB", [0, 1]);
  el(meta, 0x0002, 0x0002, "UI", str("1.2.840.10008.5.1.4.1.1.2", 0));
  el(meta, 0x0002, 0x0003, "UI", str(instanceUid, 0));
  el(meta, 0x0002, 0x0010, "UI", str("1.2.840.10008.1.2.1", 0));
  out.push(0x02, 0x00, 0x00, 0x00, 0x55, 0x4c, 0x04, 0x00, meta.length & 0xff, (meta.length >> 8) & 0xff, 0, 0, ...meta);
  const bytes = [];
  el(bytes, 0x0008, 0x0018, "UI", str(instanceUid, 0));
  el(bytes, 0x0020, 0x000e, "UI", str(seriesUid, 0));
  el(bytes, 0x0008, 0x103e, "LO", str("CT Ball"));
  el(bytes, 0x0028, 0x0002, "US", us(1));
  el(bytes, 0x0028, 0x0004, "CS", str("MONOCHROME2"));
  el(bytes, 0x0028, 0x0010, "US", us(8));
  el(bytes, 0x0028, 0x0011, "US", us(8));
  el(bytes, 0x0028, 0x0100, "US", us(16));
  el(bytes, 0x0028, 0x0103, "US", us(0));
  el(bytes, 0x0028, 0x0030, "DS", str("1\\1"));
  el(bytes, 0x0020, 0x0032, "DS", str(position.join("\\")));
  el(bytes, 0x0020, 0x0037, "DS", str("1\\0\\0\\0\\1\\0"));
  const raw = [];
  for (const p of pixels) raw.push(p & 0xff, (p >> 8) & 0xff);
  el(bytes, 0x7fe0, 0x0010, "OW", raw);
  return new Uint8Array([...out, ...bytes]);
}

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
async function dragGizmo(name, dx, dy, steps, ms, midway = async () => {}) {
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

  // ---- boot: the CT head, selected, its Vessels volume linked, its surface hidden ----
  const boot = await waitFor(
    `(() => { const v = __showcase.items().find((m) => m.name.includes("headsq") && m.drawn);
      return v !== undefined && __showcase.mpr()?.volumeId === v.volumeId && __showcase.sceneSlices().shown === 3 &&
        __showcase.sceneVolume().preset === "Vessels" && __showcase.volume().preset === "Vessels" && v.visible === false && v; })()`,
    90_000,
  );
  verdict(
    "boot",
    boot !== undefined,
    boot === undefined ? `items ${JSON.stringify(await hook("items()"))}` : `${boot.name} (${boot.triangles})`,
  );

  // ---- demo: the demo button loads A, B and C ----
  await clickOn("[data-action=demo-meshes]");
  const demo = await waitFor(
    `(() => { const want = [${q(A)}, ${q(B)}, ${q(C)}];
      const got = want.map((n) => __showcase.items().find((m) => m.name === n)).filter((m) => m !== undefined && m.drawn);
      return got.length === 3 && __showcase.status().status === "idle" && got; })()`,
    60_000,
  );
  verdict(
    "demo",
    demo !== undefined && demo.map((m) => m.name[0]).join("") === "ABC",
    demo === undefined ? `items ${JSON.stringify(await hook("items()"))}` : demo.map((m) => `${m.name} (${m.triangles})`).join(", "),
  );

  // Pristine A/B/C stage: select A through the list, remove the boot
  // head, and restore the default camera (boot framed the head).
  await clickOn(`[data-mesh=${JSON.stringify(A)}] .mesh-name`);
  if (boot !== undefined) {
    await clickOn(`[data-mesh=${JSON.stringify(boot.name)}] button.danger`);
    await waitFor(`!__showcase.items().some((m) => m.name === ${JSON.stringify(boot.name)})`, 10_000);
  }
  await clickOn("[data-action=view-reset]");
  await sleep(300);

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

  // ---- multiselect: shift-click B in the canvas ----
  const atB = await hook(`screenPoint(${JSON.stringify(B)})`);
  await mouse("mouseMoved", atB.x, atB.y, { button: "none", modifiers: 8 });
  await mouse("mousePressed", atB.x, atB.y, { buttons: 1, modifiers: 8 });
  await mouse("mouseReleased", atB.x, atB.y, { buttons: 0, modifiers: 8 });
  // The queue hint and the seeded source render in the passive effect after
  // the multi list lands: wait for all three, not just the list.
  const multi = await waitFor(
    `JSON.stringify(__showcase.multi()) === ${JSON.stringify(JSON.stringify([A, B]))} && (document.querySelector("[data-join-source]")?.value ?? "") !== ""`,
    5000,
  );
  const queued = await page(`document.querySelector("[data-join-queue]")?.textContent ?? null`);
  const seeded = await page(`document.querySelector("[data-join-source]")?.value ?? ""`);
  verdict(
    "multiselect",
    multi !== undefined && queued !== null && queued.includes(B) && seeded !== "",
    `multi ${JSON.stringify(await hook("multi()"))}; queue ${JSON.stringify(queued)}; source ${seeded}`,
  );

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
  await pick("a", A);
  await pick("b", B);
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

  // ---- readouts: paused while A is dragged away from B, then refreshed ----
  // A query held across the drag's start (the test hook holds its result)
  // lands mid-drag however fast or slow the renderer (SwiftShader): it
  // describes the pre-drag pose, so it is dropped. No query starts while
  // the gizmo drags; after it the readouts settle on the final pose (the
  // gap and intersecting as queried now). How many refreshes that takes
  // varies: the last moves' change records may land after the release.
  await waitFor(collision("c.settled"), 10_000);
  const before = await hook("collision()");
  const tolerance = await page("(() => { const el = document.querySelector('[data-control=tolerance]'); return { value: Number(el.value), max: Number(el.max) }; })()");
  await hook(`holdReadouts("collision")`);
  await setRange("tolerance", tolerance.value === 0 ? tolerance.max : 0); // a refresh, held
  const held = await waitFor(collision(`c.starts > ${before.starts} && !c.settled`), 10_000);
  let landed;
  const away = await dragGizmo(A, -260, 0, 40, 25, async () => {
    await hook(`releaseReadouts("collision")`);
    landed = await waitFor(collision(`__showcase.dragging() && c.dropped > ${before.dropped}`), 10_000);
  });
  await hook(`releaseReadouts("collision")`); // if the drag never got midway
  // Settled on readouts that match the pose as queried now, with no refresh
  // landing meanwhile (the pose no longer changes once they match).
  const apart = await waitFor(
    `(async () => {
      const c = __showcase.collision();
      const now = await __showcase.pairNow(${q(A)}, ${q(B)});
      const r = c.readouts;
      return !__showcase.dragging() && c.settled && __showcase.collision().updates === c.updates && r !== null
        && !r.intersects && r.gap > 0 && r.penetration?.depth === 0
        && r.gap === now.gap && r.intersects === now.intersects && { c, now };
    })()`,
    10_000,
  );
  const final = apart?.c ?? (await hook("collision()"));
  const now = apart?.now ?? (await hook(`pairNow(${q(A)}, ${q(B)})`));
  const startedDuring = final.startsWhileDragging - before.startsWhileDragging;
  const dropped = final.dropped - before.dropped;
  verdict(
    "readouts",
    away.dragging &&
      held !== undefined &&
      landed !== undefined &&
      apart !== undefined &&
      startedDuring === 0 &&
      dropped >= 1 &&
      final.updatesAfterDrag >= 1,
    `${startedDuring} query(ies) started while dragging (paused); ${dropped} result(s) landed mid-drag and dropped${held === undefined ? " (no query held before the drag)" : ""}; ${final.updatesAfterDrag} refresh(es) after the drag; readouts ${final.readouts === null ? "none" : `gap ${fmt(final.readouts.gap ?? NaN)}, intersecting ${final.readouts.intersects}`}${apart === undefined ? " (not settled on the final pose)" : ""} vs the final pose's gap ${now.gap === null ? "none" : fmt(now.gap)}, intersecting ${now.intersects}`,
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
  // Members are A and B (the boot head left): tick C.
  await waitFor(`document.querySelector('[data-group-member=${JSON.stringify(C)}]') !== null`, 10_000);
  await page(`document.querySelector('[data-group-member=${JSON.stringify(C)}]').click()`);
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

  // ---- drop: a synthetic file drop on the canvas loads ----
  const dropName = "dropped.ply";
  await page(`(() => {
    const file = new File([${JSON.stringify(TETRA_PLY)}], ${JSON.stringify(dropName)}, { type: "model/ply" });
    const dt = new DataTransfer();
    dt.items.add(file);
    const canvas = document.querySelector("canvas");
    const r = canvas.getBoundingClientRect();
    const opts = { bubbles: true, cancelable: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, dataTransfer: dt };
    canvas.dispatchEvent(new DragEvent("dragover", opts));
    canvas.dispatchEvent(new DragEvent("drop", opts));
  })()`);
  const dropLoaded = await waitFor(
    `__showcase.items().find((m) => m.name === ${JSON.stringify(dropName)} && m.drawn)`,
    20_000,
  );
  verdict(
    "drop",
    dropLoaded !== undefined,
    dropLoaded === undefined ? `items ${JSON.stringify(await hook("items()"))}` : `${dropName} drawn`,
  );

  // ---- dicom: a synthetic sphere series dropped together meshes ----
  const series = [];
  for (let z = 0; z < 8; z++) {
    const pixels = [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        pixels.push(Math.hypot(x - 3.5, y - 3.5, z - 3.5) <= 2.5 ? 100 : 0);
      }
    }
    series.push({
      name: `ct${z}.dcm`,
      bytes: dicomSlice({ seriesUid: "1.9.9", instanceUid: `1.9.9.${z}`, position: [0, 0, z], pixels }),
    });
  }
  await page(`(() => {
    const dt = new DataTransfer();
    for (const f of ${JSON.stringify(series.map((s) => ({ name: s.name, bytes: [...s.bytes] })))}) {
      dt.items.add(new File([new Uint8Array(f.bytes).buffer], f.name));
    }
    const canvas = document.querySelector("canvas");
    const r = canvas.getBoundingClientRect();
    const opts = { bubbles: true, cancelable: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, dataTransfer: dt };
    canvas.dispatchEvent(new DragEvent("dragover", opts));
    canvas.dispatchEvent(new DragEvent("drop", opts));
  })()`);
  const iso = await waitFor(`__showcase.items().find((m) => m.name.includes("(iso ") && m.drawn)`, 20_000);
  verdict(
    "dicom",
    iso !== undefined && iso.triangles > 0,
    iso === undefined ? `items ${JSON.stringify(await hook("items()"))}` : `${iso.name} (${iso.triangles})`,
  );

  // ---- dicom-demo: the Vessel tree phantom meshes ----
  await clickOn("[data-action=dicom-demo-vessels]");
  const vessel = await waitFor(
    `__showcase.items().find((m) => m.name.startsWith("Vessel tree") && m.drawn)`,
    60_000,
  );
  verdict(
    "dicom-demo",
    vessel !== undefined && vessel.triangles > 1000,
    vessel === undefined ? `items ${JSON.stringify(await hook("items()"))}` : `${vessel.name} (${vessel.triangles})`,
  );

  // ---- sample: the bundled LIDC2 chest CT meshes (boot covers headsq) ----
  await clickOn("[data-action=sample-lidc2]");
  const chest = await waitFor(
    `__showcase.items().find((m) => m.name.includes("LIDC2") && m.drawn)`,
    90_000,
  );
  verdict(
    "sample",
    chest !== undefined && chest.triangles > 100_000,
    `chest ${chest === undefined ? "missing" : chest.triangles}`,
  );

  // ---- volume: the phantom's source volume renders on selection ----
  if (vessel !== undefined) {
    await clickOn(`[data-mesh="${vessel.name}"] .mesh-name`);
  }
  const vol = await waitFor(
    `__showcase.volume().ready && __showcase.volume().planes === 3 ? __showcase.volume() : null`,
    60_000,
  );
  const volCanvas = await page(`document.querySelector("[data-volume] canvas") !== null`);
  const volMeshHidden =
    vessel === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(vessel.name)})?.visible === false ? true : null`,
          20_000,
        );
  verdict(
    "volume",
    vol !== undefined && vol.preset === "Bone" && vol.mip === false && volCanvas === true && volMeshHidden === true,
    vol === undefined
      ? "never ready"
      : `${vol.preset} canvas=${volCanvas} planes=${vol.planes} meshHidden=${volMeshHidden === true}`,
  );

  // ---- volume-preset: Vessels + MIP ----
  await page(`(() => {
    const sel = document.querySelector('[data-action="volume-preset"]');
    sel.value = "2";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    document.querySelector('[data-action="volume-mip"]').click();
    return true;
  })()`);
  const volMip = await waitFor(
    `__showcase.volume().preset === "Vessels" && __showcase.volume().mip ? __showcase.volume() : null`,
    20_000,
  );
  verdict(
    "volume-preset",
    volMip !== undefined && volMip.ready === true,
    volMip === undefined ? JSON.stringify(await hook("volume()")) : `${volMip.preset} mip=${volMip.mip}`,
  );

  // ---- volume-planes: the checkbox hides and shows the slice planes ----
  await clickOn('[data-action="volume-planes"]');
  const planesOff = await waitFor(`__showcase.volume().planes === 0 ? __showcase.volume() : null`, 20_000);
  await clickOn('[data-action="volume-planes"]');
  const planesOn = await waitFor(`__showcase.volume().planes === 3 ? __showcase.volume() : null`, 20_000);
  verdict(
    "volume-planes",
    planesOff !== undefined && planesOn !== undefined,
    `off=${planesOff !== undefined} on=${planesOn !== undefined}`,
  );

  // ---- volume-slices: axial/coronal/sagittal views render ----
  await clickOn('[data-action="volume-tab-slices"]');
  const slices = await waitFor(`__showcase.slices().ready ? __showcase.slices() : null`, 60_000);
  const sliceCanvases = await page(`document.querySelectorAll("[data-slice] canvas").length`);
  verdict(
    "volume-slices",
    slices !== undefined && sliceCanvases === 3,
    slices === undefined ? "never ready" : `canvases=${sliceCanvases} axial=${slices.axial}`,
  );

  // ---- volume-overlay: the floating toolbar steers the shared MPR state ----
  // The vessel stays selected with its slices showing: the overlay offers
  // the same preset, tab and Scene switch as the panel.
  const overlayShown = await page(
    `document.querySelector('[data-action="overlay-volume-tab-slices"]')?.getAttribute("aria-selected") ?? null`,
  );
  await clickOn('[data-action="overlay-volume-tab-3d"]');
  const overlay3d = await waitFor(
    `__showcase.mpr()?.tab === "volume" ? document.querySelector('[data-action="volume-tab-3d"]').getAttribute("aria-selected") : null`,
    20_000,
  );
  await page(`(() => {
    const sel = document.querySelector('[data-action="overlay-volume-preset"]');
    sel.value = "0";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  const overlayBone = await waitFor(`__showcase.volume().preset === "Bone" ? true : null`, 20_000);
  const overlaySceneBone = await waitFor(`__showcase.sceneVolume().preset === "Bone" ? true : null`, 20_000);
  await page(`(() => {
    const sel = document.querySelector('[data-action="overlay-volume-preset"]');
    sel.value = "2";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  const overlayVessels = await waitFor(`__showcase.volume().preset === "Vessels" ? true : null`, 20_000);
  const overlaySceneVessels = await waitFor(`__showcase.sceneVolume().preset === "Vessels" ? true : null`, 20_000);
  await clickOn('[data-action="overlay-volume-scene"]');
  const overlaySceneOff = await waitFor(`__showcase.sceneSlices().shown === 0 ? true : null`, 20_000);
  await clickOn('[data-action="overlay-volume-scene"]');
  const overlaySceneOn = await waitFor(`__showcase.sceneSlices().shown === 3 ? true : null`, 20_000);
  await clickOn('[data-action="overlay-volume-tab-slices"]');
  const overlayBack = await waitFor(
    `__showcase.mpr()?.tab === "slices" && __showcase.slices().ready ? true : null`,
    60_000,
  );
  // The overlay only shows for a volume mesh: mesh A hides it, and
  // reselecting the vessel brings it back (remounted on the 3D tab).
  await clickOn(`[data-mesh=${JSON.stringify(A)}] .mesh-name`);
  const overlayHidden = await waitFor(
    `document.querySelector('[data-action="overlay-volume-tab-slices"]') === null ? true : null`,
    20_000,
  );
  const overlaySceneHidden = await waitFor(`__showcase.sceneVolume().id === null ? true : null`, 20_000);
  if (vessel !== undefined) {
    await clickOn(`[data-mesh="${vessel.name}"] .mesh-name`);
  }
  const overlayRestored =
    vessel === undefined
      ? undefined
      : await waitFor(
          `__showcase.mpr()?.volumeId === ${JSON.stringify(vessel.volumeId)} && document.querySelector('[data-action="overlay-volume-tab-slices"]') !== null ? true : null`,
          20_000,
        );
  const overlaySceneRestored =
    vessel === undefined
      ? undefined
      : await waitFor(`__showcase.sceneVolume().id === ${JSON.stringify(vessel.volumeId)} ? true : null`, 20_000);
  await clickOn('[data-action="volume-tab-slices"]');
  const slicesAgain = await waitFor(`__showcase.slices().ready ? true : null`, 60_000);
  verdict(
    "volume-overlay",
    vessel !== undefined &&
      overlayShown === "true" &&
      overlay3d === "true" &&
      overlayBone === true &&
      overlayVessels === true &&
      overlaySceneBone === true &&
      overlaySceneVessels === true &&
      overlaySceneOff === true &&
      overlaySceneOn === true &&
      overlayBack === true &&
      overlayHidden === true &&
      overlaySceneHidden === true &&
      overlayRestored === true &&
      overlaySceneRestored === true &&
      slicesAgain === true,
    `shown=${overlayShown} tab3d=${overlay3d} bone=${overlayBone}/${overlaySceneBone} vessels=${overlayVessels}/${overlaySceneVessels} scene=${overlaySceneOff}/${overlaySceneOn} back=${overlayBack} hidden=${overlayHidden}/${overlaySceneHidden} restored=${overlayRestored}/${overlaySceneRestored} slices=${slicesAgain}`,
  );

  // ---- volume-slice-move: the slider and window preset steer the views ----
  await page(`(() => {
    const el = document.querySelector('[data-action="slice-axial"]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, "10");
    el.dispatchEvent(new Event("input", { bubbles: true }));
    const w = document.querySelector('[data-action="volume-window"]');
    w.value = "3";
    w.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  const sliceMoved = await waitFor(
    `__showcase.slices().axial === 10 && __showcase.slices().window === "Full range" ? __showcase.slices() : null`,
    20_000,
  );
  verdict(
    "volume-slice-move",
    sliceMoved !== undefined && sliceMoved.ready === true,
    sliceMoved === undefined
      ? JSON.stringify(await hook("slices()"))
      : `axial=${sliceMoved.axial} window=${sliceMoved.window}`,
  );

  // ---- volume-navigate: clicking the axial view moves the other slices ----
  // Off-centre (the exact centre sits on a texel boundary): the coronal and
  // sagittal slices jump there while the axial one stays.
  await page(`document.querySelector('[data-slice="axial"] canvas').scrollIntoView({ block: "center" })`);
  const spot = await page(`(() => {
    const r = document.querySelector('[data-slice="axial"] canvas').getBoundingClientRect();
    return { x: r.left + r.width * 0.6, y: r.top + r.height * 0.4 };
  })()`);
  await click(spot.x, spot.y);
  const nav = await waitFor(
    `__showcase.slices().axial === 10 && __showcase.slices().sagittal !== 47 && __showcase.slices().coronal !== 47 ? __showcase.slices() : null`,
    20_000,
  );
  verdict(
    "volume-navigate",
    nav !== undefined,
    nav === undefined
      ? JSON.stringify(await hook("slices()"))
      : `axial=${nav.axial} sagittal=${nav.sagittal} coronal=${nav.coronal}`,
  );

  // ---- volume-scene: the volume and its slices float in the main scene ----
  const inScene = await waitFor(`__showcase.sceneSlices().shown === 3 ? __showcase.sceneSlices() : null`, 20_000);
  await clickOn('[data-action="volume-scene"]');
  const sceneOff = await waitFor(`__showcase.sceneSlices().shown === 0 ? __showcase.sceneSlices() : null`, 20_000);
  const sceneMeshBack =
    vessel === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(vessel.name)})?.visible === true ? true : null`,
          20_000,
        );
  await clickOn('[data-action="volume-scene"]');
  const sceneOn = await waitFor(`__showcase.sceneSlices().shown === 3 ? __showcase.sceneSlices() : null`, 20_000);
  const sceneMeshHidden =
    vessel === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(vessel.name)})?.visible === false ? true : null`,
          20_000,
        );
  verdict(
    "volume-scene",
    inScene !== undefined &&
      sceneOff !== undefined &&
      sceneOn !== undefined &&
      sceneMeshBack === true &&
      sceneMeshHidden === true,
    `show=${inScene !== undefined} off=${sceneOff !== undefined} on=${sceneOn !== undefined} meshBack=${sceneMeshBack === true} meshHidden=${sceneMeshHidden === true}`,
  );

  // ---- scene-drag: an axial corner ball drags the slice through the stack ----
  const handles = await waitFor(`__showcase.sliceHandles()?.axial ?? null`, 20_000);
  const axialBefore = await hook(`mpr().slices.axial`);
  if (handles !== undefined) {
    const len = Math.hypot(handles.dx, handles.dy) || 1;
    const ux = handles.dx / len;
    const uy = handles.dy / len;
    await mouse("mouseMoved", handles.x, handles.y, { button: "none" });
    await mouse("mousePressed", handles.x, handles.y, { buttons: 1 });
    for (let i = 1; i <= 10; i++) {
      await mouse("mouseMoved", handles.x + (ux * 120 * i) / 10, handles.y + (uy * 120 * i) / 10, { buttons: 1 });
      await sleep(40);
    }
    await mouse("mouseReleased", handles.x + ux * 120, handles.y + uy * 120, { buttons: 0 });
  }
  const axialAfter = await hook(`mpr().slices.axial`);
  const slicesFollow = await waitFor(`__showcase.slices().axial === ${axialAfter} ? true : null`, 20_000);
  verdict(
    "scene-drag",
    handles !== undefined && axialAfter > axialBefore && slicesFollow === true,
    `axial ${axialBefore} -> ${axialAfter} (slices tab follows: ${slicesFollow === true})`,
  );

  // ---- volume-source: the source picker switches the viewed volume ----
  const chestItem = await hook(`items().find((m) => m.name.includes("LIDC2"))`);
  if (chestItem !== undefined) {
    await page(`(() => {
      const sel = document.querySelector('[data-action="volume-source"]');
      sel.value = ${JSON.stringify(chestItem.volumeId)};
      sel.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    })()`);
  }
  const switched =
    chestItem === undefined
      ? undefined
      : await waitFor(`__showcase.mpr()?.volumeId === ${JSON.stringify(chestItem.volumeId)} ? __showcase.mpr() : null`, 20_000);
  const overlayFollows =
    chestItem === undefined
      ? undefined
      : await waitFor(`__showcase.sceneSlices().id === ${JSON.stringify(chestItem.volumeId)}`, 20_000);
  verdict(
    "volume-source",
    switched !== undefined && overlayFollows !== undefined,
    `mpr=${switched?.volumeId} overlay=${overlayFollows !== undefined}`,
  );

  // ---- deselect: a tap on empty space clears the selection ----
  // The chest is selected (its surface hides for the scene volume):
  // tapping where no mesh is brings the surface back. A few canvas
  // spots, until one misses (a wrong tap just selects a mesh).
  const viewBox = await page(`(() => {
    const r = document.querySelector("main canvas").getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  })()`);
  const taps = [
    [viewBox.x + viewBox.width - 25, viewBox.y + viewBox.height * 0.55],
    [viewBox.x + viewBox.width / 2, viewBox.y + viewBox.height - 25],
    [viewBox.x + viewBox.width / 2, viewBox.y + 70],
    [viewBox.x + viewBox.width - 25, viewBox.y + viewBox.height - 25],
  ];
  let deselected = await hook(`selected() === null`);
  for (const [tx, ty] of taps) {
    if (deselected === true) {
      break;
    }
    await click(tx, ty);
    deselected = await waitFor(`__showcase.selected() === null ? true : null`, 5_000);
  }
  const chestBack =
    chestItem === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(chestItem.name)})?.visible === true ? true : null`,
          20_000,
        );
  verdict(
    "deselect",
    deselected === true && chestBack === true,
    `selected=null:${deselected === true} chest mesh back:${chestBack === true}`,
  );

  // ---- sculpt-surface: arming sculpt shows the surface, not the volume ----
  // Reselect the chest (volume mode hides its surface), arm Inflate (the
  // surface comes back for the brush, quads and volume stand down), then
  // disarm (volume mode returns).
  if (chestItem !== undefined) {
    await clickOn(`[data-mesh=${JSON.stringify(chestItem.name)}] .mesh-name`);
  }
  const sculptVol =
    chestItem === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(chestItem.name)})?.visible === false && __showcase.sceneVolume().id === ${JSON.stringify(chestItem.volumeId)} ? true : null`,
          20_000,
        );
  await clickOn('[data-action="sculpt-tool-inflate"]');
  const sculptArmed = await waitFor(
    `document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true" ? true : null`,
    20_000,
  );
  const sculptSurface =
    chestItem === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(chestItem.name)})?.visible === true && __showcase.sceneVolume().id === null && __showcase.sceneSlices().shown === 0 ? true : null`,
          20_000,
        );
  await clickOn('[data-action="sculpt-tool-inflate"]');
  const sculptBack =
    chestItem === undefined
      ? undefined
      : await waitFor(
          `__showcase.items().find((m) => m.name === ${JSON.stringify(chestItem.name)})?.visible === false && __showcase.sceneVolume().id === ${JSON.stringify(chestItem.volumeId)} ? true : null`,
          20_000,
        );
  verdict(
    "sculpt-surface",
    chestItem !== undefined &&
      sculptVol === true &&
      sculptArmed === true &&
      sculptSurface === true &&
      sculptBack === true,
    `volume=${sculptVol === true} armed=${sculptArmed === true} surface=${sculptSurface === true} back=${sculptBack === true}`,
  );

  // ---- remesh-gate: the optional remesh stands down on huge meshes ----
  // The chest (591 k) disables it but keeps the stretch splits, whose cost
  // follows the stretched edges; the vessel (27 k) keeps both enabled.
  const gateOff = await waitFor(
    `document.querySelector('[data-action="sculpt-remesh"]')?.disabled === true && document.querySelector('[data-action="sculpt-live-remesh"]')?.disabled === false ? true : null`,
    20_000,
  );
  if (vessel !== undefined) {
    await clickOn(`[data-mesh=${JSON.stringify(vessel.name)}] .mesh-name`);
  }
  const gateOn =
    vessel === undefined
      ? undefined
      : await waitFor(
          `document.querySelector('[data-action="sculpt-remesh"]')?.disabled === false && document.querySelector('[data-action="sculpt-live-remesh"]')?.disabled === false ? true : null`,
          20_000,
        );
  verdict(
    "remesh-gate",
    gateOff === true && gateOn === true,
    `chest disabled=${gateOff === true} vessel enabled=${gateOn === true}`,
  );

  // ---- stretch-stroke: a real grab stroke splits what it stretched, nothing else ----
  // A demo mesh's buffers before and after a long drag on the canvas, compared in the page.
  let stretch = null;
  const stroked = (await hook("items()")).find((m) => m.volumeId === null && m.triangles > 10_000);
  if (stroked !== undefined) {
    const name = JSON.stringify(stroked.name);
    await clickOn(`[data-mesh=${name}] .mesh-name`);
    await clickOn('[data-action="sculpt-tool-grab"]');
    await waitFor(`document.querySelector('[data-action="sculpt-tool-grab"]')?.getAttribute("aria-pressed") === "true" ? true : null`, 5000);
    await page(`(() => { window.__before = { p: __showcase.positions(${name}), i: __showcase.indices(${name}), s: __showcase.sculpt() }; })()`);
    // Hover until the brush ball shows: then the pointer is on the mesh.
    const center = await hook(`screenPoint(${name})`);
    const box = await page(`(() => { const r = document.querySelector("main canvas").getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; })()`);
    const tries = [center];
    for (let row = 1; row < 16; row++) {
      for (let col = 1; col < 16; col++) {
        tries.push({ x: box.x + (box.w * col) / 16, y: box.y + (box.h * row) / 16 });
      }
    }
    let at = null;
    for (const t of tries) {
      await mouse("mouseMoved", t.x, t.y, { button: "none" });
      await new Promise((resolve) => setTimeout(resolve, 120));
      if ((await hook("sculpt().hovering")) === true) {
        at = t;
        break;
      }
    }
    if (at !== null) {
      await mouse("mousePressed", at.x, at.y);
      for (let k = 1; k <= 60; k++) {
        await mouse("mouseMoved", at.x, at.y - k * 4);
        await new Promise((resolve) => setTimeout(resolve, 30));
      }
      await mouse("mouseReleased", at.x, at.y - 240);
      await new Promise((resolve) => setTimeout(resolve, 1500));
      stretch = await page(`(() => {
        const { p, i, s } = window.__before;
        const q = __showcase.positions(${name});
        const j = __showcase.indices(${name});
        const now = __showcase.sculpt();
        const moved = (v) => p[3 * v] !== q[3 * v] || p[3 * v + 1] !== q[3 * v + 1] || p[3 * v + 2] !== q[3 * v + 2];
        let movedCount = 0;
        for (let v = 0; v < p.length / 3; v++) movedCount += moved(v) ? 1 : 0;
        let unmoved = 0;
        let rewritten = 0;
        for (let t = 0; t < i.length; t += 3) {
          if (!moved(i[t]) && !moved(i[t + 1]) && !moved(i[t + 2])) {
            unmoved++;
            if (j[t] !== i[t] || j[t + 1] !== i[t + 1] || j[t + 2] !== i[t + 2]) rewritten++;
          }
        }
        return { slotsBefore: p.length / 3, slotsAfter: q.length / 3, trisBefore: i.length / 3, trisAfter: j.length / 3, movedCount, unmoved, rewritten, passes: now.liveRemeshes - s.liveRemeshes, remeshes: now.remeshes - s.remeshes, dabs: now.dabs - s.dabs };
      })()`);
    }
    await clickOn('[data-action="sculpt-tool-grab"]');
  }
  verdict(
    "stretch-stroke",
    stretch !== null &&
      stretch.dabs > 0 &&
      stretch.passes > 0 &&
      stretch.remeshes === 0 &&
      stretch.rewritten === 0 &&
      stretch.unmoved > 0 &&
      stretch.trisAfter > stretch.trisBefore,
    JSON.stringify(stretch),
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
    // Measured at 1M: left on, the load would simplify the mesh first.
    await page(`(() => { const box = document.querySelector("[data-action=decimate-on-load]"); if (box.checked) box.click(); })()`);
    await sleep(500); // a quiet baseline before the upload
    await page("(__probe.start = performance.now(), __probe.maxGap = 0, true)");
    await upload([path]);
    const big = await waitFor(
      `(() => { const m = __showcase.items().find((i) => i.name === "big-sphere.stl");
        return m !== undefined && m.drawn && __showcase.jobs().every((j) => j.name !== "big-sphere.stl" || j.phase === "done") && performance.now() - __probe.start; })()`,
      180_000,
    );
    await sleep(1000); // a few more frames with the mesh drawn
    const bigPreview = await waitFor(`(() => { const s = __showcase.meshPreview("big-sphere.stl"); return s.triangles > 0 && s; })()`, 30_000);
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
      `  picture of ${bigPreview?.triangles} triangles: ${fmt(bigPreview?.last?.wallMs ?? NaN)} ms wall, ${fmt(bigPreview?.last?.busyMs ?? NaN)} ms on the main thread in ${bigPreview?.last?.slices} slices, longest ${fmt(bigPreview?.last?.longestSliceMs ?? NaN)} ms`,
    );
    // Frame gaps while the viewport redraws (a wheel zoom: no press, under which
    // pictures wait), with and without the picture redrawing throughout.
    const zoomGaps = async (redraw) => {
      const r = await page(`(() => { const b = document.querySelector(".viewport canvas").getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; })()`);
      await page(`(() => {
        const g = { gaps: [], last: performance.now(), on: true };
        const f = (t) => { g.gaps.push(t - g.last); g.last = t; if (g.on) requestAnimationFrame(f); };
        requestAnimationFrame(f);
        window.__gaps = g;
        ${redraw ? `g.timer = setInterval(() => __showcase.redrawMeshPreview("big-sphere.stl"), 200);` : ""}
      })()`);
      for (let i = 1; i <= 120; i++) {
        await browser.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: r.x, y: r.y, deltaX: 0, deltaY: i % 40 < 20 ? 20 : -20 });
        await sleep(16);
      }
      const gaps = await page(`(() => { const g = window.__gaps; g.on = false; clearInterval(g.timer); return g.gaps.slice(1); })()`);
      const sorted = [...gaps].sort((a, b) => a - b);
      return { frames: gaps.length, median: sorted[gaps.length >> 1], p95: sorted[Math.floor(gaps.length * 0.95)], longest: sorted.at(-1), over33: gaps.filter((g) => g > 33.4).length };
    };
    const quiet = await zoomGaps(false);
    const before = (await hook(`meshPreview("big-sphere.stl")`)).renders;
    const busy = await zoomGaps(true);
    await sleep(600);
    const after = await hook(`meshPreview("big-sphere.stl")`);
    const gapText = (g) => `${g.frames} frames, median ${fmt(g.median)} ms, p95 ${fmt(g.p95)} ms, longest ${fmt(g.longest)} ms, ${g.over33} over 33 ms`;
    console.log(`  zoom without picture redraws: ${gapText(quiet)}`);
    console.log(`  zoom with ${after.renders - before} picture redraws (last: ${fmt(after.last.busyMs)} ms busy, longest slice ${fmt(after.last.longestSliceMs)} ms): ${gapText(busy)}`);
    console.log(
      `  loaded in ${big === undefined ? "-" : `${fmt(big)} ms`}; ${probe.tasks.length} long task(s)${probe.tasks.length > 0 ? ` (longest ${fmt(longest)} ms: ${probe.tasks.map((t) => fmt(t.duration)).join(", ")})` : ""}; longest frame gap ${fmt(probe.maxGap)} ms`,
    );
    verdict(
      "big",
      big !== undefined && probe.triangles === count && probe.tasks.length === 0 && calibrated && bigPreview !== undefined && bigPreview.triangles === count,
      `${count.toLocaleString("en")} triangles; longest main-thread task ${probe.tasks.length === 0 ? "under 50 ms (no long task)" : `${fmt(longest)} ms`}; the probe sees an 80 ms task: ${calibrated}`,
    );
  }

  if (option("--screenshot", undefined) !== undefined) {
    const { data } = await browser.send("Page.captureScreenshot", { format: "png" });
    writeFileSync(option("--screenshot"), Buffer.from(data, "base64"));
  }

  // ---- units: the display unit changes text, the import scale the geometry, once ----
  {
  // A fresh page: 1 in cubes loaded "as inches", a known gap, known area and
  // volume. Every expected string is written out: 1 in = 25.4 mm,
  // 1 in² = 645.16 mm², 1 in³ = 16387.064 mm³.
  await browser.goto(`http://127.0.0.1:${port}/`, "window.__showcase !== undefined");
  const bootHead = await waitFor(`__showcase.items().find((m) => m.name.includes("headsq") && m.drawn)?.name`, 60_000);
  await clickOn(`[data-mesh=${q(bootHead)}] button.danger`);
  await waitFor(`__showcase.items().length === 0`, 10_000);
  const unitCube = ([x, y, z], open = false) => {
    const v = (a, b, c) => [x + a, y + b, z + c];
    const quad = (a, b, c, d) => [[...a, ...b, ...c], [...a, ...c, ...d]];
    const [p000, p100, p110, p010, p001, p101, p111, p011] = [
      v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0), v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1),
    ];
    return [
      ...quad(p000, p010, p110, p100),
      ...(open ? [] : quad(p001, p101, p111, p011)),
      ...quad(p000, p100, p101, p001),
      ...quad(p100, p110, p111, p101),
      ...quad(p110, p010, p011, p111),
      ...quad(p010, p000, p001, p011),
    ];
  };
  const setSelect = (control, value) =>
    page(`(() => {
      const el = document.querySelector('[data-control=${control}]');
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set.call(el, ${q(value)});
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    })()`);
  const textOf = (selector) => page(`document.querySelector(${q(selector)})?.textContent ?? null`);
  const waitText = (selector, part, ms = 20_000) =>
    waitFor(`document.querySelector(${q(selector)})?.textContent?.includes(${q(part)})`, ms);
  const positionsOf = (name) => `__showcase.positions(${q(name)})`;
  // The views hold f32 positions about the mesh's origin: the span is the size.
  const spanOf = (name) => `(() => { const p = ${positionsOf(name)}; return Math.max(...p) - Math.min(...p); })()`;
  const isInchCube = (span) => Math.abs(span - 25.4) < 1e-5;
  const useInches = () => clickOn("[data-action=units]");

  const unitFiles = {
    "cube-a.stl": binaryStl(unitCube([0, 0, 0])),
    "cube-b.stl": binaryStl(unitCube([2, 0, 0])),
    "open-c.stl": binaryStl(unitCube([0, 3, 0], true)),
  };
  for (const [name, data] of Object.entries(unitFiles)) {
    writeFileSync(join(fixtures, name), data);
  }
  const startUnits = await hook("units()");
  await setSelect("import-units", "in");
  await upload(Object.keys(unitFiles).map((n) => join(fixtures, n)));
  const imported = await waitFor(
    `(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn) && i; })()`,
    30_000,
  );
  const sA = "cube-a.stl";
  const sB = "cube-b.stl";
  const extentA = await page(spanOf(sA));
  const rowScale = await page(`document.querySelector('[data-mesh=${q(sA)}] [data-import-scale]')?.textContent ?? null`);
  verdict(
    "units-import",
    startUnits.units === "mm" && imported !== undefined && isInchCube(extentA) && rowScale === "×25.4"
      && Math.abs(imported[0].radius - (25.4 * Math.sqrt(3)) / 2) < 1e-9 && imported[0].importScale.factor === 25.4,
    `started in ${startUnits.units}; a 1 unit cube loaded as inches spans ${extentA}, radius ${imported?.[0].radius}, row ${rowScale}`,
  );

  // A in the collision pair, B 1 in away; the readouts in both units.
  await clickOn(`[data-mesh=${q(sA)}] .mesh-name`);
  await pick("a", sA);
  await pick("b", sB);
  await waitFor(collision(`c.a === ${q(sA)} && c.b === ${q(sB)} && c.coloring[${q(sA)}]?.status === "ready" && c.readouts?.gap > 0`), 30_000);
  const readoutsMm = await waitText(".readouts", "25.4 mm");
  const labelMm = await page(`[...document.querySelectorAll(".slider-head > span:first-child")].some((e) => e.textContent === "Colour from (mm)")`);
  const before = await page(`JSON.stringify([__showcase.collision().coloring, __showcase.settings().heatmap, ${positionsOf(sA)}])`);
  await useInches();
  const readoutsIn = await waitText(".readouts", "1 in");
  const labels = await page(`[...document.querySelectorAll(".slider-head > span:first-child")].map((e) => e.textContent).join(" | ")`);
  const after = await page(`JSON.stringify([__showcase.collision().coloring, __showcase.settings().heatmap, ${positionsOf(sA)}])`);
  const noMove = before === after;
  // Back through the input: dragging "Colour from" to 1 on the inch scale is 25.4 mm.
  await setRange("gap", 1);
  const gapStored = await waitFor(`(() => { const g = __showcase.settings().heatmap?.gap; return Math.abs(g - 25.4) < 1e-9 && g; })()`, 5000);
  const gapRange = await waitFor(
    `(() => { const r = __showcase.collision().coloring[${q(sA)}]?.range; return Math.abs(r?.[1] - 25.4) < 1e-9 && r; })()`,
    10_000,
  );
  const gapShown = await page(`document.querySelector("[data-control=gap]").value`);
  const csv = await hook("exportCsv()");
  verdict(
    "units-collision",
    readoutsMm === true && labelMm === true && readoutsIn === true && noMove && labels.includes("Colour from (in)")
      && !labels.includes("(mm)") && gapStored !== undefined && gapRange !== undefined && Math.abs(Number(gapShown) - 1) < 1e-9
      && csv.startsWith("mesh_a,mesh_b,min_gap_in,intersects,max_depth_in\n"),
    `gap reads 25.4 mm ${readoutsMm === true} and 1 in ${readoutsIn === true}; labels ${labels}; switching moved nothing ${noMove}; slider at ${gapShown} in stores ${gapStored} mm, colours to ${gapRange?.[1]}; CSV header ${csv.split("\n")[0]}`,
  );

  // The group table and the repair panel's perimeter and typed inputs.
  await clickOn("[data-mode=group]");
  const groupIn = await waitText(".pair-readouts", "1 in", 30_000);
  await clickOn("[data-mode=pair]");
  await clickOn(`[data-mesh="open-c.stl"] .mesh-name`);
  const perimeterIn = await waitText(".holes", "perimeter 4 in");
  await page(`(() => {
    const el = document.querySelector('label.field[title="Brush radius"] input');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, "0.1");
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
  await sleep(100);
  const typedIn = await page(`document.querySelector('label.field[title="Brush radius"] input').value`);
  await useInches(); // to mm
  const perimeterMm = await waitText(".holes", "perimeter 102 mm");
  const typedMm = await page(`document.querySelector('label.field[title="Brush radius"] input').value`);
  await useInches(); // back to inches
  verdict(
    "units-repair",
    groupIn === true && perimeterIn === true && perimeterMm === true && typedIn === "0.1" && typedMm === "2.54",
    `group table in inches ${groupIn === true}; perimeter 4 in ${perimeterIn === true}, 102 mm ${perimeterMm === true}; a radius typed as ${typedIn} in reads ${typedMm} mm`,
  );

  // Click-to-query: the point, the coloring value, area (squared) and volume (cubed).
  await clickOn(`[data-mesh=${q(sA)}] .mesh-name`);
  await clickOn(`[data-mesh=${q(sA)}] button[title="Point the camera at this mesh"]`);
  await clickOn(`[data-mesh=${q(sA)}] .mesh-name`);
  await clickOn("[data-action=inspect-query]");
  await sleep(1000);
  const onA = await hook(`screenPoint(${q(sA)})`);
  await click(onA.x, onA.y);
  const queried = await waitText("[data-readout=query]", "area", 10_000);
  const queryIn = await textOf("[data-readout=query]");
  await useInches(); // to mm
  await waitText("[data-readout=query]", "mm²", 5000);
  const queryMm = await textOf("[data-readout=query]");
  await useInches();
  const measureOk =
    queried === true && queryIn?.includes("area 6 in²") && queryIn.includes("volume 1 in³")
    && queryMm?.includes("area 3871 mm²") && queryMm.includes("volume 16387 mm³");
  verdict("units-measure", measureOk === true, `in: "${queryIn}"; mm: "${queryMm}"`);

  // The ruler, between two of A's vertices: the panel's distance and its unit.
  await clickOn("[data-action=inspect-ruler]");
  await sleep(300);
  // Wider until the two clicks snap to different vertices (a third click restarts the ruler).
  let ruled;
  for (const reach of [60, 100, 140, 180, 220]) {
    await click(onA.x - reach, onA.y);
    await sleep(300);
    await click(onA.x + reach, onA.y);
    ruled = await waitFor(`__showcase.inspectTools().ruler.distance`, 2000);
    if (ruled !== undefined) {
      break;
    }
  }
  await waitText("[data-readout=ruler]", " in", 5000);
  const rulerIn = await textOf("[data-readout=ruler]");
  await useInches(); // to mm
  await waitText("[data-readout=ruler]", " mm", 5000);
  const rulerMm = await textOf("[data-readout=ruler]");
  await useInches();
  // The ruler's distance is canonical (mm): the same number in both.
  const sig3 = (x) => (x >= 100 ? x.toFixed(0) : String(Number(x.toPrecision(3))));
  verdict(
    "units-ruler",
    ruled > 0 && rulerIn?.includes(`= ${sig3(ruled / 25.4)} in`) && rulerMm?.includes(`= ${sig3(ruled)} mm`),
    `${ruled} mm; "${rulerIn}"; "${rulerMm}"`,
  );
  await clickOn("[data-action=ruler-clear]");
  await clickOn("[data-action=inspect-ruler]");

  // The project: units and scale stored; a reload scales nothing twice.
  const savedText = await hook("saveProject()");
  const saved = JSON.parse(savedText);
  await page(`__showcase.openProject(${q(savedText)})`);
  await waitFor(`(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn); })()`, 30_000);
  await sleep(200);
  const again = await hook("saveProject()");
  const reopened = await page(`(() => {
    const a = __showcase.items().find((m) => m.name === ${q(sA)});
    return { radius: a.radius, scale: a.importScale, extent: ${spanOf(sA)} };
  })()`);
  const samePositions = JSON.parse(again).meshes.every((m, i) => m.positions === saved.meshes[i].positions);
  // An older file (no units, no scale): same geometry, on the global default.
  await useInches(); // the default, by hand, is mm now
  const older = JSON.parse(savedText);
  delete older.units;
  for (const m of older.meshes) {
    delete m.importScale;
  }
  await page(`__showcase.openProject(${q(JSON.stringify(older))})`);
  await waitFor(`(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn); })()`, 30_000);
  const olderOpen = await page(`({ u: __showcase.units(), a: __showcase.items().find((m) => m.name === ${q(sA)}), extent: ${spanOf(sA)} })`);
  // A project in inches, opened with mm the default: the project's unit shows, the default stays.
  await page(`__showcase.openProject(${q(savedText)})`);
  await waitFor(`(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn); })()`, 30_000);
  const projectUnits = await hook("units()");
  verdict(
    "units-project",
    saved.units === "in" && saved.meshes[0].importScale?.factor === 25.4 && samePositions && isInchCube(reopened.extent)
      && reopened.scale.factor === 25.4 && Math.abs(reopened.radius - (25.4 * Math.sqrt(3)) / 2) < 1e-5
      && olderOpen.u.units === "mm" && isInchCube(olderOpen.extent) && olderOpen.a.importScale.factor === 1
      && projectUnits.units === "in" && projectUnits.defaultUnits === "mm",
    `saved ${saved.units}, ×${saved.meshes[0].importScale?.factor}; reopened spans ${reopened.extent}, same bytes ${samePositions}; older file: ${olderOpen.u.units}, spans ${olderOpen.extent}, ×${olderOpen.a.importScale.factor}; project units ${projectUnits.units}, default ${projectUnits.defaultUnits}`,
  );

  // DICOM speaks mm: inches assumed for files changes nothing about a series.
  await setSelect("import-units", "in");
  await page(`(() => {
    const dt = new DataTransfer();
    for (const f of ${JSON.stringify(series.map((s) => ({ name: s.name, bytes: [...s.bytes] })))}) {
      dt.items.add(new File([new Uint8Array(f.bytes).buffer], f.name));
    }
    const canvas = document.querySelector("canvas");
    const r = canvas.getBoundingClientRect();
    const opts = { bubbles: true, cancelable: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, dataTransfer: dt };
    canvas.dispatchEvent(new DragEvent("dragover", opts));
    canvas.dispatchEvent(new DragEvent("drop", opts));
  })()`);
  const seriesMesh = await waitFor(`__showcase.items().find((m) => m.name.includes("(iso ") && m.drawn)`, 30_000);
  verdict(
    "units-dicom",
    // The sphere has a 2.5 voxel radius at 1 mm spacing: a half diagonal near 4.3 mm, not 110.
    seriesMesh !== undefined && seriesMesh.importScale.factor === 1 && seriesMesh.radius > 2 && seriesMesh.radius < 8,
    seriesMesh === undefined ? "no series" : `${seriesMesh.name}: radius ${seriesMesh.radius}, ×${seriesMesh.importScale.factor}`,
  );

  // The global default survives a reload; Reset settings restores mm.
  await useInches(); // the project's in -> mm, by hand
  await useInches(); // -> in, by hand: the default too
  await browser.goto(`http://127.0.0.1:${port}/`, "window.__showcase !== undefined");
  const reloaded = await hook("units()");
  const buttonText = await textOf("[data-action=units]");
  await clickOn("[data-action=reset-settings]");
  const resetUnits = await hook("units()");
  verdict(
    "units-default",
    reloaded.units === "in" && reloaded.defaultUnits === "in" && reloaded.importScale.factor === 1 && buttonText === "in"
      && resetUnits.units === "mm" && resetUnits.defaultUnits === "mm",
    `after a reload ${reloaded.units} (default ${reloaded.defaultUnits}, import ×${reloaded.importScale.factor}), after reset ${resetUnits.units}`,
  );
  await page("localStorage.clear(); true");
  }

  // ---- previews: each built-in model's picture, and the loaded meshes' ----
  {
  await browser.goto(`http://127.0.0.1:${port}/`, "window.__showcase !== undefined");
  await waitFor(`__showcase.items().find((m) => m.name.includes("headsq") && m.drawn)`, 60_000);
  // Decoded to premultiplied RGBA at the picture's size, so empty pixels compare equal.
  await page(`(() => {
    window.__pixels = async (src) => {
      const img = new Image();
      img.src = src;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = c.height = 96;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0, 96, 96);
      const d = g.getImageData(0, 0, 96, 96).data;
      for (let i = 0; i < d.length; i += 4) { const a = d[i + 3] / 255; d[i] *= a; d[i + 1] *= a; d[i + 2] *= a; }
      return d;
    };
    window.__diff = (a, b, step = 1, from = 0) => { let n = 0, sum = 0; for (let i = from; i < a.length; i += step) { sum += Math.abs(a[i] - b[i]); n++; } return sum / n; };
    window.__blank = (d) => {
      let covered = 0, sum = 0, sq = 0;
      for (let i = 0; i < d.length; i += 4) { if (d[i + 3] > 0) { covered++; const l = (d[i] + d[i + 1] + d[i + 2]) / 3; sum += l; sq += l * l; } }
      const mean = sum / Math.max(covered, 1);
      return { covered: covered / (d.length / 4), spread: Math.sqrt(Math.max(0, sq / Math.max(covered, 1) - mean * mean)) };
    };
  })()`);
  const cards = await page(`Promise.all([...document.querySelectorAll(".model-card")].map(async (b) => {
    const img = b.querySelector("img");
    await img.decode();
    return { action: b.dataset.action, src: img.src, alt: img.alt, width: img.naturalWidth, height: img.naturalHeight, blank: __blank(await __pixels(img.src)) };
  }))`);
  const wantCards = ["dicom-demo-vessels", "dicom-demo-brain", "dicom-demo-heart", "sample-headsq", "sample-lidc2", "demo-meshes"];
  const liveLoads = [];
  for (const card of cards) {
    const from = await hook("items().length");
    await clickOn(`[data-action=${card.action}]`);
    liveLoads.push(
      await waitFor(
        `(() => { const all = __showcase.items(); return all.length > ${from} && all.slice(${from}).every((m) => m.drawn) && __showcase.jobs().every((j) => j.phase === "done"); })()`,
        120_000,
      ),
    );
    await page(`(window.__live ??= {}, window.__live[${q(card.action)}] = __showcase.modelPreview(${from}), true)`);
  }
  // Every committed picture against every model's live drawing.
  const matrix = await page(`(async () => {
    const out = {};
    const cards = [...document.querySelectorAll(".model-card")].map((b) => [b.dataset.action, b.querySelector("img").src]);
    for (const [name, src] of cards) {
      const shown = await __pixels(src);
      out[name] = {};
      for (const [other] of cards) {
        out[name][other] = __diff(shown, await __pixels(__live[other]));
      }
    }
    return out;
  })()`);
  const nearestOther = (action) => Math.min(...Object.entries(matrix[action] ?? {}).filter(([k]) => k !== action).map(([, v]) => v));
  const badCards = cards
    .filter((c) => {
      const own = matrix[c.action]?.[c.action];
      // Own: the same pixels but for WebP's loss. Another model is far off.
      return !(c.width === 96 && c.height === 96 && c.alt.length > 0 && c.blank.covered > 0.02 && c.blank.spread > 3 && own < 6 && nearestOther(c.action) > 2 * own);
    })
    .map((c) => `${c.action}: ${c.width}x${c.height}, alt ${q(c.alt)}, covered ${fmt(c.blank.covered)}, spread ${fmt(c.blank.spread)}, own ${fmt(matrix[c.action]?.[c.action] ?? NaN)}, nearest other ${fmt(nearestOther(c.action))}`);
  verdict(
    "previews",
    badCards.length === 0 && q(cards.map((c) => c.action)) === q(wantCards) && liveLoads.every((l) => l === true),
    badCards.length > 0 ? badCards.join("; ") : cards.map((c) => `${c.action} own ${fmt(matrix[c.action][c.action])}, nearest other ${fmt(nearestOther(c.action))}`).join("; "),
  );
  // Each row's picture is of its own mesh: its silhouette (alpha) matches the mesh drawn now.
  await waitFor(`document.querySelectorAll("img.mesh-preview").length === __showcase.items().length`, 60_000);
  const rows = await page(`(async () => {
    const out = [];
    const items = __showcase.items();
    const imgs = [...document.querySelectorAll(".meshes > li")].map((li) => li.querySelector("img.mesh-preview"));
    for (let i = 0; i < imgs.length; i++) {
      const shown = await __pixels(imgs[i].src);
      const live = await __pixels(__showcase.rowPreview(i));
      out.push({ name: items[i].name, alt: imgs[i].alt, silhouette: __diff(shown, live, 4, 3), blank: __blank(shown) });
    }
    return out;
  })()`);
  const badRows = rows.filter((r) => !(r.alt.includes(r.name) && r.silhouette < 1 && r.blank.covered > 0.02 && r.blank.spread > 3));
  verdict(
    "previews-rows",
    rows.length >= 8 && badRows.length === 0,
    badRows.length > 0 ? q(badRows) : `${rows.length} rows, worst silhouette difference ${fmt(Math.max(...rows.map((r) => r.silhouette)))}`,
  );

  // ---- previews-edit: a stroke does not redraw; the picture follows it ----
  await page(`(() => {
    for (const li of document.querySelectorAll(".meshes > li")) {
      if (![${q(A)}, ${q(B)}, ${q(C)}].includes(li.dataset.mesh)) li.querySelector("button.danger").click();
    }
  })()`);
  await waitFor(`__showcase.items().length === 3`, 20_000);
  await clickOn("[data-action=view-reset]");
  await clickOn(`[data-mesh=${q(A)}] .mesh-name`);
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true" ? true : null`, 20_000);
  const settled = await waitFor(`(() => { const s = __showcase.meshPreview(${q(A)}); return s.triangles > 0 && !s.pending && s; })()`, 60_000);
  const srcOf = `document.querySelector(${q(`[data-mesh="${A}"] img.mesh-preview`)}).src`;
  const original = await page(srcOf);
  await page(`(window.__before = __pixels(${q(original)}), true)`);
  await sleep(100);
  const strokeAt = await hook(`screenPoint(${q(A)})`);
  const shapeOf = `__showcase.positions(${q(A)}).join()`;
  const shapeBefore = await page(shapeOf);
  const undoBefore = await hook("status().canUndo");
  await mouse("mouseMoved", strokeAt.x, strokeAt.y, { button: "none" });
  await mouse("mousePressed", strokeAt.x, strokeAt.y, { buttons: 1 });
  for (let i = 1; i <= 30; i++) {
    await mouse("mouseMoved", strokeAt.x + (i % 8) * 2, strokeAt.y + (i % 5) * 2, { buttons: 1 });
    await sleep(16);
  }
  const during = await hook(`meshPreview(${q(A)})`);
  await mouse("mouseReleased", strokeAt.x, strokeAt.y, { buttons: 0 });
  const edited = await waitFor(`(() => { const s = __showcase.meshPreview(${q(A)}); return s.renders > ${settled.renders} && ${srcOf} !== ${q(original)} && s; })()`, 20_000);
  const shapeChanged = (await page(shapeOf)) !== shapeBefore;
  const changed = edited === undefined ? null : await page(`(async () => __diff(await __before, await __pixels(${srcOf})))()`);
  await clickOn('[data-action="sculpt-tool-inflate"]');
  // Live remeshing splits a stroke into several undo steps: back to the start.
  for (let i = 0; i < 200 && (await hook("status().canUndo")); i++) {
    await clickOn("[data-action=undo]");
    await sleep(20);
  }
  const pictureDiff = `(async () => __diff(await __before, await __pixels(${srcOf})))()`;
  await waitFor(`${pictureDiff}.then((d) => d < 1)`, 20_000);
  const restored = await page(pictureDiff);
  verdict(
    "previews-edit",
    shapeChanged && !undoBefore && during.starts === settled.starts && changed !== null && changed > 0.05 && restored < 1,
    `shape changed ${shapeChanged}; ${during.starts - settled.starts} redraw(s) started during the stroke; picture changed by ${fmt(changed ?? NaN)} after it, ${fmt(restored ?? NaN)} from the original after Undo`,
  );
  }

  // ---- local: ?local, the synchronous Scene on the main thread ----
  await browser.goto(`http://127.0.0.1:${port}/?local`, "window.__showcase !== undefined");
  await waitFor(`document.querySelector("[data-action=demo-meshes]") !== null`, 10_000);
  await clickOn("[data-action=demo-meshes]");
  const localDemo = await waitFor(
    `(() => { const want = [${q(A)}, ${q(B)}, ${q(C)}];
      const got = want.map((n) => __showcase.items().find((m) => m.name === n)).filter((m) => m !== undefined && m.drawn);
      return __showcase.mode === "local" && got.length === 3 && got; })()`,
    60_000,
  );
  await clickOn(`[data-mesh=${JSON.stringify(A)}] .mesh-name`);
  await pick("a", A);
  await pick("b", B);
  const localHeat = await waitFor(
    collision(`c.a === ${q(A)} && c.b === ${q(B)} && c.coloring[${q(A)}]?.status === "ready"
      && c.readouts?.penetration?.depth > 0 && c.readouts.intersects`),
    30_000,
  );
  await clickOn("[data-mode=group]");
  // Members are A and B plus maybe the boot head (it can arrive late):
  // untick the head and tick C atomically, so a late head finds the
  // members full and stays out.
  await waitFor(`document.querySelector('[data-group-member=${JSON.stringify(C)}]') !== null`, 10_000);
  await page(`(() => {
    const box = (match) => [...document.querySelectorAll("[data-group-member]")].find((e) =>
      (e.getAttribute("data-group-member") ?? "").includes(match),
    );
    const head = box("headsq");
    if (head !== undefined && head.checked) {
      head.click();
    }
    const c = box(${q(C)});
    if (c !== undefined && !c.checked) {
      c.click();
    }
    return true;
  })()`);
  // Members locked: drop the boot head if it is here, to lighten software
  // rendering while the group settles (a later head finds members full).
  const localBoot = await hook(`items().find((m) => m.name.includes("headsq"))?.name ?? null`);
  if (localBoot !== null) {
    await clickOn(`[data-mesh=${JSON.stringify(localBoot)}] button.danger`);
    await waitFor(`!__showcase.items().some((m) => m.name === ${JSON.stringify(localBoot)})`, 10_000);
  }
  const localGroup = await waitFor(
    groupState(`${to(A, [B, C])} && g.settled && g.readouts?.pairs.length === 3 && ${pairOf(A, B)}?.depth > 0`),
    // Local mode shares the main thread (and CI draws it in software): settling takes a while.
    90_000,
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
