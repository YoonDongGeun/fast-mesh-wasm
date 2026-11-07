// Checks the showcase (examples/showcase) on touch: headless Chrome with phone
// metrics and touch emulation, against `vite preview`, with real multi-touch
// input through the DevTools protocol (`Input.dispatchTouchEvent`), the path a
// touchscreen's events take: gesture recognition, touch-action and pointer
// events all run. It asserts on the camera itself, not on event counts.
//
//   node scripts/showcase-touch-check.mjs               # reuse examples/showcase/dist
//   node scripts/showcase-touch-check.mjs --build       # vite build first
//   node scripts/showcase-touch-check.mjs --shots <dir> # save screenshots there
//   node scripts/showcase-touch-check.mjs --swiftshader # software WebGL (the default on Linux and under CI)
//   node scripts/showcase-touch-check.mjs --verbose     # echo the page's console
//   node scripts/showcase-touch-check.mjs --slow-frames 1000 # block each frame that long (ms), like a slow runner
//
// Waits are on rendered frames and on state, never on a guess at how long a
// frame takes: software WebGL on a CI runner can take over a second per frame.
// The camera's matrices only change when a frame renders, so a screen point
// read straight after moving the camera is stale. Without --shots the device
// pixel ratio is 1 and the boot head (its volume is costly in software) goes
// once its slice handles are checked.
//
// Checks (each prints PASS or FAIL; the exit code is 1 on any FAIL):
// - layout-*: phone portrait, phone landscape and tablet portrait: the canvas
//   is big enough to use, no sideways page scroll, and the last control of
//   each panel and of the status bar can be scrolled to and hit.
// - targets: every toolbar, panel and status bar button is at least 44 px tall and wide.
// - previews-phone: the model buttons' pictures fit the phone width with alt text and their data-action;
//   a tap on one loads the model and lists its own picture.
// - overlay-pinch: a pinch that starts on a viewport overlay does not zoom the page.
// - orbit: one finger rotates about the target at a fixed distance.
// - pinch: two fingers apart dolly in and together dolly out; the target stays.
// - pan: two fingers moved together move the target, not the distance.
// - pinch-to-orbit: lifting one finger of a pinch continues as an orbit, without a jump.
// - slice-finger: one finger on a slice handle drags the slice; a second finger
//   landing on it during an orbit does not.
// - section-finger: the same for the section plane's handle.
// - slice-nudge: a finger or mouse press on a slice handle that moves less than
//   the tap slop keeps the selection; the same press on nothing clears it.
// - mixed-pinch: with one finger on the canvas and one on a toolbar button, the
//   button's finger moves nothing, moving and lifting; the canvas's finger orbits.
// - resting-*: with a finger resting on a panel, one canvas finger still taps and
//   grabs the gizmo, a slice, the section handle and the hover tip; a second
//   canvas finger still does not tap, and two still pinch.
// - tap: a tap selects a mesh, a tap on nothing clears, a two-finger tap does neither.
// - gizmo: one finger drags the handle; a second finger's moves and lift do not
//   disturb the drag; it is one undo step.
// - gizmo-cancel: a touchcancel ends a gizmo drag and the camera works afterwards.
// - gizmo-second-finger: a second finger landing on the handle during an orbit
//   does not grab it.
// - no-gizmo: orbit still works after the gizmo is disposed (deselect), which
//   clears the canvas's inline touch-action.
// - hover-tap: in Group mode a tap shows the hover tip, inside the screen, and
//   the next drag clears it.
// - mouse: with touch off, desktop-sized buttons; a drag orbits, the wheel zooms, a click selects.
// - console: no uncaught exception, in the page or its Workers.
import { spawn } from "node:child_process";
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
const verbose = flag("--verbose");
const port = Number(option("--port", 4194));
const shots = option("--shots", undefined);
const slowFrames = option("--slow-frames", undefined);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
function verdict(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
  if (!ok) {
    failures++;
  }
}

if (flag("--build")) {
  await new Promise((done, fail) => {
    const child = spawn(process.execPath, [vite, "build"], { cwd: example, stdio: "inherit" });
    child.on("exit", (code) => (code === 0 ? done() : fail(new Error(`vite build exited ${code}`))));
  });
}

const server = await startServer(
  process.execPath,
  [vite, "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"],
  { cwd: example, url: `http://127.0.0.1:${port}/` },
);
const swiftshader = flag("--swiftshader") || process.platform === "linux" || process.env.CI !== undefined;
const browser = await launchChrome({
  verbose,
  args: swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [],
});
const page = (expr) => browser.evaluate(expr);
const hook = (expr) => page(`window.__showcase.${expr}`);
const q = JSON.stringify;

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

/** Resolves after `n` rendered frames: what the last input did has been drawn, the camera's matrices included. */
const frames = (n = 2) =>
  page(`new Promise((done) => { let left = ${n}; const f = () => (--left > 0 ? requestAnimationFrame(f) : done(true)); requestAnimationFrame(f); })`);

/** The value of `expr` once it has held for three frames (a drag's last move may land a frame late). */
const settled = (expr) =>
  page(`new Promise((done) => {
    const t = performance.now();
    let last;
    let same = 0;
    const f = () => {
      const v = JSON.stringify(${expr});
      same = v === last ? same + 1 : 0;
      last = v;
      if (same >= 3 || performance.now() - t > 15000) done(JSON.parse(v));
      else requestAnimationFrame(f);
    };
    requestAnimationFrame(f);
  })`);

// ---- Device and fingers ----

const metrics = (width, height) =>
  browser.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: shots === undefined ? 1 : 2, mobile: true });

/**
 * A touch event: `touchStart` with every finger on the glass (new ids land),
 * `touchMove` with those that move, `touchEnd` with those that lift (none: all).
 */
const fingers = (type, points) =>
  browser.send("Input.dispatchTouchEvent", {
    type,
    touchPoints: Object.entries(points).map(([id, [x, y]]) => ({ id: Number(id), x, y })),
  });
const lift = () => fingers("touchEnd", {});

/** `steps` interpolated touchMoves from `from` to `to` (finger id to [x, y]), both sharing their fingers. */
async function moveFingers(from, to, steps = 12) {
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const at = {};
    for (const id of Object.keys(to)) {
      at[id] = [from[id][0] + (to[id][0] - from[id][0]) * t, from[id][1] + (to[id][1] - from[id][1]) * t];
    }
    await fingers("touchMove", at);
    await sleep(20);
  }
  await frames(); // the controls' update() ran
}

async function tap(x, y) {
  await fingers("touchStart", { 0: [x, y] });
  await sleep(30);
  await lift();
  await frames();
}

/** Taps a DOM element through the screen, so whatever covers it takes the tap instead. */
async function tapElement(selector) {
  const c = await page(`(() => {
    const el = document.querySelector(${q(selector)});
    if (el === null) return null;
    el.scrollIntoView({ block: "center" });
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  })()`);
  if (c === null) {
    throw new Error(`no ${selector}`);
  }
  await sleep(50);
  await tap(c.x, c.y);
}

/**
 * A finger's place off the canvas, resting on a panel: the left gutter of the
 * first sidebar on screen. The page must be scrolled to the top.
 */
async function panelPoint() {
  const at = await page(`(() => {
    for (const box of [".sidebar-left", ".sidebar-right"]) {
      const r = document.querySelector(box).getBoundingClientRect();
      const top = Math.max(r.top, 0);
      const bottom = Math.min(r.bottom, innerHeight);
      if (bottom - top >= 40) {
        const p = [r.left + 3, (top + bottom) / 2];
        return document.elementFromPoint(p[0], p[1]).closest("canvas, button") === null && p;
      }
    }
    return false;
  })()`);
  if (!at) {
    throw new Error("no panel on screen");
  }
  return at;
}

/** Arms a wait for the next click on the page; `clicked` resolves once it came and its effects are drawn. */
const armClick = () => page(`void (window.__clicked = false, window.__onClick ??= () => (window.__clicked = true), addEventListener("click", window.__onClick, { capture: true, once: true }))`);
const clicked = async () => {
  const got = await waitFor("window.__clicked === true", 5_000);
  await frames();
  return got === true;
};

const AT_DEFAULT_POSE = `(() => { const p = __showcase.cameraPose().position; return Math.hypot(p[0] + 0.6, p[1] - 3.4, p[2] - 8.2) < 1e-9; })()`;
/** Reset view, by a tap on its button or (`tapped` false) a click; on return the pose has been drawn. */
async function resetView(tapped = true) {
  if (tapped) {
    await tapElement("[data-action=view-reset]");
  } else {
    await page(`document.querySelector("[data-action=view-reset]").click()`);
  }
  const done = await waitFor(AT_DEFAULT_POSE, 20_000);
  if (done === undefined) {
    throw new Error("Reset view did not reset the camera");
  }
  await page(`window.scrollTo(0, 0)`);
  await frames();
}

const sub = (a, b) => a.map((v, i) => v - b[i]);
const len = (v) => Math.hypot(...v);
const pose = async () => {
  const p = await hook("cameraPose()");
  return { ...p, distance: len(sub(p.position, p.target)) };
};
const moved = (a, b) => len(sub(a.position, b.position));
const targetMoved = (a, b) => len(sub(a.target, b.target));
const canvasRect = () =>
  page(`(() => { const r = __showcase_canvas().getBoundingClientRect(); return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height }; })()`);
const fmt = (x) => x.toFixed(3);

async function shot(name) {
  if (shots === undefined) {
    return;
  }
  mkdirSync(shots, { recursive: true });
  await frames(3);
  const { data } = await browser.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(shots, `${name}.png`), Buffer.from(data, "base64"));
}

/** A quiet corner of the canvas: no mesh, handle or overlay. */
const emptyPoint = (r) => [r.left + 16, r.bottom - 16];

const A = "A · blobby sphere";
const B = "B · rounded box";
const C = "C · open shell + piece";

/** The layout at the current metrics: no sideways scroll, a usable canvas, every panel's end reachable. */
async function checkLayout(name, minWidth, minHeight) {
  await frames(3);
  const facts = await page(`(async () => {
    const out = { overflow: document.documentElement.scrollWidth - window.innerWidth, blocked: [] };
    const r = __showcase_canvas().getBoundingClientRect();
    out.canvas = [Math.round(r.width), Math.round(r.height)];
    for (const box of [".sidebar-left", ".sidebar-right", ".statusbar", ".toolbar"]) {
      const all = [...document.querySelectorAll(box + " button:not(:disabled)")].filter((b) => b.offsetParent !== null);
      const last = all.at(-1);
      if (last === undefined) { out.blocked.push(box + ": no button"); continue; }
      last.scrollIntoView({ block: "center", inline: "center" });
      await new Promise((f) => requestAnimationFrame(() => requestAnimationFrame(f)));
      const b = last.getBoundingClientRect();
      const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
      if (hit === null || !(hit === last || last.contains(hit))) {
        out.blocked.push(box + " " + (last.textContent ?? "") + " under " + (hit?.className || hit?.tagName || "nothing"));
      }
    }
    window.scrollTo(0, 0);
    document.querySelector(".sidebar-left")?.scrollTo(0, 0);
    document.querySelector(".sidebar-right")?.scrollTo(0, 0);
    return out;
  })()`);
  verdict(
    name,
    facts.overflow <= 0 && facts.canvas[0] >= minWidth && facts.canvas[1] >= minHeight && facts.blocked.length === 0,
    `canvas ${facts.canvas.join("x")} (min ${minWidth}x${minHeight}), sideways overflow ${facts.overflow}px, blocked ${q(facts.blocked)}`,
  );
}

try {
  await metrics(390, 844);
  await browser.send("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
  await browser.goto(`http://127.0.0.1:${port}/`, "window.__showcase !== undefined");
  await page(`window.__showcase_canvas = () => document.querySelector(".viewport canvas")`);
  const media = await page(`({ coarse: matchMedia("(pointer: coarse)").matches, touch: navigator.maxTouchPoints, gl: !!document.createElement("canvas").getContext("webgl2") })`);
  console.log(`${browser.name}; pointer: coarse ${media.coarse}; maxTouchPoints ${media.touch}; WebGL2 ${media.gl}`);

  const boot = await waitFor(
    `(() => { const v = __showcase.items().find((m) => m.name.includes("headsq") && m.drawn); return v !== undefined && __showcase.sceneSlices().shown === 3 && v; })()`,
    90_000,
  );
  verdict("boot", boot !== undefined, boot === undefined ? q(await hook("items()")) : boot.name);
  await shot("phone-boot");
  if (slowFrames !== undefined) {
    await page(`(function f() { const t = performance.now(); while (performance.now() - t < ${Number(slowFrames)}); requestAnimationFrame(f); })()`);
  }

  // ---- layout and targets ----
  await checkLayout("layout-phone", 350, 300);
  const small = await page(`[...document.querySelectorAll(".toolbar button, .sidebar button, .statusbar button")]
    .filter((b) => b.offsetParent !== null)
    .map((b) => ({ b, r: b.getBoundingClientRect() }))
    .filter(({ r }) => r.height < 44 || r.width < 44)
    .map(({ b, r }) => (b.dataset.action ?? b.textContent ?? "?").trim().slice(0, 18) + " " + Math.round(r.width) + "x" + Math.round(r.height))`);
  verdict("targets", small.length === 0, small.length === 0 ? "all at least 44x44" : `${small.length} too small: ${small.slice(0, 12).join("; ")}`);

  // ---- a pinch on an overlay must not zoom the whole page ----
  const bar = await page(`(() => { const r = document.querySelector(".toolbar").getBoundingClientRect(); return [r.left + 60, r.top + 30]; })()`);
  await fingers("touchStart", { 0: bar });
  await sleep(30);
  await fingers("touchStart", { 0: bar, 1: [bar[0] + 40, bar[1] + 40] });
  await sleep(30);
  await moveFingers({ 0: bar, 1: [bar[0] + 40, bar[1] + 40] }, { 0: [bar[0] - 40, bar[1]], 1: [bar[0] + 200, bar[1] + 200] });
  await lift();
  await frames();
  const zoom = await page(`visualViewport.scale`);
  verdict("overlay-pinch", zoom === 1, `page zoom ${zoom}`);

  // ---- a nudge of a slice handle is not a tap on nothing: finger, then mouse ----
  {
    const emptySpot = emptyPoint(await canvasRect());
    const head = await page(`__showcase.items().find((m) => m.name.includes("headsq")).name`);
    // The slices show while their mesh is selected, as it is after boot. The name's click toggles.
    const selectHead = async () => {
      if ((await hook("selected()")) !== head) {
        await page(`document.querySelector('[data-mesh*="headsq"] .mesh-name').click()`);
      }
      await waitFor(`__showcase.sliceHandles()?.axial && __showcase.gizmo()?.mesh === ${q(head)}`, 20_000);
      await frames();
    };
    const button = (type, x, y, extra = {}) => browser.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });
    const grabbing = () => page(`__showcase_canvas().style.cursor === "grabbing"`);
    const fingerPress = async ([x, y], dx) => {
      await armClick();
      await fingers("touchStart", { 0: [x, y] });
      const grabbed = await grabbing();
      await sleep(30);
      await fingers("touchMove", { 0: [x + dx, y] });
      await lift();
      return { click: await clicked(), grabbed };
    };
    const mousePress = async ([x, y], dx) => {
      await armClick();
      await button("mouseMoved", x, y, { button: "none" });
      await frames();
      await button("mousePressed", x, y, { buttons: 1 });
      const grabbed = await grabbing();
      await button("mouseMoved", x + dx, y, { buttons: 1 });
      await button("mouseReleased", x + dx, y, { buttons: 0 });
      return { click: await clicked(), grabbed };
    };
    // A slow frame can stretch a press into a long press, which clicks nothing: that try says nothing, so press again.
    const pressUntilClick = async (press, at, dx) => {
      let result;
      for (let i = 0; i < 4 && !result?.click; i++) {
        result = await press(at, dx);
      }
      return result;
    };
    const nudge = async (press, dx) => {
      // The control: the same press on nothing is a tap, and clears.
      await selectHead();
      const onNothing = await pressUntilClick(press, emptySpot, dx);
      const cleared = await waitFor(`__showcase.selected() === null`, 5_000);
      await selectHead();
      // A corner clear of the mesh, whose click would otherwise land on the mesh and keep it selected.
      const h = (await hook("sliceHandles(1)")).coronal;
      const onHandle = await pressUntilClick(press, [h.x, h.y], dx);
      return { control: onNothing.click && cleared === true, clicked: onHandle.click, grabbed: onHandle.grabbed, kept: await hook("selected()") };
    };
    const byFinger = await nudge(fingerPress, 8);
    // Within 2 px: three.js fiber reports no miss for a mouse click further than that.
    const byMouse = await nudge(mousePress, 1);
    verdict(
      "slice-nudge",
      [byFinger, byMouse].every((b) => b.control && b.clicked && b.grabbed && b.kept === head),
      `finger ${q(byFinger)}; mouse ${q(byMouse)}`,
    );
    await selectHead(); // a failure here should not take the next checks' handles with it
  }

  // ---- slice handles: a tool under one finger, not under a second ----
  const empty = emptyPoint(await canvasRect());
  // The boot head's slices: the camera frames them, so the handles are in view.
  const handles = await waitFor(`(() => { const h = __showcase.sliceHandles(); return h?.axial && h; })()`, 10_000);
  const axial = handles?.axial;
  if (axial === undefined) {
    verdict("slice-finger", false, "no slice handle in view");
  } else {
    const before = await hook("mpr().slices.axial");
    const n = Math.hypot(axial.dx, axial.dy) || 1;
    const along = [axial.x + (axial.dx / n) * 80, axial.y + (axial.dy / n) * 80];
    const at0 = await pose();
    await fingers("touchStart", { 0: [axial.x, axial.y] });
    await sleep(30);
    await moveFingers({ 0: [axial.x, axial.y] }, { 0: along });
    const dragged = await hook("mpr().slices.axial");
    await lift();
    await frames();
    const at1 = await pose();
    const fresh = (await hook("sliceHandles()")).axial;
    await fingers("touchStart", { 0: empty });
    await sleep(30);
    await fingers("touchStart", { 0: empty, 1: [fresh.x, fresh.y] });
    await sleep(30);
    await moveFingers({ 0: empty, 1: [fresh.x, fresh.y] }, { 0: empty, 1: [fresh.x - (axial.dx / n) * 80, fresh.y - (axial.dy / n) * 80] });
    const during = await hook("mpr().slices.axial");
    await lift();
    await frames();
    verdict(
      "slice-finger",
      dragged !== before && moved(at0, at1) < 1e-9 && during === dragged,
      `one finger moved the slice ${before} -> ${dragged}, camera ${fmt(moved(at0, at1))}; a second finger left it at ${during}`,
    );
  }

  if (axial !== undefined) {
    // With a finger resting on a panel, one canvas finger still drags a slice.
    const rest = await panelPoint();
    const h = (await hook("sliceHandles()")).axial;
    const s0 = await hook("mpr().slices.axial");
    const n = Math.hypot(h.dx, h.dy) || 1;
    await fingers("touchStart", { 2: rest });
    await sleep(30);
    await fingers("touchStart", { 2: rest, 0: [h.x, h.y] });
    await sleep(30);
    await moveFingers({ 2: rest, 0: [h.x, h.y] }, { 2: rest, 0: [h.x - (h.dx / n) * 80, h.y - (h.dy / n) * 80] });
    const s1 = await hook("mpr().slices.axial");
    await lift();
    await frames();
    verdict("resting-slice", s1 !== s0, `slice ${s0} -> ${s1}`);
  }

  // ---- demo meshes, framed ----
  await tapElement("[data-action=demo-meshes]");
  const demo = await waitFor(
    `(() => { const want = [${q(A)}, ${q(B)}, ${q(C)}];
      const got = want.map((n) => __showcase.items().find((m) => m.name === n)).filter((m) => m !== undefined && m.drawn);
      return got.length === 3 && __showcase.status().status === "idle"; })()`,
    60_000,
  );
  verdict("demo-tap", demo === true, "tapped the demo link; A, B and C drawn");
  // Drop the head: its scene volume is by far the costliest thing to draw, and no later check needs it.
  await page(`document.querySelector('[data-mesh*="headsq"] button.danger')?.click()`);
  await waitFor(`!__showcase.items().some((m) => m.name.includes("headsq"))`, 20_000);
  await page(`window.scrollTo(0, 0)`);
  await resetView();
  await shot("phone-demo");

  // ---- orbit ----
  const start = emptyPoint(await canvasRect());
  const p0 = await pose();
  await fingers("touchStart", { 0: start });
  await moveFingers({ 0: start }, { 0: [start[0] + 120, start[1] - 60] });
  await lift();
  await frames();
  const p1 = await pose();
  verdict(
    "orbit",
    moved(p0, p1) > 0.1 && targetMoved(p0, p1) < 1e-6 && Math.abs(p1.distance - p0.distance) < 1e-3,
    `camera moved ${fmt(moved(p0, p1))}, target ${fmt(targetMoved(p0, p1))}, distance ${fmt(p0.distance)} -> ${fmt(p1.distance)}`,
  );

  // ---- pinch and pan ----
  const r = await canvasRect();
  const cx = (r.left + r.right) / 2;
  const cy = r.top + r.height * 0.75;
  const near = { 0: [cx - 40, cy], 1: [cx + 40, cy] };
  const far = { 0: [cx - 140, cy], 1: [cx + 140, cy] };
  const twoDown = async (points) => {
    await fingers("touchStart", { 0: points[0] });
    await sleep(30);
    await fingers("touchStart", points);
    await sleep(30);
  };
  const p2 = await pose();
  await twoDown(near);
  await moveFingers(near, far);
  await lift();
  await frames();
  const p3 = await pose();
  await twoDown(far);
  await moveFingers(far, near);
  await lift();
  await frames();
  const p4 = await pose();
  verdict(
    "pinch",
    p3.distance < p2.distance * 0.8 && p4.distance > p3.distance * 1.2 && targetMoved(p2, p3) < 0.5,
    `distance ${fmt(p2.distance)} -> apart ${fmt(p3.distance)} -> together ${fmt(p4.distance)}`,
  );
  const panned = { 0: [near[0][0], near[0][1] - 60], 1: [near[1][0], near[1][1] - 60] };
  await twoDown(near);
  await moveFingers(near, panned);
  await lift();
  await frames();
  const p5 = await pose();
  verdict(
    "pan",
    targetMoved(p4, p5) > 0.05 && Math.abs(p5.distance - p4.distance) < p4.distance * 0.02,
    `target moved ${fmt(targetMoved(p4, p5))}, distance ${fmt(p4.distance)} -> ${fmt(p5.distance)}`,
  );

  // ---- pinch, then lift one finger: it orbits on, smoothly ----
  await twoDown(near);
  await moveFingers(near, { 0: [near[0][0] - 10, cy], 1: [near[1][0] + 10, cy] }, 4);
  const stay = [near[0][0] - 10, cy];
  await fingers("touchEnd", { 1: [near[1][0] + 10, cy] });
  await frames();
  const p6 = await pose();
  await moveFingers({ 0: stay }, { 0: [stay[0] + 2, stay[1]] }, 1);
  const p7 = await pose();
  await moveFingers({ 0: [stay[0] + 2, stay[1]] }, { 0: [stay[0] + 80, stay[1] - 30] });
  const p8 = await pose();
  await lift();
  await frames();
  verdict(
    "pinch-to-orbit",
    moved(p6, p7) < 0.2 * p6.distance && moved(p7, p8) > 0.05 && Math.abs(p8.distance - p7.distance) < 1e-3,
    `first step ${fmt(moved(p6, p7))}, then ${fmt(moved(p7, p8))}, distance ${fmt(p7.distance)} -> ${fmt(p8.distance)}`,
  );

  // ---- one finger on the canvas, one on a toolbar button: only the canvas's drives the camera ----
  await resetView();
  {
    const onCanvas = emptyPoint(await canvasRect());
    const button = await page(`(() => { const r = document.querySelector("[data-action=view-reset]").getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()`);
    const m0 = await pose();
    await fingers("touchStart", { 0: onCanvas });
    await sleep(30);
    await fingers("touchStart", { 0: onCanvas, 1: button });
    await sleep(30);
    await frames();
    const m1 = await pose();
    // The button's finger moves alone.
    const button2 = [button[0] + 30, button[1] + 20];
    await moveFingers({ 0: onCanvas, 1: button }, { 0: onCanvas, 1: button2 });
    const m2 = await pose();
    // The canvas's finger orbits, smoothly, wherever the other has been.
    const step = [onCanvas[0] + 2, onCanvas[1]];
    await moveFingers({ 0: onCanvas, 1: button2 }, { 0: step, 1: button2 }, 1);
    const m3 = await pose();
    const end = [step[0] + 100, step[1] - 50];
    await moveFingers({ 0: step, 1: button2 }, { 0: end, 1: button2 });
    const m4 = await pose();
    // The button's finger lifts: nothing moves, then or at the next step.
    await fingers("touchEnd", { 1: button2 });
    await frames();
    const m5 = await pose();
    const end2 = [end[0] + 2, end[1]];
    await moveFingers({ 0: end }, { 0: end2 }, 1);
    const m6 = await pose();
    await lift();
    await frames();
    const small = 0.2 * m0.distance;
    verdict(
      "mixed-pinch",
      moved(m0, m1) < 1e-9 && moved(m1, m2) < 1e-9 && moved(m2, m3) < small && moved(m3, m4) > 0.1 && targetMoved(m0, m4) < 1e-6 && Math.abs(m4.distance - m0.distance) < 1e-3 && moved(m4, m5) < small && moved(m5, m6) < small,
      `landing ${fmt(moved(m0, m1))}; button finger's moves ${fmt(moved(m1, m2))}; canvas finger's first step ${fmt(moved(m2, m3))}, then ${fmt(moved(m3, m4))}, target ${fmt(targetMoved(m0, m4))}; button finger's lift ${fmt(moved(m4, m5))}, next step ${fmt(moved(m5, m6))}`,
    );
  }

  // ---- taps ----
  await resetView();
  const overA = await hook(`screenPoint(${q(A)})`);
  await tap(overA.x, overA.y);
  const tapped = await waitFor(`__showcase.selected() === ${q(A)}`, 20_000);
  const before2 = await hook("selected()");
  const overB = await hook(`screenPoint(${q(B)})`);
  await twoDown({ 0: [overB.x, overB.y], 1: [overB.x + 30, overB.y] });
  await lift();
  await frames(3);
  const afterTwo = await hook("selected()");
  const nothing = emptyPoint(await canvasRect());
  await twoDown({ 0: nothing, 1: [nothing[0] + 30, nothing[1]] });
  await lift();
  await frames(3);
  const afterTwoEmpty = await hook("selected()");
  verdict(
    "tap",
    tapped === true && afterTwo === before2 && afterTwoEmpty === before2,
    `tap selected ${before2}; two-finger tap on B left ${afterTwo}; on nothing left ${afterTwoEmpty}`,
  );
  await shot("phone-selected");

  // ---- a finger resting on a panel: the canvas's fingers count by themselves ----
  {
    await page(`window.scrollTo(0, 0)`);
    await frames();
    const rest = await panelPoint();
    const restDown = async () => {
      await fingers("touchStart", { 2: rest });
      await sleep(30);
    };
    const allUp = async () => {
      await lift();
      await frames();
    };
    /** A tap by finger 0 with the resting finger down; whether it reached the page as a click. */
    const restTap = async (x, y) => {
      await armClick();
      await fingers("touchStart", { 2: rest, 0: [x, y] });
      await sleep(30);
      await fingers("touchEnd", { 0: [x, y] });
      return clicked();
    };

    // A tap selects; two canvas fingers do not; a tap on nothing clears.
    await restDown();
    const b0 = await hook(`screenPoint(${q(B)})`);
    const tapClick = await restTap(b0.x, b0.y);
    const selB = await waitFor(`__showcase.selected() === ${q(B)}`, 5_000);
    await allUp();
    await restDown();
    const a0 = await hook(`screenPoint(${q(A)})`);
    await fingers("touchStart", { 2: rest, 0: [a0.x, a0.y] });
    await sleep(30);
    await fingers("touchStart", { 2: rest, 0: [a0.x, a0.y], 1: [a0.x + 30, a0.y] });
    await sleep(30);
    await fingers("touchEnd", { 0: [a0.x, a0.y], 1: [a0.x + 30, a0.y] });
    await frames(3);
    const stillB = await hook("selected()");
    await allUp();
    await restDown();
    const none = emptyPoint(await canvasRect());
    const noneClick = await restTap(none[0], none[1]);
    const cleared = await waitFor(`__showcase.selected() === null`, 5_000);
    await allUp();

    // The gizmo: B again, then grabbed by one canvas finger.
    const b1 = await hook(`screenPoint(${q(B)})`);
    await tap(b1.x, b1.y);
    const gizmoOnB = await waitFor(`__showcase.gizmo()?.mesh === ${q(B)}`, 20_000);
    await frames();
    const tb0 = await hook(`translation(${q(B)})`);
    const b2 = await hook(`screenPoint(${q(B)})`);
    const cb0 = await pose();
    await restDown();
    await fingers("touchStart", { 2: rest, 0: [b2.x, b2.y] });
    const grabbed = await waitFor("__showcase.dragging()", 5_000);
    await moveFingers({ 2: rest, 0: [b2.x, b2.y] }, { 2: rest, 0: [b2.x + 40, b2.y - 30] });
    const tb1 = await settled(`__showcase.translation(${q(B)})`);
    await allUp();
    await waitFor("__showcase.dragging() === false", 5_000);
    const cb1 = await pose();
    if (len(sub(tb1, tb0)) > 0) {
      await page(`document.querySelector("[data-action=undo]").click()`);
      await waitFor(`Math.hypot(...__showcase.translation(${q(B)}).map((v, i) => v - ${q(tb0)}[i])) < 1e-9`, 20_000);
    }

    // Two canvas fingers still pinch.
    const d0 = await pose();
    await restDown();
    await fingers("touchStart", { 2: rest, 0: near[0] });
    await sleep(30);
    await fingers("touchStart", { 2: rest, ...near });
    await sleep(30);
    await moveFingers({ 2: rest, ...near }, { 2: rest, ...far });
    const d1 = await pose();
    await allUp();
    verdict(
      "resting-finger",
      tapClick && selB === true && stillB === B && noneClick && cleared === true && gizmoOnB === true && grabbed === true && len(sub(tb1, tb0)) > 1e-3 && moved(cb0, cb1) < 1e-9 && d1.distance < d0.distance * 0.8,
      `tap clicked ${tapClick}, selected ${selB === true}; two canvas fingers left ${stillB}; tap on nothing clicked ${noneClick}, cleared ${cleared === true}; gizmo grabbed ${grabbed === true}, moved ${fmt(len(sub(tb1, tb0)))}, camera ${fmt(moved(cb0, cb1))}; pinch ${fmt(d0.distance)} -> ${fmt(d1.distance)}`,
    );

    // Back as the next check expects: the default view, A selected and its gizmo up.
    await resetView(false);
    const a1 = await hook(`screenPoint(${q(A)})`);
    await tap(a1.x, a1.y);
  }

  // ---- gizmo: one finger drags, a second cannot disturb it ----
  const attached = await waitFor(`__showcase.gizmo()?.mesh === ${q(A)}`, 20_000);
  await frames(); // the gizmo picks against matrices set when it is drawn
  const t0 = await hook(`translation(${q(A)})`);
  const from = await hook(`screenPoint(${q(A)})`);
  const cam0 = await pose();
  await fingers("touchStart", { 0: [from.x, from.y] });
  const dragging = await waitFor("__showcase.dragging()", 20_000);
  const to = [from.x + 40, from.y - 30];
  await moveFingers({ 0: [from.x, from.y] }, { 0: to });
  const t1 = await settled(`__showcase.translation(${q(A)})`);
  const spot = [from.x - 90, from.y + 120];
  await fingers("touchStart", { 0: to, 1: spot });
  await sleep(30);
  await moveFingers({ 0: to, 1: spot }, { 0: to, 1: [spot[0] - 40, spot[1] + 40] });
  const t2 = await settled(`__showcase.translation(${q(A)})`);
  await fingers("touchEnd", { 1: [spot[0] - 40, spot[1] + 40] });
  await frames();
  const still = await hook("dragging()");
  const to2 = [to[0] + 30, to[1]];
  await moveFingers({ 0: to }, { 0: to2 });
  const t3 = await settled(`__showcase.translation(${q(A)})`);
  await lift();
  await frames();
  const cam1 = await pose();
  await page(`document.querySelector("[data-action=undo]").click()`);
  await waitFor(`Math.hypot(...__showcase.translation(${q(A)}).map((v, i) => v - ${q(t0)}[i])) < 1e-9`, 20_000);
  const undone = await hook(`translation(${q(A)})`);
  verdict(
    "gizmo",
    attached === true && dragging === true && len(sub(t1, t0)) > 1e-3 && len(sub(t2, t1)) < 1e-9 && still === true && len(sub(t3, t2)) > 1e-3 && moved(cam0, cam1) < 1e-9 && len(sub(undone, t0)) < 1e-9,
    `gizmo on A ${attached === true}; moved ${fmt(len(sub(t1, t0)))}; second finger moved it ${fmt(len(sub(t2, t1)))}; still dragging ${still}; then ${fmt(len(sub(t3, t2)))}; camera ${fmt(moved(cam0, cam1))}; one undo left it ${fmt(len(sub(undone, t0)))} from the start`,
  );

  // ---- the system takes a drag over (touchcancel): it ends, and the camera works again ----
  const c0 = await hook(`screenPoint(${q(A)})`);
  await fingers("touchStart", { 0: [c0.x, c0.y] });
  const cancelGrab = await waitFor("__showcase.dragging()", 20_000);
  await moveFingers({ 0: [c0.x, c0.y] }, { 0: [c0.x + 20, c0.y - 10] }, 4);
  await fingers("touchCancel", {});
  const cancelEnd = await waitFor("__showcase.dragging() === false", 20_000);
  const cancelled = await hook(`translation(${q(A)})`);
  await page(`document.querySelector("[data-action=undo]").click()`);
  await waitFor(`Math.hypot(...__showcase.translation(${q(A)}).map((v, i) => v - ${q(cancelled)}[i])) > 1e-9`, 20_000);
  await frames();
  const c1 = await pose();
  const cAt = emptyPoint(await canvasRect());
  await fingers("touchStart", { 0: cAt });
  await moveFingers({ 0: cAt }, { 0: [cAt[0] + 60, cAt[1] - 30] });
  await lift();
  await frames();
  const c2 = await pose();
  await resetView(false);
  verdict(
    "gizmo-cancel",
    cancelGrab === true && cancelEnd === true && moved(c1, c2) > 0.1,
    `dragging ${cancelGrab === true}, ended by touchcancel ${cancelEnd === true}; orbit after it moved ${fmt(moved(c1, c2))}`,
  );

  // ---- a second finger landing on the gizmo during an orbit does not grab it ----
  const handle = await hook(`screenPoint(${q(A)})`);
  const t4 = await hook(`translation(${q(A)})`);
  const orbitAt = emptyPoint(await canvasRect());
  await fingers("touchStart", { 0: orbitAt });
  await sleep(30);
  await fingers("touchStart", { 0: orbitAt, 1: [handle.x, handle.y] });
  await sleep(30);
  const grabbed = await hook("dragging()");
  await moveFingers({ 0: orbitAt, 1: [handle.x, handle.y] }, { 0: orbitAt, 1: [handle.x + 50, handle.y + 20] });
  const t5 = await settled(`__showcase.translation(${q(A)})`);
  await lift();
  await frames();
  // The same handle under one finger does grab: otherwise the above proves nothing.
  const grabAt = await hook(`screenPoint(${q(A)})`);
  await fingers("touchStart", { 0: [grabAt.x, grabAt.y] });
  const grabbable = await waitFor("__showcase.dragging()", 20_000);
  await lift();
  await waitFor("__showcase.dragging() === false", 20_000);
  await frames();
  verdict(
    "gizmo-second-finger",
    grabbed === false && len(sub(t5, t4)) < 1e-9 && grabbable === true,
    `second finger: dragging ${grabbed}, mesh moved ${fmt(len(sub(t5, t4)))}; one finger on the handle: dragging ${grabbable === true}`,
  );

  // ---- orbit with the gizmo gone: deselected, its dispose clears the inline touch-action ----
  const gone = emptyPoint(await canvasRect());
  await sleep(400); // the app ignores a missed tap for 350 ms after a gizmo drag ends (onMissed)
  await tap(gone[0], gone[1]);
  const cleared = await waitFor(`__showcase.selected() === null && __showcase.gizmo() === null`, 20_000);
  const touchAction = await page(`getComputedStyle(document.querySelector(".viewport")).touchAction`);
  const q0 = await pose();
  await fingers("touchStart", { 0: gone });
  await moveFingers({ 0: gone }, { 0: [gone[0] + 120, gone[1] - 60] });
  await lift();
  await frames();
  const q1 = await pose();
  await twoDown(near);
  await moveFingers(near, far);
  await lift();
  await frames();
  const q2 = await pose();
  verdict(
    "no-gizmo",
    cleared === true && touchAction === "none" && moved(q0, q1) > 0.1 && Math.abs(q1.distance - q0.distance) < 1e-3 && q2.distance < q1.distance * 0.8,
    `deselected ${cleared === true}; touch-action ${touchAction}; orbit moved ${fmt(moved(q0, q1))}; pinch ${fmt(q1.distance)} -> ${fmt(q2.distance)}`,
  );

  // ---- section handle ----
  await resetView();
  await page(`document.querySelector("[data-action=inspect-section]")?.click()`);
  await waitFor(`document.querySelector("[data-action=section-enable]") !== null`, 20_000);
  await page(`document.querySelector("[data-action=section-enable]")?.click()`);
  const sec = await waitFor(`__showcase.sectionHandle()`, 20_000);
  await page(`window.scrollTo(0, 0)`);
  await frames();
  const secAt = await hook("sectionHandle()");
  if (sec === undefined || secAt === null) {
    verdict("section-finger", false, "no section handle");
  } else {
    const so0 = (await hook("inspectTools()")).section?.plane?.origin ?? null;
    const secEmpty = emptyPoint(await canvasRect());
    await fingers("touchStart", { 0: secEmpty });
    await sleep(30);
    await fingers("touchStart", { 0: secEmpty, 1: [secAt.x, secAt.y] });
    await sleep(30);
    const secDrag = await hook("dragging()");
    await moveFingers({ 0: secEmpty, 1: [secAt.x, secAt.y] }, { 0: secEmpty, 1: [secAt.x + 40, secAt.y - 60] });
    const so1 = (await hook("inspectTools()")).section?.plane?.origin ?? null;
    await lift();
    await frames();
    await resetView(false);
    const secAt2 = await hook("sectionHandle()");
    await fingers("touchStart", { 0: [secAt2.x, secAt2.y] });
    const oneDrag = await waitFor("__showcase.dragging()", 20_000);
    await moveFingers({ 0: [secAt2.x, secAt2.y] }, { 0: [secAt2.x + 40, secAt2.y - 60] });
    const so2 = (await hook("inspectTools()")).section?.plane?.origin ?? null;
    await lift();
    await frames();
    verdict(
      "section-finger",
      secDrag === false && so0 !== null && len(sub(so1, so0)) < 1e-9 && oneDrag === true && len(sub(so2, so1)) > 1e-3,
      `second finger: dragging ${secDrag}, plane moved ${so0 === null ? "?" : fmt(len(sub(so1, so0)))}; one finger: dragging ${oneDrag === true}, plane moved ${so0 === null ? "?" : fmt(len(sub(so2, so1)))}`,
    );

    // With a finger resting on a panel, one canvas finger still grabs the handle.
    await page(`window.scrollTo(0, 0)`);
    await frames();
    const rest = await panelPoint();
    const secAt3 = await hook("sectionHandle()");
    await fingers("touchStart", { 2: rest });
    await sleep(30);
    await fingers("touchStart", { 2: rest, 0: [secAt3.x, secAt3.y] });
    const restDrag = await waitFor("__showcase.dragging()", 5_000);
    await lift();
    await frames();
    verdict("resting-section", restDrag === true, `dragging ${restDrag === true}`);

    await page(`document.querySelector("[data-action=section-disable]")?.click()`);
    await page(`document.querySelector("[data-action=inspect-section]")?.click()`);
    await frames();
  }

  // ---- the hover tip's tap fallback ----
  await page(`document.querySelector("[data-mode=group]")?.click()`);
  await waitFor(`document.querySelector('[data-group-member=${q(C)}]') !== null`, 10_000);
  await page(`document.querySelector('[data-group-member=${q(C)}]').click()`);
  await waitFor(`(() => { const g = __showcase.group(); return g.members.length === 3 && g.settled; })()`, 30_000);
  await resetView(false);
  const overA2 = await hook(`screenPoint(${q(A)})`);
  await tap(overA2.x, overA2.y);
  const tipState = await waitFor(`(() => { const h = __showcase.group().hover; return h?.mesh === ${q(A)} && h.partner !== null && h; })()`, 30_000);
  const tipBox = await page(`(() => { const t = document.querySelector("[data-hover-tip]"); if (t === null) return null; const r = t.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: innerWidth, h: innerHeight, text: t.textContent }; })()`);
  await shot("phone-hover-tip");
  // Near the screen's edge the tip stays on screen.
  const e0 = await canvasRect();
  const dragAt = [e0.left + 16, e0.bottom - 16];
  await fingers("touchStart", { 0: dragAt });
  await moveFingers({ 0: dragAt }, { 0: [dragAt[0] + 80, dragAt[1] - 40] });
  const tipDuring = await page(`document.querySelector("[data-hover-tip]") !== null`);
  await lift();
  await frames();
  verdict(
    "hover-tap",
    tipState !== undefined && tipBox !== null && tipBox.left >= 0 && tipBox.right <= tipBox.w && tipDuring === false,
    `tip ${tipBox === null ? "absent" : `${q(tipBox.text)} at ${Math.round(tipBox.left)}..${Math.round(tipBox.right)} of ${tipBox.w}`}; shown during a later drag ${tipDuring}`,
  );

  // A finger resting on a panel: a tap still shows the tip.
  await page(`window.scrollTo(0, 0)`);
  await frames();
  const tipRest = await panelPoint();
  const overA3 = await hook(`screenPoint(${q(A)})`);
  await fingers("touchStart", { 2: tipRest });
  await sleep(30);
  await fingers("touchStart", { 2: tipRest, 0: [overA3.x, overA3.y] });
  await sleep(30);
  await fingers("touchEnd", { 0: [overA3.x, overA3.y] });
  const tipRested = await waitFor(`document.querySelector("[data-hover-tip]") !== null`, 10_000);
  await lift();
  await frames();
  verdict("resting-tip", tipRested === true, `tip shown ${tipRested === true}`);

  // ---- other shapes ----
  await metrics(844, 390);
  await checkLayout("layout-landscape", 400, 300);
  await shot("landscape");
  await metrics(768, 1024);
  await checkLayout("layout-tablet", 400, 400);
  await shot("tablet");
  await metrics(390, 844);
  await frames(3);
  await page(`document.querySelector(".sidebar-right button:not(:disabled)")?.scrollIntoView({ block: "center" })`);
  await shot("phone-panels");

  // ---- previews on a phone: pictures fit, alt text, hooks kept; a tap loads the model and lists its picture ----
  await page(`document.querySelector(".model-card").scrollIntoView({ block: "center" })`);
  await frames();
  const pictures = await page(`Promise.all([...document.querySelectorAll(".model-card")].map(async (b) => {
    const img = b.querySelector("img");
    await img.decode();
    const r = b.getBoundingClientRect();
    return { action: b.dataset.action, alt: img.alt, width: img.naturalWidth, fits: r.left >= 0 && r.right <= innerWidth, height: Math.round(r.height) };
  }))`);
  await shot("phone-previews");
  const beforeTap = (await hook("items()")).length;
  await tapElement("[data-action=dicom-demo-heart]");
  const rowPicture = await waitFor(
    `(() => { const all = [...document.querySelectorAll(".meshes > li")]; const img = all.at(-1)?.querySelector("img.mesh-preview"); return all.length > ${beforeTap} && img?.complete && img.naturalWidth === 96 && img.alt.length > 0 && innerWidth >= img.getBoundingClientRect().right && img.alt; })()`,
    60_000,
  );
  const overflowAfter = await page(`document.documentElement.scrollWidth - innerWidth`);
  await page(`document.querySelector(".meshes").lastElementChild.scrollIntoView({ block: "center" })`);
  await shot("phone-rows");
  await page(`window.scrollTo(0, 0)`);
  verdict(
    "previews-phone",
    pictures.length === 6 && pictures.every((p) => p.action && p.alt && p.width === 96 && p.fits && p.height >= 44) && rowPicture !== undefined && overflowAfter <= 0,
    `${pictures.length} pictures on buttons (tallest ${Math.max(...pictures.map((p) => p.height))}px), row: ${q(rowPicture ?? null)}, sideways overflow ${overflowAfter}px`,
  );

  // ---- a mouse, after all that: desktop sizes, and the camera and clicks as before ----
  await browser.send("Emulation.setTouchEmulationEnabled", { enabled: false });
  await browser.send("Emulation.clearDeviceMetricsOverride");
  await browser.send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
  await frames();
  const mouse = (type, x, y, extra = {}) => browser.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });
  const desk = await page(`({ coarse: matchMedia("(pointer: coarse)").matches, undo: document.querySelector("[data-action=undo]").getBoundingClientRect().height, box: getComputedStyle(document.querySelector(".toolbar")).flexWrap })`);
  await resetView(false);
  const mr = await canvasRect();
  const m0 = await pose();
  const mAt = [mr.left + 30, mr.bottom - 30];
  await mouse("mouseMoved", mAt[0], mAt[1], { button: "none" });
  await mouse("mousePressed", mAt[0], mAt[1], { buttons: 1 });
  for (let i = 1; i <= 10; i++) {
    await mouse("mouseMoved", mAt[0] + 12 * i, mAt[1] - 6 * i, { buttons: 1 });
    await sleep(20);
  }
  await mouse("mouseReleased", mAt[0] + 120, mAt[1] - 60, { buttons: 0 });
  await frames();
  const m1 = await pose();
  await browser.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: mr.left + 200, y: mr.bottom - 200, deltaX: 0, deltaY: -240 });
  await frames(3);
  const m2 = await pose();
  await resetView(false);
  const mA = await hook(`screenPoint(${q(B)})`);
  await mouse("mouseMoved", mA.x, mA.y, { button: "none" });
  await mouse("mousePressed", mA.x, mA.y, { buttons: 1 });
  await mouse("mouseReleased", mA.x, mA.y, { buttons: 0 });
  const mSel = await waitFor(`__showcase.selected() === ${q(B)}`, 20_000);
  verdict(
    "mouse",
    !desk.coarse && desk.undo < 40 && desk.box !== "wrap" && moved(m0, m1) > 0.1 && Math.abs(m1.distance - m0.distance) < 1e-3 && m2.distance !== m1.distance && mSel === true,
    `undo button ${Math.round(desk.undo)} px tall; drag moved the camera ${fmt(moved(m0, m1))}; wheel ${fmt(m1.distance)} -> ${fmt(m2.distance)}; click selected ${mSel === true}`,
  );

  // ---- console ----
  const problems = browser.consoleLines.filter((l) => /^(page|worker): (error|assert): |^(page|worker): exception: /.test(l));
  for (const p of problems) {
    console.log(`  ${p}`);
  }
  verdict("console", problems.length === 0, `${browser.consoleLines.length} console line(s), ${problems.length} error(s)`);
} catch (e) {
  console.error(e);
  console.error(browser.consoleLines.join("\n"));
  failures++;
} finally {
  await browser.close();
  server.stop();
}
if (failures > 0) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
