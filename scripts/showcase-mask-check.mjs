// Checks the sculpt mask in the showcase (examples/showcase) on real pixels:
// headless Chrome over the DevTools protocol against `vite preview`, painting
// with the mouse through the UI and comparing screenshots of the canvas.
//
//   node scripts/showcase-mask-check.mjs               # reuse examples/showcase/dist
//   node scripts/showcase-mask-check.mjs --build       # vite build first
//   node scripts/showcase-mask-check.mjs --big         # also the cost on a 1M-triangle mesh
//   node scripts/showcase-mask-check.mjs --shots <dir> # save screenshots there
//   node scripts/showcase-mask-check.mjs --swiftshader # software WebGL (the default on Linux and under CI)
//   node scripts/showcase-mask-check.mjs --verbose     # echo the page's console
//
// Checks (each prints PASS or FAIL; the exit code is 1 on any FAIL):
// - paint: tinted pixels appear around the painted point, and only there.
// - armed: the tint goes when nothing is armed, and comes back when a sculpt tool arms.
// - hide: the Show button hides and restores the tint; the mask stays.
// - invert: the painted point loses its tint, the rest of the mesh gains it.
// - clear: every pixel is back to what it was before the paint.
// - light, dark: on a near-white and a near-black mesh the tint is far from the surface colour.
// - heatmap: with a contact heatmap on the mesh, painting changes no vertex
//   colour and no pixel away from the tint, and clearing restores the pixels.
// - big (--big): a 1M-triangle sphere; painting with the tint shown and hidden,
//   the overlay's build time, frame gaps and long tasks. Passes on the build time.
// - protect: a dab on the mask's edge leaves every masked position byte-identical
//   and moves free vertices; undo restores them all and keeps the mask; the
//   mask's vertices stay put with the stroke-end remesh on, which stands down.
// - grab: a grab pressed on the mask's edge drags free vertices and leaves masked
//   positions byte-identical through the whole stroke; undo restores them.
// - covered: a dab with only masked vertices in its ball is dropped: no undo step.
// - stale: a topology edit that takes a masked vertex clears the mask and its
//   tint, with the notice.
// - weld: with the brush armed and its caches filled, a weld that takes a masked
//   vertex but keeps the slot count clears the mask loudly on the next dab, which lands.
// - follow: after the mesh's colours go, a structural change that keeps its
//   positions, the tint still follows a move of the masked vertices.
// - removed: removing the selected mesh with the mask paint armed, and no mask on
//   it, disarms the paint: the next selected mesh gets its gizmo back.
// - console: no uncaught exception, in the page or its Workers.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
const port = Number(option("--port", 4195));
const shots = option("--shots", undefined);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
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
if (shots !== undefined) {
  mkdirSync(shots, { recursive: true });
}

const server = await startServer(
  process.execPath,
  [vite, "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"],
  { cwd: example, url: `http://127.0.0.1:${port}/` },
);
const swiftshader = flag("--swiftshader") || process.platform === "linux" || process.env.CI !== undefined;
const browser = await launchChrome({
  verbose,
  args: ["--window-size=1400,900", ...(swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [])],
});
const EVAL_TIMEOUT_MS = 120_000;
const page = (expr) =>
  Promise.race([
    browser.evaluate(expr),
    new Promise((_, fail) => setTimeout(() => fail(new Error(`the page did not answer in ${EVAL_TIMEOUT_MS} ms: ${expr.slice(0, 120)}`)), EVAL_TIMEOUT_MS).unref?.()),
  ]);
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

const mouse = (type, x, y, extra = {}) =>
  browser.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });
const clickOn = (selector) =>
  page(`(() => { const el = document.querySelector(${q(selector)}); el?.click(); return el !== null; })()`);
/** Waits for a few rendered frames: a frame after a state change draws it. */
const frames = (n = 4) =>
  page(`new Promise((resolve) => { let k = ${n}; const f = () => (--k <= 0 ? resolve(true) : requestAnimationFrame(f)); requestAnimationFrame(f); })`);

async function pick(side, name) {
  await page(`(() => {
    const sel = document.querySelector('[data-pick=${side}]');
    const opt = [...sel.options].find((o) => o.text === ${q(name)});
    sel.value = opt.value;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
}

/** How far from the press a default-radius paint on A can tint, in CSS pixels at the default view. */
const TINT_REACH = 90;
/** A UV sphere as a binary STL: 2 · segments · (rings − 1) triangles (1,000,000 for 1000 x 501). */
function bigSphereStl(segments, rings) {
  const count = 2 * segments * (rings - 1);
  const buf = Buffer.alloc(84 + 50 * count);
  buf.write("fast-mesh mask check big sphere", 0, "latin1");
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

const A = "A · blobby sphere";
const B = "B · rounded box";

try {
  await browser.goto(`http://127.0.0.1:${port}/`, "window.__showcase !== undefined");
  const bootHead = await waitFor(`__showcase.items().find((m) => m.name.includes("headsq") && m.drawn)?.name`, 90_000);
  await clickOn(`[data-mesh=${q(bootHead)}] button.danger`);
  await clickOn("[data-action=demo-meshes]");
  await waitFor(
    `(() => { const i = __showcase.items(); return i.length === 3 && i.every((m) => m.drawn) && __showcase.selected() !== undefined; })()`,
    60_000,
  );
  await clickOn(`[data-mesh=${q(A)}] .mesh-name`);
  await waitFor(`__showcase.selected() === ${q(A)}`);

  // The demo meshes fill the view: the boot framed the head they replaced.
  await clickOn("[data-action=view-reset]");
  await frames(8);

  const rect = await page(
    `(() => { const r = document.querySelector("canvas").getBoundingClientRect(); return { x: r.left, y: r.top, width: r.width, height: r.height }; })()`,
  );
  /** A resting pointer, away from every mesh: no ball, no hover tip. */
  const park = () => mouse("mouseMoved", rect.x + 8, rect.y + rect.height - 8, { button: "none" });

  /**
   * The canvas now, stored in the page as RGBA under `name`. The app's own
   * panels over the canvas (the toolbox's count changes with the mask) are
   * hidden for the shot.
   */
  async function shot(name) {
    await park();
    // The brush ball goes once the parked pointer's raycast answers, which a
    // slow runner takes frames over; it would tint the shot.
    await waitFor("__showcase.sculpt().hovering === false", 10_000);
    await frames();
    await page(`(() => {
      const s = document.createElement("style");
      s.id = "shot-hide";
      s.textContent = ".toolbox { visibility: hidden !important; }";
      document.head.append(s);
      return true;
    })()`);
    const { data } = await browser.send("Page.captureScreenshot", { format: "png", clip: { ...rect, scale: 1 } });
    await page(`(document.getElementById("shot-hide").remove(), true)`);
    if (shots !== undefined) {
      writeFileSync(join(shots, `${name}.png`), Buffer.from(data, "base64"));
    }
    await page(`(async () => {
      const bytes = Uint8Array.from(atob(${q(data)}), (c) => c.charCodeAt(0));
      const img = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
      const c = new OffscreenCanvas(img.width, img.height);
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(img, 0, 0);
      (window.__shots ??= {})[${q(name)}] = g.getImageData(0, 0, img.width, img.height);
      return true;
    })()`);
  }

  /**
   * The pixels of shot `b` differing from shot `a` by more than `tol` in a
   * channel: how many, their box, and how many near the page point `at`.
   */
  const diff = (a, b, tol = 24, at = null, within = 0) =>
    page(`(() => {
      const A = __shots[${q(a)}], B = __shots[${q(b)}], w = A.width, d0 = A.data, d1 = B.data;
      const at = ${q(at === null ? null : { x: at.x - rect.x, y: at.y - rect.y })}, within = ${within};
      let count = 0, near = 0, sx = 0, sy = 0, x0 = 1e9, x1 = -1, y0 = 1e9, y1 = -1;
      for (let i = 0; i < d0.length; i += 4) {
        if (Math.abs(d0[i] - d1[i]) > ${tol} || Math.abs(d0[i + 1] - d1[i + 1]) > ${tol} || Math.abs(d0[i + 2] - d1[i + 2]) > ${tol}) {
          const x = (i / 4) % w, y = Math.floor(i / 4 / w);
          count++; sx += x; sy += y;
          x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
          if (at !== null && Math.hypot(x - at.x, y - at.y) <= within) near++;
        }
      }
      return { count, near, cx: count ? sx / count : 0, cy: count ? sy / count : 0, box: [x0, y0, x1, y1] };
    })()`);
  /** The mean colour of shot `name` in a square of half-size `half` around the page point `at`. */
  const meanAt = (name, at, half = 4) =>
    page(`(() => {
      const S = __shots[${q(name)}], w = S.width, cx = ${Math.round(at.x - rect.x)}, cy = ${Math.round(at.y - rect.y)};
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = cy - ${half}; y <= cy + ${half}; y++) for (let x = cx - ${half}; x <= cx + ${half}; x++) {
        const i = 4 * (y * w + x);
        r += S.data[i]; g += S.data[i + 1]; b += S.data[i + 2]; n++;
      }
      return [r / n, g / n, b / n];
    })()`);
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  /** One press where `at` is: a dab of mask paint (or a sculpt press). */
  async function press(at) {
    const before = await hook("sculpt().maskOverlay.builds");
    await mouse("mouseMoved", at.x, at.y, { button: "none" });
    await sleep(150);
    await mouse("mousePressed", at.x, at.y, { buttons: 1 });
    await sleep(150);
    await mouse("mouseReleased", at.x, at.y, { buttons: 0 });
    await waitFor(`__showcase.sculpt().maskOverlay.builds > ${before}`, 10_000);
  }
  const maskCount = () =>
    page(`Number(document.querySelector('[data-action="sculpt-mask-clear"]')?.parentElement?.querySelector("span")?.textContent?.match(/^\\d+/)?.[0] ?? 0)`);

  // ---- heatmap: the tint leaves the contact colours alone ----
  const center = await hook(`screenPoint(${q(A)})`);
  const armPaint = async () => {
    await clickOn('[data-action="sculpt-mask-paint"]');
    await waitFor(`document.querySelector('[data-action="sculpt-mask-paint"]')?.getAttribute("aria-pressed") === "true"`);
  };
  await armPaint();
  await pick("a", A);
  await pick("b", B);
  const heat = await waitFor(
    `(() => { const c = __showcase.collision(); return c.a === ${q(A)} && c.b === ${q(B)} && c.coloring[${q(A)}]?.status === "ready" && c.settled; })()`,
    30_000,
  );
  await frames(8);
  const colors0 = await hook(`colorCounts(${q(A)})`);
  await shot("heat-base");
  await press(center);
  await shot("heat-tint");
  const colors1 = await hook(`colorCounts(${q(A)})`);
  const heatTint = await diff("heat-base", "heat-tint", 24, center, TINT_REACH);
  await clickOn('[data-action="sculpt-mask-clear"]');
  await shot("heat-cleared");
  const heatClear = await diff("heat-base", "heat-cleared", 8);
  verdict(
    "heatmap",
    heat !== undefined &&
      colors0 !== null &&
      (colors0.red > 0 || colors0.gap > 0) &&
      q(colors0) === q(colors1) &&
      heatTint.count > 1000 &&
      heatTint.near === heatTint.count &&
      heatClear.count === 0,
    `vertex colours ${q(colors0)} unchanged ${q(colors0) === q(colors1)}; tinted ${heatTint.count} (${heatTint.near} within ${TINT_REACH} px of the press); ${heatClear.count} pixels differ after clear`,
  );

  // A plain surface from here on: without B, A has no contact colours. That
  // drops the colour attribute while the tint shows, keeping the positions.
  await press(center);
  await clickOn(`[data-mesh=${q(B)}] button.danger`);
  await waitFor(`__showcase.items().length === 2 && __showcase.collision().coloring[${q(A)}] === undefined`);
  await frames(8);

  // ---- follow: the tint stays on the surface after that ----
  await hook(`runRepair(${q(A)}, { op: "offset", distance: 0.15 })`);
  await shot("follow");
  const followed = await meanAt("follow", center);
  verdict("follow", followed[0] - followed[1] > 60, `after an offset the press reads ${q(followed.map(Math.round))}: tinted ${followed[0] - followed[1] > 60}`);
  await page(`(document.querySelector("[data-action=undo]").click(), true)`);
  await sleep(300);
  await clickOn('[data-action="sculpt-mask-clear"]');
  await clickOn('[data-action="sculpt-mask-paint"]');
  await frames(8);

  // ---- paint ----
  await shot("idle");
  await armPaint();
  await shot("base");
  await press(center);
  await shot("painted");
  const painted = await diff("base", "painted", 24, center, TINT_REACH);
  const masked = await maskCount();
  verdict(
    "paint",
    masked > 0 && painted.count > 1000 && painted.near === painted.count && Math.hypot(painted.cx + rect.x - center.x, painted.cy + rect.y - center.y) < 40,
    `${masked} masked; ${painted.count} tinted pixels, ${painted.near} within ${TINT_REACH} px of the press, centred ${fmt(Math.hypot(painted.cx + rect.x - center.x, painted.cy + rect.y - center.y))} px from it, box ${q(painted.box)}; overlay ${q(await hook("sculpt().maskOverlay"))}`,
  );

  // ---- armed: nothing armed, no tint; a sculpt tool arms it again ----
  await clickOn('[data-action="sculpt-mask-paint"]');
  await shot("disarmed");
  const disarmed = await diff("idle", "disarmed", 24, center, 16);
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);
  await shot("sculpt-armed");
  const armedDiff = await diff("painted", "sculpt-armed", 24, center, 16);
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await clickOn('[data-action="sculpt-mask-paint"]');
  verdict(
    "armed",
    disarmed.near === 0 && armedDiff.near === 0,
    `disarmed vs idle: ${disarmed.near} pixels near the press; sculpt-armed vs painted: ${armedDiff.near}`,
  );

  // ---- hide: the Show button; the mask stays ----
  await clickOn('[data-action="sculpt-mask-show"]');
  await shot("hidden");
  const hidden = await diff("base", "hidden", 24, center, 16);
  const hiddenMasked = await maskCount();
  await clickOn('[data-action="sculpt-mask-show"]');
  await shot("shown");
  const shown = await diff("painted", "shown", 24, center, 16);
  verdict(
    "hide",
    hidden.near === 0 && hiddenMasked === masked && shown.near === 0,
    `hidden vs base: ${hidden.near} near the press, ${hiddenMasked} still masked; shown vs painted: ${shown.near}`,
  );

  // ---- invert ----
  await clickOn('[data-action="sculpt-mask-invert"]');
  await shot("inverted");
  const atPress = distance(await meanAt("inverted", center), await meanAt("base", center));
  const inverted = await diff("base", "inverted");
  verdict(
    "invert",
    atPress < 16 && inverted.count > painted.count * 2 && (await maskCount()) !== masked,
    `the press is ${fmt(atPress)} from the base colour; ${inverted.count} tinted pixels (was ${painted.count})`,
  );

  // ---- clear: back to the base pixels ----
  await clickOn('[data-action="sculpt-mask-clear"]');
  await shot("cleared");
  const cleared = await diff("base", "cleared", 8);
  verdict("clear", cleared.count === 0 && (await hook("sculpt().maskOverlay.triangles")) === 0, `${cleared.count} pixels differ from the base`);

  // ---- protect: a dab on the mask's edge moves only free vertices ----
  const positions = () => hook(`positions(${q(A)})`);
  await press(center);
  await frames();
  const held = await hook("sculptMask()");
  await shot("edge-tint");
  const paintedTint = await diff("base", "edge-tint", 24, center, TINT_REACH);
  // On the tint's rim, along +x: half the brush ball over the mask.
  const edge = { x: paintedTint.box[2] + rect.x, y: center.y };
  const heldMoved = (a, b) => held.filter((v) => a[3 * v] !== b[3 * v] || a[3 * v + 1] !== b[3 * v + 1] || a[3 * v + 2] !== b[3 * v + 2]).length;
  const freeMoved = (a, b) => {
    const set = new Set(held);
    let n = 0;
    for (let v = 0; v < a.length / 3; v++) {
      if (!set.has(v) && (a[3 * v] !== b[3 * v] || a[3 * v + 1] !== b[3 * v + 1] || a[3 * v + 2] !== b[3 * v + 2])) n++;
    }
    return n;
  };
  const stroke = async (at, ms) => {
    await mouse("mouseMoved", at.x, at.y, { button: "none" });
    await sleep(200);
    await mouse("mousePressed", at.x, at.y, { buttons: 1 });
    await sleep(ms);
    await mouse("mouseReleased", at.x, at.y, { buttons: 0 });
    await sleep(600);
  };
  const p0 = await positions();
  const s0 = await hook("sculpt()");
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);
  await stroke(edge, 500);
  const p1 = await positions();
  const s1 = await hook("sculpt()");
  await shot("edge-dabbed");
  const stillMasked = await hook("sculptMask()");
  verdict(
    "protect",
    s1.dabs > s0.dabs && s1.maskedDabs === s0.maskedDabs && heldMoved(p0, p1) === 0 && freeMoved(p0, p1) > 0 && q(stillMasked) === q(held) && p1.length === p0.length,
    `${s1.dabs - s0.dabs} dabs, ${s1.maskedDabs - s0.maskedDabs} dropped; ${heldMoved(p0, p1)} of ${held.length} masked moved, ${freeMoved(p0, p1)} free moved; slots ${p0.length / 3} -> ${p1.length / 3}; remeshes ${s1.remeshes - s0.remeshes}`,
  );

  // ---- undo: every position back, the mask kept ----
  let undos = 0;
  while (undos < 200 && (await page(`document.querySelector("[data-action=undo]")?.disabled === false`))) {
    await page(`(document.querySelector("[data-action=undo]").click(), true)`);
    undos++;
    await sleep(40);
    if (q(await positions()) === q(p0)) {
      break;
    }
  }
  const p2 = await positions();
  verdict("undo", q(p2) === q(p0) && q(await hook("sculptMask()")) === q(held), `${undos} undo(s) back to the painted positions; mask kept ${q(await hook("sculptMask()")) === q(held)}`);

  // ---- grab: a press on the mask's edge drags only free vertices ----
  await clickOn('[data-action="sculpt-tool-grab"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-grab"]')?.getAttribute("aria-pressed") === "true"`);
  const g0 = await positions();
  await mouse("mouseMoved", edge.x, edge.y, { button: "none" });
  await sleep(200);
  await mouse("mousePressed", edge.x, edge.y, { buttons: 1 });
  await sleep(200);
  let grabHeld = 0;
  for (let i = 1; i <= 6; i++) {
    await mouse("mouseMoved", edge.x + 8 * i, edge.y - 5 * i, { buttons: 1 });
    await sleep(120);
    grabHeld = Math.max(grabHeld, heldMoved(g0, await positions()));
  }
  await mouse("mouseReleased", edge.x + 48, edge.y - 30, { buttons: 0 });
  await sleep(500);
  const g1 = await positions();
  grabHeld = Math.max(grabHeld, heldMoved(g0, g1));
  const grabFree = freeMoved(g0, g1);
  let grabUndos = 0;
  while (grabUndos < 50 && q(await positions()) !== q(g0) && (await page(`document.querySelector("[data-action=undo]")?.disabled === false`))) {
    await page(`(document.querySelector("[data-action=undo]").click(), true)`);
    grabUndos++;
    await sleep(60);
  }
  verdict(
    "grab",
    grabHeld === 0 && grabFree > 0 && q(await positions()) === q(g0),
    `${grabHeld} of ${held.length} masked moved at worst, ${grabFree} free moved; ${grabUndos} undo(s) restored ${q(await positions()) === q(g0)}`,
  );
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);

  // ---- covered: a dab with only masked vertices in its ball is dropped ----
  await page(`(() => {
    const el = document.querySelector('[data-action="sculpt-radius"]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, "0.05");
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
  const s2 = await hook("sculpt()");
  const undoBefore = await page(`document.querySelector("[data-action=undo]")?.disabled`);
  await stroke(center, 300);
  const s3 = await hook("sculpt()");
  const p3 = await positions();
  verdict(
    "covered",
    s3.maskedDabs > s2.maskedDabs && s3.dabs === s2.dabs && q(p3) === q(p0) && (await page(`document.querySelector("[data-action=undo]")?.disabled`)) === undoBefore,
    `${s3.maskedDabs - s2.maskedDabs} dropped, ${s3.dabs - s2.dabs} applied; positions unchanged ${q(p3) === q(p0)}`,
  );
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await clickOn('[data-action="sculpt-mask-paint"]');
  await clickOn('[data-action="sculpt-mask-clear"]');

  // ---- light and dark: the tint is far from the surface colour on both ----
  for (const [name, hex] of [
    ["light", 0xf2f2f2],
    ["dark", 0x161616],
  ]) {
    await hook(`meshColor(${q(A)}, ${hex})`);
    await shot(`${name}-base`);
    await press(center);
    await shot(`${name}-tint`);
    const at = await meanAt(`${name}-tint`, center);
    const base = await meanAt(`${name}-base`, center);
    const shift = distance(at, base);
    await clickOn('[data-action="sculpt-mask-clear"]');
    verdict(name, shift > 100, `surface ${q(base.map(Math.round))} -> tinted ${q(at.map(Math.round))}: distance ${fmt(shift)}`);
  }
  await hook(`meshColor(${q(A)}, null)`);

  // ---- weld: a topology edit that keeps the slot count, with the brush armed ----
  await frames();
  await press(center);
  await clickOn('[data-action="sculpt-mask-invert"]');
  await frames();
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);
  // No remesh passes: their new slots would refresh a cache keyed on the slot count.
  await clickOn('[data-action="sculpt-live-remesh"]');
  await clickOn('[data-action="sculpt-remesh"]');
  await stroke(center, 200);
  const weldSlots = (await positions()).length;
  const welded = await hook(`runRepair(${q(A)}, { op: "weld", tolerance: 0.03 })`);
  await sleep(500);
  const weldKept = (await positions()).length === weldSlots;
  const w0 = await hook("sculpt()");
  // On the mask's edge: the dab's ball holds masked vertices, some now dead.
  await stroke(edge, 300);
  const w1 = await hook("sculpt()");
  const weldNotice = await hook("sculptMaskNotice()");
  const weldMask = await hook("sculptMask()");
  await clickOn('[data-action="sculpt-live-remesh"]');
  await clickOn('[data-action="sculpt-remesh"]');
  verdict(
    "weld",
    welded.mergedVertices > 0 && weldKept && weldMask === null && typeof weldNotice === "string" && w1.maskStale === w0.maskStale + 1 && w1.dabs > w0.dabs,
    `weld ${q(welded)}, slots kept ${weldKept}; mask ${weldMask === null ? "cleared" : "kept"}, notice ${q(weldNotice)}; dabs ${w1.dabs - w0.dabs}`,
  );
  while (await page(`document.querySelector("[data-action=undo]")?.disabled === false`)) {
    await page(`(document.querySelector("[data-action=undo]").click(), true)`);
    await sleep(60);
  }
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await clickOn('[data-action="sculpt-mask-paint"]');
  await waitFor(`document.querySelector('[data-action="sculpt-mask-paint"]')?.getAttribute("aria-pressed") === "true"`);

  // ---- stale: a topology edit that takes a masked vertex clears the mask and its tint ----
  await frames();
  await press(center);
  // Nearly the whole mesh, so the collapses below are sure to take a masked vertex.
  await clickOn('[data-action="sculpt-mask-invert"]');
  await frames();
  const simplified = await hook(`runRepair(${q(A)}, { op: "simplify", targetTriangles: 200 })`);
  await sleep(500);
  await clickOn('[data-action="sculpt-tool-inflate"]');
  await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);
  const staleBefore = (await hook("sculpt()")).maskStale;
  const onMesh = await hook(`screenPoint(${q(A)})`);
  await stroke(onMesh, 200);
  await frames();
  const notice = await hook("sculptMaskNotice()");
  const staleTris = await hook("sculpt().maskOverlay.triangles");
  verdict(
    "stale",
    (await hook("sculptMask()")) === null && typeof notice === "string" && (await hook("sculpt()")).maskStale === staleBefore + 1 && staleTris === 0,
    `simplify ${q(simplified)}; mask ${q((await hook("sculptMask()"))?.length ?? null)}, notice ${q(notice)}; tint triangles ${staleTris}`,
  );
  await clickOn('[data-action="sculpt-tool-inflate"]');

  // ---- removed: the paint goes with the mesh it was armed on ----
  {
    const other = (await hook("items()")).map((i) => i.name).find((n) => n !== A);
    await clickOn(`[data-mesh=${q(other)}] .mesh-name`);
    await waitFor(`__showcase.selected() === ${q(other)}`);
    if (await page(`document.querySelector('[data-action="sculpt-mask-paint"]')?.getAttribute("aria-pressed") !== "true"`)) {
      await clickOn('[data-action="sculpt-mask-paint"]');
    }
    await frames();
    await clickOn(`[data-mesh=${q(other)}] button.danger`);
    await waitFor(`__showcase.items().length === 1`);
    await clickOn(`[data-mesh=${q(A)}] .mesh-name`);
    await waitFor(`__showcase.selected() === ${q(A)}`);
    const gizmo = await waitFor(`__showcase.gizmo()?.mesh === ${q(A)}`, 5000);
    const armedAfter = await page(`document.querySelector('[data-action="sculpt-mask-paint"]')?.getAttribute("aria-pressed")`);
    verdict("removed", gizmo === true && armedAfter === "false", `gizmo back ${gizmo === true}; paint armed ${armedAfter}`);
  }

  // ---- big: what the tint costs while painting at 1M triangles ----
  if (flag("--big")) {
    // A fresh page, with load-time decimation out of the way: the point is a
    // mesh that stays at 1M triangles.
    await browser.goto(`http://127.0.0.1:${port}/?decimate-tris=2000000000`, "window.__showcase !== undefined");
    const head = await waitFor(`__showcase.items().find((m) => m.name.includes("headsq") && m.drawn)?.name`, 90_000);
    await clickOn(`[data-mesh=${q(head)}] button.danger`);
    const dir = mkdtempSync(join(tmpdir(), "fast-mesh-mask-"));
    const { buf, count } = bigSphereStl(1000, 501);
    const path = join(dir, "big-sphere.stl");
    writeFileSync(path, buf);
    const { root: doc } = await browser.send("DOM.getDocument", { depth: 0 });
    const { nodeId } = await browser.send("DOM.querySelector", { nodeId: doc.nodeId, selector: "#file-input" });
    await browser.send("DOM.setFileInputFiles", { nodeId, files: [path] });
    const name = "big-sphere.stl";
    const landed = await waitFor(`__showcase.items().find((i) => i.name === ${q(name)})?.drawn`, 180_000);
    if (landed === undefined) {
      throw new Error(`the 1M mesh was not drawn: ${q(await hook("jobs()"))}`);
    }
    await clickOn(`[data-mesh=${q(name)}] .mesh-name`);
    await waitFor(`__showcase.selected() === ${q(name)}`);
    await clickOn("[data-action=view-reset]");
    await frames(8);
    const at = await hook(`screenPoint(${q(name)})`);
    await armPaint();

    // Long tasks and the worst frame gap over a window.
    await page(`(() => {
      const probe = { tasks: [], maxGap: 0, last: performance.now(), frames: 0 };
      new PerformanceObserver((list) => { for (const e of list.getEntries()) probe.tasks.push(e.duration); }).observe({ type: "longtask" });
      const frame = (t) => { probe.maxGap = Math.max(probe.maxGap, t - probe.last); probe.last = t; probe.frames++; requestAnimationFrame(frame); };
      requestAnimationFrame(frame);
      window.__probe = probe;
      return true;
    })()`);
    /** Holds the paint and moves it for `moves` steps; the window's frames, worst gap and long tasks. */
    const paintWindow = async (dx, moves) => {
      await mouse("mouseMoved", at.x, at.y, { button: "none" });
      await sleep(200);
      await page("(__probe.tasks = [], __probe.maxGap = 0, __probe.frames = 0, __probe.last = performance.now(), true)");
      const t0 = Date.now();
      await mouse("mousePressed", at.x, at.y, { buttons: 1 });
      for (let i = 1; i <= moves; i++) {
        await mouse("mouseMoved", at.x + (dx * i) / moves, at.y + (20 * i) / moves, { buttons: 1 });
        await sleep(50);
      }
      await mouse("mouseReleased", at.x + dx, at.y + 20, { buttons: 0 });
      await sleep(300);
      const r = await page("({ frames: __probe.frames, maxGap: __probe.maxGap, tasks: __probe.tasks })");
      return { ...r, ms: Date.now() - t0 };
    };
    const line = (r) => `${r.frames} frames in ${r.ms} ms, worst gap ${fmt(r.maxGap)} ms, ${r.tasks.length} long task(s)${r.tasks.length ? ` (longest ${fmt(Math.max(...r.tasks))} ms)` : ""}`;

    await clickOn('[data-action="sculpt-mask-show"]');
    const off = await paintWindow(120, 30);
    await clickOn('[data-action="sculpt-mask-clear"]');
    await clickOn('[data-action="sculpt-mask-show"]');
    await frames();
    const before = await hook("sculpt().maskOverlay");
    const on = await paintWindow(120, 30);
    const paint = await hook("sculpt().maskOverlay");
    const paintedVerts = await maskCount();
    console.log(`  1M paint, tint hidden: ${line(off)}`);
    console.log(`  1M paint, tint shown:  ${line(on)}`);
    console.log(`  overlay: ${paint.builds - before.builds} builds, last ${fmt(paint.lastMs)} ms, worst ${fmt(paint.maxMs)} ms; ${paint.triangles} triangles over ${paintedVerts} masked vertices`);

    await clickOn('[data-action="sculpt-mask-invert"]');
    await waitFor(`__showcase.sculpt().maskOverlay.builds > ${paint.builds}`, 20_000);
    await frames();
    const inverted = await hook("sculpt().maskOverlay");
    console.log(`  invert: build ${fmt(inverted.lastMs)} ms; ${inverted.triangles} triangles over ${await maskCount()} masked vertices`);
    await frames();
    verdict(
      "big",
      paint.builds > before.builds && paint.triangles > 0 && inverted.triangles > paint.triangles && paint.maxMs < 50,
      `${count} triangles; overlay build worst ${fmt(paint.maxMs)} ms while painting, ${fmt(inverted.lastMs)} ms for the inverted mask`,
    );
    // What the protection adds to a dab: the same hold under the inverted mask and without one.
    /** Holds a brush at the press point; dabs applied and the window's frames. */
    const holdWindow = async () => {
      const before = (await hook("sculpt()")).dabs;
      const r = await paintWindow(0, 40);
      return { ...r, dabs: (await hook("sculpt()")).dabs - before };
    };
    await clickOn('[data-action="sculpt-mask-paint"]');
    await clickOn('[data-action="sculpt-tool-inflate"]');
    await waitFor(`document.querySelector('[data-action="sculpt-tool-inflate"]')?.getAttribute("aria-pressed") === "true"`);
    await frames();
    const guarded = await holdWindow();
    await clickOn('[data-action="sculpt-mask-clear"]');
    await frames();
    const free = await holdWindow();
    console.log(`  1M dabs on the mask's edge: ${guarded.dabs} applied, ${line(guarded)}`);
    console.log(`  1M dabs without a mask:     ${free.dabs} applied, ${line(free)}`);
    await clickOn('[data-action="sculpt-tool-inflate"]');
    rmSync(dir, { recursive: true, force: true });
  }

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
