// Generates the pictures on the showcase's built-in model buttons
// (examples/showcase/public/previews/<data-action>.webp): loads each model
// through its button in headless Chrome, draws it with the app's own preview
// renderer and writes the result. Run it when a model changes, then rebuild;
// scripts/showcase-check.mjs (the `previews` check) fails on a stale picture.
//
//   node scripts/showcase-previews.mjs --build   # build the example first
import { spawn } from "node:child_process";
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChrome, startServer } from "./chrome-cdp.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const example = join(root, "examples", "showcase");
const vite = join(example, "node_modules", "vite", "bin", "vite.js");
const out = join(example, "public", "previews");
const port = 4196;

if (process.argv.includes("--build")) {
  await new Promise((resolve, reject) => {
    spawn(process.execPath, [vite, "build"], { cwd: example, stdio: "inherit" }).on("exit", (code) =>
      code === 0 ? resolve() : reject(new Error(`vite build exited ${code}`)),
    );
  });
}
const server = await startServer(process.execPath, [vite, "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"], {
  cwd: example,
  url: `http://127.0.0.1:${port}/`,
});
const swiftshader = process.platform === "linux" || process.env.CI !== undefined;
const browser = await launchChrome({ args: swiftshader ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"] : [] });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(expr, ms = 90_000) {
  for (const t = Date.now(); Date.now() - t < ms; await sleep(50)) {
    const v = await browser.evaluate(expr);
    if (v) {
      return v;
    }
  }
  throw new Error(`timed out: ${expr}`);
}

let failed = false;
try {
  await browser.goto(`http://127.0.0.1:${port}/`, "window.__showcase !== undefined");
  await waitFor(`__showcase.items().some((m) => m.drawn)`);
  const actions = await browser.evaluate(`[...document.querySelectorAll(".model-card")].map((b) => b.dataset.action)`);
  if (actions.length === 0) {
    throw new Error("no .model-card button on the page");
  }
  mkdirSync(out, { recursive: true });
  let total = 0;
  for (const action of actions) {
    const from = await browser.evaluate(`__showcase.items().length`);
    await browser.evaluate(`document.querySelector("[data-action=${action}]").click()`);
    await waitFor(`(() => { const all = __showcase.items(); return all.length > ${from} && all.slice(${from}).every((m) => m.drawn) && __showcase.jobs().every((j) => j.phase === "done"); })()`);
    const url = await browser.evaluate(`__showcase.modelPreview(${from}, "image/webp", 0.85)`);
    if (typeof url !== "string" || !url.startsWith("data:image/webp;base64,")) {
      throw new Error(`${action}: no WebP picture`);
    }
    const path = join(out, `${action}.webp`);
    writeFileSync(path, Buffer.from(url.slice(url.indexOf(",") + 1), "base64"));
    const size = statSync(path).size;
    total += size;
    console.log(`${action}.webp: ${size} bytes`);
  }
  console.log(`${actions.length} pictures, ${total} bytes in all`);
} catch (e) {
  console.error(e);
  failed = true;
} finally {
  await browser.close();
  server.stop();
}
process.exit(failed ? 1 : 0);
