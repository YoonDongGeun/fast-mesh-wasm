// Checks the Worker scene's multithreaded engine in a real browser: headless
// Chrome over the DevTools protocol (or Firefox / WebKit through Playwright)
// against `vite preview` of examples/webgpu, whose check page
// (examples/webgpu/check.html, src/check.ts) runs the checks.
//
//   node scripts/browser-check.mjs                 # every check, Chrome
//   node scripts/browser-check.mjs --build         # build the example first
//   node scripts/browser-check.mjs --dev           # against `vite` (dev server) instead
//   node scripts/browser-check.mjs --only compare,restart
//   node scripts/browser-check.mjs --browser firefox|webkit   # needs Playwright (PLAYWRIGHT=<its module path>)
//   node scripts/browser-check.mjs --bench [--n 708] [--kind 9a|native] [--seconds 10] [--lock-step 8] [--threads 8]
//
// Checks (each prints PASS or FAIL; the exit code is 1 on any FAIL):
// - page: the page is cross-origin isolated (vite's COOP/COEP headers).
// - compare: one op sequence (two terrains, wide colouring both ways, drags,
//   an edit, a filled section, two undos) with `threads: 1`, then threaded;
//   every change record and every mirror buffer, placement, field value and
//   section result equal bit for bit, with threads > 1 and == 1.
// - memory: meshes added to a threaded scene until its shared memory passes
//   512 MB (`--memory-mb`); then a field, its colours and a raycast work.
// - restart: a threaded scene's Worker crashes, `restart()` starts a new one
//   with a new pool (threads > 1, it recolours, a drag back gives the first
//   colours bit for bit); Chrome only: the old Worker and its pool are gone,
//   and none is left after `terminate()`.
// - nested: the pool's nested Workers cannot load; the scene starts
//   single-threaded (after the pool timeout), warns once and works.
//
// --bench runs the wide-range drag (bench/scene.bench.mjs 5f) single-threaded
// and threaded instead, and prints both.
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { launchChrome, startServer } from "./chrome-cdp.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const example = join(root, "examples", "webgpu");
const vite = join(example, "node_modules", "vite", "bin", "vite.js");

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i === -1 ? fallback : argv[i + 1];
};
const browserName = option("--browser", "chrome");
const only = option("--only", "page,compare,memory,restart,nested").split(",");
const verbose = flag("--verbose");
const threadsOpt = option("--threads", undefined);
const threads = threadsOpt === undefined ? undefined : Number(threadsOpt);

const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
const mb = (bytes) => `${fmt(bytes / 2 ** 20)} MB`;
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

/** Firefox or WebKit through Playwright, with the shape of `launchChrome` (no Worker list). */
async function launchPlaywright(name) {
  const where = process.env.PLAYWRIGHT ?? "playwright";
  const pw = await import(where.includes("/") || where.includes("\\") ? pathToFileURL(where).href : where);
  const browser = await pw[name].launch();
  const page = await browser.newPage();
  const consoleLines = [];
  page.on("console", (m) => {
    consoleLines.push(`page: ${m.type()}: ${m.text()}`);
    if (verbose) {
      console.log(`  [page] ${m.type()}: ${m.text()}`);
    }
  });
  page.on("pageerror", (e) => consoleLines.push(`page: exception: ${e.message}`));
  return {
    name: `${name} ${browser.version()}`,
    async goto(url, ready) {
      await page.goto(url);
      await page.waitForFunction(ready, undefined, { timeout: 30_000 });
    },
    evaluate: (expression) => page.evaluate(expression),
    workers: undefined,
    consoleLines,
    close: () => browser.close(),
  };
}

if (flag("--build")) {
  await run(process.execPath, [vite, "build"], example);
}
const port = Number(option("--port", flag("--dev") ? 5183 : 4183));
const server = await startServer(
  process.execPath,
  [vite, flag("--dev") ? "serve" : "preview", "--port", String(port), "--strictPort", "--host", "127.0.0.1"],
  { cwd: example, url: `http://127.0.0.1:${port}/check.html` },
);
const browser = browserName === "chrome" ? await launchChrome({ verbose }) : await launchPlaywright(browserName);
const call = (fn, arg = {}) => browser.evaluate(`window.fmCheck.${fn}(${JSON.stringify(arg)})`);
const settleWorkers = async (expected) => {
  // Terminated Workers detach a little later.
  for (let i = 0; i < 50 && browser.workers().length !== expected; i++) {
    await new Promise((r) => setTimeout(r, 100));
  }
  return browser.workers().length;
};

try {
  await browser.goto(`http://127.0.0.1:${port}/check.html`, "window.fmCheck !== undefined");
  if (flag("--dev")) {
    // The dev server's first visit may re-optimize dependencies and reload the
    // page under a running check: load it again once that has settled.
    await new Promise((r) => setTimeout(r, 2000));
    await browser.goto(`http://127.0.0.1:${port}/check.html`, "window.fmCheck !== undefined");
  }
  const info = await call("info");
  console.log(`${browser.name} (${flag("--dev") ? "vite dev" : "vite preview"}); ${info.cores} logical cores`);
  console.log(`  ${info.userAgent}`);

  if (flag("--bench")) {
    const opts = {
      n: Number(option("--n", 708)),
      kind: option("--kind", "9a"),
      seconds: Number(option("--seconds", 10)),
      lockStep: Number(option("--lock-step", 8)),
    };
    for (const t of [1, threads]) {
      const r = await call("bench", { ...opts, threads: t });
      const s = (x) => (x === null ? "-" : `median ${fmt(x.median)} ms (p95 ${fmt(x.p95)}, max ${fmt(x.max)}, n ${x.n})`);
      console.log(
        `wide drag, ${r.triangles.toLocaleString("en")} triangles per mesh (${r.kind} terrain), threads ${r.threads}: set up ${fmt(r.setupMs / 1e3)} s, memory ${mb(r.memoryBytes)}`,
      );
      console.log(`  lock step: move -> colours applied ${s(r.lockStep.moveToColours)}; ${r.lockStep.recoloured} poses recoloured`);
      console.log(`    longest Worker block per pose ${s(r.lockStep.workerBlockPerPose)}`);
      console.log(`    first field step (unit open) ${s(r.lockStep.unitOpen)}; field steps per pose median ${r.lockStep.stepsPerPose.median}`);
      const c = r.continuous;
      console.log(
        `  continuous ${c.hz} Hz for ${c.seconds} s: ${c.moves} moves, ${c.recolouredPoses} poses recoloured, ${c.abandonedUnits} units abandoned`,
      );
      console.log(`    move -> colours applied ${s(c.moveToColours)}`);
      console.log(`    update interval during the drag ${s(c.updateInterval)}`);
      console.log(
        `    release -> last colours applied ${fmt(c.releaseToColours)} ms (recoloured: ${c.releaseRecoloured}); drained ${fmt(c.drainedAfterLastMove)} ms after the last move`,
      );
      console.log(
        `    longest field step ${fmt(c.longestFieldStep)} ms, longest Worker block ${fmt(c.longestWorkerBlock)} ms, most batches waiting ${c.mostBatchesWaiting}; main thread per frame ${s(c.mainPerFrame)}`,
      );
      console.log(`  JSON ${JSON.stringify(r)}`);
    }
  } else {
    if (only.includes("page")) {
      verdict("page", info.isolated && info.sharedArrayBuffer, `crossOriginIsolated ${info.isolated}`);
    }

    if (only.includes("compare")) {
      const r = await call("compare", { n: Number(option("--n", 300)), threads });
      for (const k of ["single", "threaded"]) {
        const x = r[k];
        console.log(
          `  ${k}: threads ${x.threads}, Worker ready ${fmt(x.startMs)} ms, sequence ${fmt(x.runMs)} ms; ${x.frames} frames, ${x.entries} dump entries (${mb(x.bytes)}), digest ${x.digest}`,
        );
      }
      for (const d of r.differences) {
        console.log(`  differs: ${d}`);
      }
      verdict(
        "compare",
        r.identical && r.single.threads === 1 && r.threaded.threads > 1 && r.colouredEntries > 0,
        `${r.triangles.toLocaleString("en")} triangles; threads ${r.single.threads} and ${r.threaded.threads}; ${r.identical ? "bit-identical" : "DIFFERENT"}`,
      );
    }

    if (only.includes("memory")) {
      const target = Number(option("--memory-mb", 512));
      const r = await call("memory", { targetMb: target, threads });
      console.log(
        `  threads ${r.threads}: ${r.meshes} meshes (${r.triangles.toLocaleString("en")} triangles) in ${fmt(r.addMs / 1e3)} s; memory ${mb(r.initialBytes)} -> ${mb(r.finalBytes)} (${r.sizes.map(mb).join(", ")})`,
      );
      verdict(
        "memory",
        r.threads > 1 && r.finalBytes > target * 2 ** 20 && r.fieldCurrent && r.coloured && r.rayHit && r.workerErrors.length === 0,
        `grew to ${mb(r.finalBytes)}; field current ${r.fieldCurrent}, coloured ${r.coloured}, raycast ${r.rayHit}`,
      );
    }

    if (only.includes("restart")) {
      const before = browser.workers ? await settleWorkers(0) : undefined;
      const open = await call("restartOpen", { threads });
      const opened = browser.workers ? await settleWorkers(1 + open.threads) : undefined;
      const r = await call("restartRun");
      const after = browser.workers ? await settleWorkers(1 + r.threads) : undefined;
      await call("close");
      const closed = browser.workers ? await settleWorkers(0) : undefined;
      console.log(
        `  crash: ${JSON.stringify(r.crashErrors)}, calls then fail with ${r.poisoned}; restart ${fmt(r.restartMs)} ms, threads ${open.threads} -> ${r.threads}, Workers made ${r.workersMade}`,
      );
      if (browser.workers) {
        console.log(`  live Workers (scene + pool): before ${before}, opened ${opened}, after restart ${after}, after terminate ${closed}`);
      }
      const counts = browser.workers === undefined || (opened === 1 + open.threads && after === 1 + r.threads && closed === 0);
      verdict(
        "restart",
        open.threads > 1 && r.threads > 1 && r.poisoned === "INSTANCE_POISONED" && r.workersMade === 2 && r.recoloured && r.backToFirst && counts,
        `recoloured ${r.recoloured}, back to the first colours ${r.backToFirst}${browser.workers ? `, Workers ${opened} -> ${after} -> ${closed}` : ""}`,
      );
    }

    if (only.includes("nested")) {
      const r = await call("nestedFail", { threads });
      if (r.rejected !== undefined) {
        console.log(`  createWorkerScene rejected after ${fmt(r.readyMs)} ms: ${r.rejected}`);
        verdict("nested", false, "the scene did not start");
      } else {
        console.log(
          `  ready after ${fmt(r.readyMs)} ms, threads ${r.threads}, Workers made ${r.workersMade}, outer Worker error events ${JSON.stringify(r.outerErrors)}`,
        );
        for (const w of r.warnings) {
          console.log(`  warned: ${w}`);
        }
        verdict(
          "nested",
          r.threads === 1 && r.works && r.warnings.length === 1 && r.outerErrors.every((e) => e.length === 0),
          `single-threaded after ${fmt(r.readyMs / 1e3)} s, ${r.warnings.length} warning, works ${r.works}`,
        );
      }
    }
  }
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
