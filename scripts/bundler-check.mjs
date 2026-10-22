// Checks that bundlers pick up the multithreaded Worker engine from an
// installed package with no configuration beyond the COOP/COEP headers:
// packs the package, installs the tarball (npm) into fresh scratch apps,
// builds them, serves them with the headers and checks in headless Chrome
// that a Worker scene starts with threads > 1 and computes.
//
//   node scripts/bundler-check.mjs                    # pack, then vite-build, vite-dev, webpack
//   node scripts/bundler-check.mjs vite-build webpack # only these
//   node scripts/bundler-check.mjs --tarball pack/    # an existing tarball, or a folder holding one
//   node scripts/bundler-check.mjs --keep             # keep the scratch apps (printed)
//
// Apps:
// - vite-build: `vite build`, the output served statically with the headers.
// - vite-dev: the `vite` dev server (`server.headers`), which pre-bundles the
//   installed package.
// - webpack: webpack 5 (`mode: "production"`, no loaders or plugins), the
//   output served statically with the headers.
// - nobundler: no bundler: the installed package's dist/ served as is, the
//   page importing `dist/worker.js` by path.
import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChrome, serveStatic, startServer } from "./chrome-cdp.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const keep = args.includes("--keep");
const verbose = args.includes("--verbose");
const tarballFlag = args.indexOf("--tarball");
let tarball = tarballFlag === -1 ? undefined : resolve(args.splice(tarballFlag, 2)[1]);
const apps = args.filter((a) => !a.startsWith("--"));
const selected = apps.length > 0 ? apps : ["vite-build", "vite-dev", "webpack", "nobundler"];

// The versions the repo's examples use.
const VITE = "8.3.4";
const WEBPACK = "5.112.0";

function sh(cmd, cwd) {
  console.log(`$ ${cmd}`);
  execSync(cmd, { cwd, stdio: verbose ? "inherit" : "pipe" });
}

const findTarball = (dir) => join(dir, readdirSync(dir).find((f) => f.endsWith(".tgz")));
if (!tarball) {
  const packDir = mkdtempSync(join(tmpdir(), "fast-mesh-pack-"));
  sh(`pnpm --filter fast-mesh-wasm pack --pack-destination "${packDir}"`, root);
  tarball = findTarball(packDir);
} else if (statSync(tarball).isDirectory()) {
  tarball = findTarball(tarball);
}
console.log(`tarball: ${tarball}`);

// The page's script: a Worker scene with the default Worker URL, two meshes,
// a distance field colouring one, and a query; the outcome in `window.result`.
const MAIN = `import { createWorkerScene } from "fast-mesh-wasm/worker";

function grid(n, lift) {
  const w = n + 1;
  const positions = new Float64Array(3 * w * w);
  const indices = new Uint32Array(6 * n * n);
  for (let j = 0; j < w; j++) {
    for (let i = 0; i < w; i++) {
      positions.set([i, j, lift + Math.sin(i * 0.1) * Math.cos(j * 0.1)], 3 * (j * w + i));
    }
  }
  let t = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * w + i;
      indices.set([a, a + 1, a + w + 1, a, a + w + 1, a + w], t);
      t += 6;
    }
  }
  return { positions, indices };
}

try {
  const scene = await createWorkerScene();
  const a = grid(100, 0);
  const b = grid(100, 3);
  const meshA = scene.addMesh(a.positions, a.indices);
  const meshB = scene.addMesh(b.positions, b.indices);
  const field = scene.addDistanceField({ name: "gap", on: meshA, to: meshB, maxDistance: 8 });
  scene.setColoring(meshA, { field, map: "turbo", range: [0, 6] });
  const gap = scene.meshDistance(meshA, meshB);
  scene.commitFrame();
  const distance = (await gap)?.distance ?? null;
  scene.commitFrame();
  window.result = {
    isolated: crossOriginIsolated,
    threads: scene.threads,
    distance,
    coloured: scene.views(meshA).colors() !== null,
  };
} catch (e) {
  window.result = { error: String(e) };
}
`;
const HTML = (src) =>
  `<!doctype html>\n<html><head><meta charset="UTF-8" /><title>bundler check</title></head>` +
  `<body><script type="module" src="${src}"></script></body></html>\n`;

function makeApp(name, deps) {
  // The long path: on Windows tmpdir() can be an 8.3 short name, which Vite's dev
  // server then refuses to serve as outside its root.
  const dir = realpathSync.native(mkdtempSync(join(tmpdir(), `fast-mesh-bundler-${name}-`)));
  writeFileSync(
    join(dir, "package.json"),
    JSON.stringify({ name: `bundler-check-${name}`, private: true, type: "module", dependencies: deps }, null, 2),
  );
  // npm, as a user would install it; a copy of the tarball keeps the path short.
  copyFileSync(tarball, join(dir, "fast-mesh-wasm.tgz"));
  const install = `npm install --prefer-offline --no-audit --no-fund ./fast-mesh-wasm.tgz ${Object.entries(deps)
    .map(([k, v]) => `${k}@${v}`)
    .join(" ")}`;
  try {
    sh(install, dir);
  } catch (e) {
    // Once more: a registry hiccup should not fail the check.
    const last = e.stderr?.toString().trim().split("\n")[0];
    console.log(`  install failed (${last || e}); retrying once`);
    sh(install, dir);
  }
  return dir;
}

// The examples' setting (examples/webgpu/vite.config.ts): the headers on every
// response, 304s included (Vite's server.headers/preview.headers miss those,
// which Safari needs).
const VITE_CONFIG = `import { defineConfig } from "vite";

const isolate = (_req, res, next) => {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
};

export default defineConfig({
  plugins: [
    {
      name: "cross-origin-isolation",
      configureServer: (server) => void server.middlewares.use(isolate),
      configurePreviewServer: (server) => void server.middlewares.use(isolate),
    },
  ],
  build: { target: "es2022" },
});
`;

const WEBPACK_BUILD = `import { fileURLToPath } from "node:url";
import webpack from "webpack";

webpack(
  {
    mode: "production",
    entry: "./src/main.js",
    output: { path: fileURLToPath(new URL("./dist", import.meta.url)), clean: true },
  },
  (err, stats) => {
    if (err || stats.hasErrors()) {
      console.error(err ?? stats.toString("errors-only"));
      process.exit(1);
    }
    console.log(stats.toString({ assets: true, modules: false, colors: false }));
  },
);
`;

/** Opens `url` in Chrome and returns the page's `window.result`, with the Workers it started. */
async function check(url) {
  const chrome = await launchChrome({ verbose });
  try {
    await chrome.goto(url, "window.result !== undefined");
    const result = await chrome.evaluate("window.result");
    return { ...result, workers: chrome.workers(), console: chrome.consoleLines };
  } finally {
    await chrome.close();
  }
}

let failures = 0;
const dirs = [];
for (const app of selected) {
  console.log(`\n== ${app}`);
  const t0 = Date.now();
  let result;
  try {
    if (app === "vite-build" || app === "vite-dev") {
      const dir = makeApp(app, { vite: VITE });
      dirs.push(dir);
      writeFileSync(join(dir, "vite.config.js"), VITE_CONFIG);
      writeFileSync(join(dir, "index.html"), HTML("/main.js"));
      writeFileSync(join(dir, "main.js"), MAIN);
      const vite = join(dir, "node_modules", "vite", "bin", "vite.js");
      if (app === "vite-build") {
        sh(`node "${vite}" build`, dir);
        const server = await serveStatic(join(dir, "dist"));
        try {
          result = await check(`${server.url}/`);
        } finally {
          await server.close();
        }
      } else {
        const port = 5199;
        const server = await startServer(process.execPath, [vite, "--port", String(port), "--strictPort", "--host", "127.0.0.1"], {
          cwd: dir,
          url: `http://127.0.0.1:${port}/`,
        });
        try {
          result = await check(`http://127.0.0.1:${port}/`);
        } finally {
          server.stop();
        }
      }
    } else if (app === "webpack") {
      const dir = makeApp(app, { webpack: WEBPACK });
      dirs.push(dir);
      mkdirSync(join(dir, "src"));
      writeFileSync(join(dir, "src", "main.js"), MAIN);
      writeFileSync(join(dir, "build.mjs"), WEBPACK_BUILD);
      sh("node build.mjs", dir);
      writeFileSync(join(dir, "dist", "index.html"), HTML("main.js"));
      const server = await serveStatic(join(dir, "dist"));
      try {
        result = await check(`${server.url}/`);
      } finally {
        await server.close();
      }
    } else if (app === "nobundler") {
      const dir = makeApp(app, {});
      dirs.push(dir);
      writeFileSync(join(dir, "index.html"), HTML("/main.js"));
      writeFileSync(
        join(dir, "main.js"),
        MAIN.replace('"fast-mesh-wasm/worker"', '"/node_modules/fast-mesh-wasm/dist/worker.js"'),
      );
      const server = await serveStatic(dir);
      try {
        result = await check(`${server.url}/`);
      } finally {
        await server.close();
      }
    } else {
      throw new Error(`unknown app ${app}: use vite-build, vite-dev, webpack or nobundler`);
    }
  } catch (e) {
    result = { error: e.stderr?.toString() || e.stdout?.toString() || String(e) };
  }
  const ok = result.error === undefined && result.isolated && result.threads > 1 && result.coloured && result.distance !== null;
  if (result.workers) {
    console.log(`  Workers: ${result.workers.join(", ")}`);
  }
  if (!ok && result.console) {
    console.log(result.console.map((l) => `  ${l}`).join("\n"));
  }
  const { workers, console: lines, ...shown } = result;
  console.log(`${ok ? "PASS" : "FAIL"} ${app}: ${JSON.stringify(shown)} (${((Date.now() - t0) / 1e3).toFixed(1)} s)`);
  failures += ok ? 0 : 1;
}

if (keep) {
  console.log(`\nscratch apps kept:\n${dirs.join("\n")}`);
} else {
  for (const dir of dirs) {
    rmSync(dir, { recursive: true, force: true });
  }
}
if (failures > 0) {
  console.log(`${failures} app(s) failed`);
  process.exit(1);
}
