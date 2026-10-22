// Rust -> .wasm -> wasm-bindgen glue + .d.ts -> wasm-opt.
// Written in Node instead of bash so the same command works on Windows, macOS and Linux.
//
// With FAST_MESH_TEST_BUILD=1 (or `--test`) the crate is built with the `test-panic`
// feature, which adds the `__test_panic` export the poisoning test needs. The output
// goes to the same place, so a test build must never be published: the package's
// `test` script makes one, `pnpm build` and `pnpm release` make a release build, and
// `prepublishOnly` refuses a build that exports `__test_panic`.
//
// With `--threads` it builds the threaded artifact instead, into src/wasm-threads: the
// crate's `threads` feature (rayon + wasm-bindgen-rayon's `initThreadPool`) on the
// nightly pinned below, with wasm atomics, a rebuilt std and a shared, imported memory.
// It is always a release build, also next to a `--test` build of src/wasm. Without the
// nightly (or its rust-src) it warns and skips, unless CI is set.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// The nightly for the threaded artifact (`-Z build-std` and wasm atomics are
// nightly-only). The only place it is set: CI reads it from here too
// (`node scripts/build-wasm.mjs --print-nightly`).
export const NIGHTLY = "nightly-2026-10-15";

if (process.argv.includes("--print-nightly")) {
  console.log(NIGHTLY);
  process.exit(0);
}

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages", "fast-mesh-wasm");
const threads = process.argv.includes("--threads");
const outDir = join(pkgDir, "src", threads ? "wasm-threads" : "wasm");
// Its own target dir, so the nightly's rebuilt std never invalidates the stable build's cache.
const targetDir = join(root, "target", ...(threads ? ["wasm-threads"] : []));
const rawWasm = join(targetDir, "wasm32-unknown-unknown", "release", "fast_mesh_wasm.wasm");
const bgWasm = join(outDir, "fast_mesh_wasm_bg.wasm");

function run(cmd, args, env) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { cwd: root, stdio: "inherit", env: env ? { ...process.env, ...env } : process.env });
}

// The nightly and its rust-src (for `-Z build-std`) must be installed. Locally a missing
// toolchain only skips the threaded artifact, loudly, so contributors without the nightly
// can still build and test the rest; under CI it is an error. prepublishOnly still refuses
// a dist without dist/wasm-threads.
if (threads) {
  const problem = nightlyProblem();
  if (problem) {
    const install = `rustup toolchain install ${NIGHTLY} --profile minimal --component rust-src --target wasm32-unknown-unknown`;
    const message = [
      `${problem}, so the threaded artifact (src/wasm-threads) cannot be built.`,
      `Install it with:`,
      `  ${install}`,
    ].join("\n");
    if (process.env.CI) {
      console.error(`error: ${message}`);
      process.exit(1);
    }
    // Drop a stale artifact, so dist never ships one that no longer matches the crate.
    rmSync(outDir, { recursive: true, force: true });
    const bar = "!".repeat(78);
    console.warn(`${bar}\nwarning: SKIPPING the threaded artifact. ${message}\n${bar}`);
    process.exit(0);
  }
}

function nightlyProblem() {
  let sysroot;
  try {
    sysroot = execFileSync("rustup", ["run", NIGHTLY, "rustc", "--print", "sysroot"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      // Never let rustup install the toolchain as a side effect of this check.
      env: { ...process.env, RUSTUP_AUTO_INSTALL: "0" },
    }).trim();
  } catch {
    return `The pinned nightly ${NIGHTLY} is not installed`;
  }
  if (!existsSync(join(sysroot, "lib", "rustlib", "src"))) {
    return `The pinned nightly ${NIGHTLY} has no rust-src component`;
  }
  return undefined;
}

const testBuild = !threads && (process.env.FAST_MESH_TEST_BUILD === "1" || process.argv.includes("--test"));
const cargoArgs = ["build", "-p", "fast-mesh-wasm", "--target", "wasm32-unknown-unknown", "--release"];
if (threads) {
  // wasm-bindgen-rayon's setup (its README and wasm-bindgen's threads guide): atomics
  // and bulk memory in every crate including std, and a shared memory imported from
  // JavaScript, so the rayon Workers can instantiate the same module on it. The TLS
  // exports let wasm-bindgen give each thread its own thread-local storage.
  // `--max-memory` is 4 GiB (65536 pages), wasm32's limit: a shared memory needs a
  // maximum, and 10M-triangle scenes must still fit.
  const rustflags = [
    "-C target-feature=+atomics,+bulk-memory,+mutable-globals",
    "-C link-arg=--shared-memory",
    "-C link-arg=--max-memory=4294967296",
    "-C link-arg=--import-memory",
    "-C link-arg=--export=__wasm_init_tls",
    "-C link-arg=--export=__tls_size",
    "-C link-arg=--export=__tls_align",
    "-C link-arg=--export=__tls_base",
  ].join(" ");
  run(
    "rustup",
    ["run", NIGHTLY, "cargo", ...cargoArgs, "--features", "threads", "--target-dir", targetDir, "-Z", "build-std=panic_abort,std"],
    { RUSTFLAGS: rustflags },
  );
} else {
  run("cargo", testBuild ? [...cargoArgs, "--features", "test-panic"] : cargoArgs);
}

rmSync(outDir, { recursive: true, force: true });
run("wasm-bindgen", [rawWasm, "--out-dir", outDir, "--target", "web"]);

// binaryen ships wasm-opt as a Node script, so no native install is needed.
const require = createRequire(import.meta.url);
const wasmOpt = join(dirname(require.resolve("binaryen/package.json")), "bin", "wasm-opt");
// Enable exactly the features rustc emits for wasm32-unknown-unknown by default.
// `--all-features` would let wasm-opt use proposals that Node and browsers do not ship yet.
const features = [
  "--enable-bulk-memory",
  "--enable-multivalue",
  "--enable-mutable-globals",
  "--enable-nontrapping-float-to-int",
  "--enable-reference-types",
  "--enable-sign-ext",
  ...(threads ? ["--enable-threads"] : []),
];
run(process.execPath, [wasmOpt, bgWasm, "-O3", ...features, "-o", bgWasm]);

if (threads) {
  // wasm-bindgen-rayon's worker helper (snippets/wasm-bindgen-rayon-*/src/workerHelpers.js)
  // imports the glue as the directory `../../..`, the way wasm-pack lays out a package;
  // bundlers resolve that through this `main`. Its `sideEffects` (`**`: the helper sits
  // three levels down, in snippets/<crate>/src/) keeps the helper's top-level message
  // handler, which runs when it is loaded as a Worker.
  const nested = { type: "module", main: "fast_mesh_wasm.js", sideEffects: ["./snippets/**"] };
  writeFileSync(join(outDir, "package.json"), `${JSON.stringify(nested, null, 2)}\n`);
}

// npm only picks up README and LICENSE files from the package directory itself.
for (const file of ["README.md", "LICENSE-MIT", "LICENSE-APACHE"]) {
  copyFileSync(join(root, file), join(pkgDir, file));
}
