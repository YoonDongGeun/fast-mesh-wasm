// Rust -> .wasm -> wasm-bindgen glue + .d.ts -> wasm-opt.
// Written in Node instead of bash so the same command works on Windows, macOS and Linux.
import { execFileSync } from "node:child_process";
import { copyFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages", "fast-mesh-wasm");
const outDir = join(pkgDir, "src", "wasm");
const rawWasm = join(root, "target", "wasm32-unknown-unknown", "release", "fast_mesh_wasm.wasm");
const bgWasm = join(outDir, "fast_mesh_wasm_bg.wasm");

function run(cmd, args) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { cwd: root, stdio: "inherit" });
}

run("cargo", ["build", "-p", "fast-mesh-wasm", "--target", "wasm32-unknown-unknown", "--release"]);

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
];
run(process.execPath, [wasmOpt, bgWasm, "-O3", ...features, "-o", bgWasm]);

// npm only picks up README and LICENSE files from the package directory itself.
for (const file of ["README.md", "LICENSE-MIT", "LICENSE-APACHE"]) {
  copyFileSync(join(root, file), join(pkgDir, file));
}
