// Installs the packed tarball into a fresh project with each package manager and imports it,
// the way a real user would. Proves npm / pnpm / bun all get a working install.
//
//   node scripts/smoke.mjs                      # pack, then test npm, pnpm and bun
//   node scripts/smoke.mjs pnpm bun             # only these managers
//   node scripts/smoke.mjs --tarball pack/ npm  # use an existing tarball or a folder holding one (CI)
import { execSync } from "node:child_process";
import { copyFileSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const tarballFlag = args.indexOf("--tarball");
let tarball = tarballFlag === -1 ? undefined : resolve(args.splice(tarballFlag, 2)[1]);
const managers = args.length > 0 ? args : ["npm", "pnpm", "bun"];

const commands = {
  npm: { install: (t) => `npm install --no-audit --no-fund "${t}"`, run: "node test.mjs" },
  pnpm: { install: (t) => `pnpm add "${t}"`, run: "node test.mjs" },
  bun: { install: (t) => `bun add "${t}"`, run: "bun test.mjs" },
};

function sh(cmd, cwd) {
  console.log(`$ ${cmd}`);
  execSync(cmd, { cwd, stdio: "inherit" });
}

const findTarball = (dir) => join(dir, readdirSync(dir).find((f) => f.endsWith(".tgz")));

if (!tarball) {
  const packDir = mkdtempSync(join(tmpdir(), "fast-mesh-pack-"));
  sh(`pnpm --filter fast-mesh-wasm pack --pack-destination "${packDir}"`, root);
  tarball = findTarball(packDir);
} else if (statSync(tarball).isDirectory()) {
  tarball = findTarball(tarball);
}

for (const pm of managers) {
  const cmd = commands[pm];
  if (!cmd) throw new Error(`Unknown package manager "${pm}". Use one of: ${Object.keys(commands).join(", ")}`);
  const dir = mkdtempSync(join(tmpdir(), `fast-mesh-smoke-${pm}-`));
  try {
    writeFileSync(join(dir, "package.json"), JSON.stringify({ name: "smoke", private: true, type: "module" }));
    copyFileSync(join(root, "fixtures", "smoke", "test.mjs"), join(dir, "test.mjs"));
    sh(cmd.install(tarball), dir);
    sh(cmd.run, dir);
    console.log(`✔ ${pm}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
