// Links a fresh checkout's node_modules to a donor checkout's instead of
// reinstalling: every worktree of this repo shares one dependency set, so a
// new worktree only needs links when its lockfile matches the donor's.
//
// Usage (from the new worktree's root, or pass its path):
//   node scripts/link-worktree-deps.mjs [worktree-dir] [--donor <dir>]
//
// The donor defaults to the main worktree (`git worktree list`). `--donor`
// overrides it (testing, or layouts without git worktrees).
//
// Safety: links are created only where nothing exists, only when both
// lockfiles are byte-identical, and never inside the donor itself. A
// mismatch (or a missing donor install) prints what to run (`pnpm install`)
// and exits 0: a setup hook must not fail worktree creation. Junctions on
// Windows (no privilege needed, like pnpm's own links), symlinks elsewhere.
//
// NOTE: a linked tree must not run `pnpm install` (it would write through
// the link into the donor). Re-run `pnpm install` only after the lockfile
// changes — delete the links first.
import { execFileSync } from "node:child_process";
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, symlinkSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const LOCKFILE = "pnpm-lock.yaml";
const TARGET = "node_modules";
/** Deep enough for root + packages/* + examples/* layouts. */
const MAX_DEPTH = 4;

function main(argv) {
  const { target, donor } = parseArgs(argv);
  const root = resolve(target);
  const from = donor === null ? mainWorktree(root) : resolve(donor);
  if (from === null) {
    console.log("link-worktree-deps: no donor worktree found; run `pnpm install` instead.");
    return;
  }
  if (from === root) {
    console.log("link-worktree-deps: this is the donor checkout; nothing to link.");
    return;
  }
  const want = join(root, LOCKFILE);
  const have = join(from, LOCKFILE);
  if (!existsSync(want) || !existsSync(have)) {
    console.log("link-worktree-deps: lockfile missing on one side; run `pnpm install` instead.");
    return;
  }
  if (!readFileSync(want).equals(readFileSync(have))) {
    console.log("link-worktree-deps: lockfiles differ; run `pnpm install` instead of linking.");
    return;
  }
  const kind = process.platform === "win32" ? "junction" : "dir";
  let linked = 0;
  for (const dir of nodeModulesDirs(from)) {
    const dest = join(root, dir.slice(from.length + 1));
    if (!missing(dest)) {
      continue;
    }
    mkdirSync(dirname(dest), { recursive: true });
    symlinkSync(join(from, dir.slice(from.length + 1)), dest, kind);
    linked += 1;
  }
  if (linked === 0) {
    console.log("link-worktree-deps: nothing to link (already installed, or the donor has no install).");
  } else {
    console.log(`link-worktree-deps: linked ${linked} node_modules from ${from}.`);
  }
}

function parseArgs(argv) {
  let target = ".";
  let donor = null;
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--donor") {
      donor = argv[i + 1] ?? null;
      i += 1;
    } else if (!argv[i].startsWith("--")) {
      target = argv[i];
    } else {
      throw new Error(`unknown flag: ${argv[i]}`);
    }
  }
  return { target, donor };
}

/** The main worktree's path, or null outside a worktree layout. */
function mainWorktree(cwd) {
  let out;
  try {
    out = execFileSync("git", ["worktree", "list", "--porcelain"], { cwd, encoding: "utf8" });
  } catch {
    return null;
  }
  for (const line of out.split("\n")) {
    if (line.startsWith("worktree ")) {
      return resolve(cwd, line.slice("worktree ".length).trim());
    }
  }
  return null;
}

/** Every `node_modules` dir under `root`, up to `MAX_DEPTH` deep. */
function nodeModulesDirs(root) {
  const found = [];
  const walk = (dir, depth) => {
    if (depth > MAX_DEPTH) {
      return;
    }
    let entries;
    try {
      entries = listDirs(dir);
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry === TARGET) {
        found.push(join(dir, entry));
      } else {
        walk(join(dir, entry), depth + 1);
      }
    }
  };
  walk(root, 0);
  return found;
}

function listDirs(dir) {
  return readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && !e.isSymbolicLink() && !e.name.startsWith("."))
    .map((e) => e.name);
}

/** True only when nothing (file, dir, link or junction) is at `path`. */
function missing(path) {
  try {
    lstatSync(path);
    return false;
  } catch {
    return true;
  }
}

try {
  main(process.argv.slice(2));
} catch (e) {
  console.error(`link-worktree-deps: ${e.message}`);
  process.exit(1);
}
