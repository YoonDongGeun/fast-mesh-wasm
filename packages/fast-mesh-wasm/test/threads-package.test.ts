// The threaded artifact in dist/wasm-threads: built, with shared memory up to
// 4 GiB, the same WasmScene API as dist/wasm, and reachable only from the
// Worker entry (worker-entry.js), never from a core entry.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const pkgDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(pkgDir, "dist");
const threadsDir = join(dist, "wasm-threads");

// Locally, without the pinned nightly, `build-wasm.mjs --threads` skips the artifact
// (with a warning), so these skip too. Under CI it must exist.
describe.skipIf(!existsSync(threadsDir) && !process.env.CI)("threaded artifact (dist/wasm-threads)", () => {
  it("threaded_artifact_built", async () => {
    const glue = readFileSync(join(threadsDir, "fast_mesh_wasm.js"), "utf8");
    expect(glue).toMatch(/export function initThreadPool\(/);
    expect(glue).toMatch(/export \{[^}]*__wbg_init as default[^}]*\}/);
    expect(readFileSync(join(threadsDir, "fast_mesh_wasm.d.ts"), "utf8")).toMatch(
      /export function initThreadPool\(num_threads: number\): Promise<any>;/,
    );

    const bytes = readFileSync(join(threadsDir, "fast_mesh_wasm_bg.wasm"));
    const module = await WebAssembly.compile(bytes);
    const memories = WebAssembly.Module.imports(module).filter((i) => i.kind === "memory");
    expect(memories).toHaveLength(1);
    // `Module.imports` gives no limits, so read them from the import section.
    expect(importedMemoryLimits(bytes)).toEqual({ shared: true, initial: expect.any(Number), maximum: 65536 });
  });

  it("ships the rayon worker helpers with their relative imports intact", () => {
    const snippets = join(threadsDir, "snippets");
    const rayon = readdirSync(snippets).filter((d) => d.startsWith("wasm-bindgen-rayon-"));
    expect(rayon).toHaveLength(1);
    const helpers = join(snippets, rayon[0]!, "src", "workerHelpers.js");
    const source = readFileSync(helpers, "utf8");
    // The glue imports the helpers by this relative path.
    const glue = readFileSync(join(threadsDir, "fast_mesh_wasm.js"), "utf8");
    expect(glue).toContain(`./snippets/${rayon[0]}/src/workerHelpers.js`);
    // The helper spawns itself and imports the glue as the directory `../../..`,
    // which resolves through dist/wasm-threads/package.json `main`.
    expect(source).toContain("new URL('./workerHelpers.js', import.meta.url)");
    expect(source).toContain("import('../../..')");
    const pkgRoot = resolve(dirname(helpers), "../../..");
    expect(pkgRoot).toBe(threadsDir);
    const nested = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8")) as {
      main?: string;
      sideEffects?: string[];
    };
    expect(nested.main).toBe("fast_mesh_wasm.js");
    // The helper is three levels down (snippets/<crate>/src/): `*` would not match it.
    expect(nested.sideEffects).toEqual(["./snippets/**"]);
    expect(existsSync(join(pkgRoot, nested.main!))).toBe(true);
  });

  it("package_threads_boundary", () => {
    const pkg = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8")) as { exports: Record<string, unknown> };
    const entries = new Set<string>();
    const collect = (value: unknown): void => {
      if (typeof value === "string") {
        if (value.endsWith(".js")) entries.add(join(pkgDir, value));
      } else if (value && typeof value === "object") {
        for (const v of Object.values(value)) collect(v);
      }
    };
    collect(pkg.exports);
    expect(entries.size).toBeGreaterThanOrEqual(7);
    expect([...entries].some((e) => e.endsWith("worker-entry.js"))).toBe(false);

    for (const entry of entries) {
      const seen = new Set<string>();
      const walk = (file: string): void => {
        if (seen.has(file)) return;
        seen.add(file);
        const source = readFileSync(file, "utf8");
        // URLs passed to `new URL(..., import.meta.url)`: assets and the Worker
        // script. Checked, not followed: the Worker runs in its own graph.
        for (const m of source.matchAll(/new URL\(\s*["'`]([^"'`]+)["'`]\s*,\s*import\.meta\.url/g)) {
          const target = resolve(dirname(file), m[1]!);
          expect(inThreads(target), `${rel(file)} references ${m[1]} (from ${rel(entry)})`).toBe(false);
        }
        for (const spec of importSpecifiers(source)) {
          if (!spec.startsWith(".")) continue;
          const target = resolve(dirname(file), spec);
          expect(inThreads(target), `${rel(file)} imports ${spec} (from ${rel(entry)})`).toBe(false);
          if (target.endsWith(".js")) walk(target);
        }
      };
      walk(entry);
    }
  });

  it("same_api_both_artifacts", () => {
    const single = wasmSceneMethods(readFileSync(join(dist, "wasm", "fast_mesh_wasm.d.ts"), "utf8"));
    const threaded = wasmSceneMethods(readFileSync(join(threadsDir, "fast_mesh_wasm.d.ts"), "utf8"));
    expect(single.length).toBeGreaterThan(10);
    expect(threaded).toEqual(single);
  });
});

function rel(file: string): string {
  return relative(dist, file);
}

function inThreads(path: string): boolean {
  return path === threadsDir || path.startsWith(threadsDir + sep);
}

/** Specifiers of static `import`/`export ... from` and dynamic `import()`. */
function importSpecifiers(source: string): string[] {
  const specs: string[] = [];
  for (const m of source.matchAll(/\b(?:import|export)\s*(?:[\w*{}\s,$]+\s*from\s*)?["']([^"']+)["']/g)) specs.push(m[1]!);
  for (const m of source.matchAll(/\bimport\(\s*["'`]([^"'`]+)["'`]\s*\)/g)) specs.push(m[1]!);
  return specs;
}

/** Method and accessor names declared in `export class WasmScene { ... }`. */
function wasmSceneMethods(dts: string): string[] {
  const start = dts.indexOf("export class WasmScene {");
  expect(start).toBeGreaterThanOrEqual(0);
  const body = dts.slice(start, dts.indexOf("\n}\n", start));
  const names = new Set<string>();
  for (const m of body.matchAll(/^ {4}(?:static |get |set )?(\w+)\(/gm)) names.add(m[1]!);
  return [...names].sort();
}

/** Limits of the (single) imported memory, read from the wasm import section. */
function importedMemoryLimits(bytes: Uint8Array): { shared: boolean; initial: number; maximum: number | undefined } {
  let pos = 8;
  const u32 = (): number => {
    let result = 0;
    let shift = 0;
    for (;;) {
      const byte = bytes[pos++]!;
      result += (byte & 0x7f) * 2 ** shift;
      shift += 7;
      if ((byte & 0x80) === 0) return result;
    }
  };
  const name = (): string => {
    const len = u32();
    const s = new TextDecoder().decode(bytes.subarray(pos, pos + len));
    pos += len;
    return s;
  };
  while (pos < bytes.length) {
    const id = bytes[pos++]!;
    const size = u32();
    const end = pos + size;
    if (id === 2) {
      const count = u32();
      for (let i = 0; i < count; i++) {
        name();
        name();
        const kind = bytes[pos++]!;
        if (kind === 0x00) u32(); // function: type index
        else if (kind === 0x01) {
          pos++; // table: reftype, then limits
          const flags = u32();
          u32();
          if (flags & 1) u32();
        } else if (kind === 0x02) {
          const flags = u32();
          const initial = u32();
          const maximum = flags & 1 ? u32() : undefined;
          return { shared: (flags & 2) !== 0, initial, maximum };
        } else if (kind === 0x03) pos += 2; // global: valtype, mutability
        else if (kind === 0x04) {
          pos++; // tag: attribute, type index
          u32();
        } else throw new Error(`unknown import kind ${kind}`);
      }
    }
    pos = end;
  }
  throw new Error("no imported memory");
}
