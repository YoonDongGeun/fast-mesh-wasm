// The browser entry needs `await init()` first. This runs in its own test file so the
// Node entry (which initializes on import) is not loaded into the same module graph.
import { describe, expect, it } from "vitest";
import { computeBounds, computeVertexNormals, init } from "../dist/web.js";

describe("fast-mesh-wasm (browser entry)", () => {
  it("explains how to fix a call made before init()", () => {
    expect(() => computeBounds([0, 0, 0])).toThrow(
      "fast-mesh-wasm is not initialized: call `await init()` before computeBounds().",
    );
    expect(() => computeVertexNormals([], [])).toThrow("before computeVertexNormals()");
  });

  it("works after init() with the module bytes", async () => {
    const { readFile } = await import("node:fs/promises");
    const bytes = await readFile(new URL("../dist/wasm/fast_mesh_wasm_bg.wasm", import.meta.url));
    await init(bytes);
    expect(computeBounds([0, 0, 0, 1, 2, 3]).max).toEqual([1, 2, 3]);
  });
});
