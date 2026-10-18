// Error codes and panic poisoning, against the raw bindings. Runs on the
// sources (not dist/) because the scene class is not public yet, and needs
// the test build of the .wasm, which exports `__test_panic`: the package's
// `test` script builds it (FAST_MESH_TEST_BUILD=1).
// Poisoning is module-wide and vitest isolates test files, so the panic here
// does not leak into the other files.
import { describe, expect, it } from "vitest";
import { computeBounds } from "../src/node.js";
import { call, FastMeshError, wasmMemory } from "../src/core.js";
import * as raw from "../src/wasm/fast_mesh_wasm.js";

const quad = new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const tris = new Uint32Array([0, 1, 2, 0, 2, 3]);

function codeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected a throw");
}

describe("core", () => {
  it("error_has_code", () => {
    const scene = new raw.WasmScene();
    const [index, generation] = call(() => scene.add_mesh(quad, tris, undefined, 0, true));
    call(() => scene.remove_mesh(index, generation));
    let err: unknown;
    try {
      call(() => scene.remove_mesh(index, generation));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(FastMeshError);
    expect(err).toBeInstanceOf(Error);
    expect((err as FastMeshError).code).toBe("UNKNOWN_MESH");
    expect((err as FastMeshError).message).toBe("unknown or removed mesh id");
    expect(codeOf(() => call(() => scene.end_step()))).toBe("STEP_NOT_OPEN");
    scene.free();
  });

  it("views read wasm memory in place", () => {
    const scene = new raw.WasmScene();
    const [i, g] = call(() => scene.add_mesh(quad, tris, undefined, 0, true));
    const memory = wasmMemory();
    const indices = new Uint32Array(memory.buffer, scene.indices_ptr(i, g), scene.indices_len(i, g));
    expect(Array.from(indices)).toEqual(Array.from(tris));
    const positions = new Float32Array(
      memory.buffer,
      scene.positions_ptr(i, g),
      scene.positions_len(i, g),
    );
    // Local coordinates around the bounding-box centre.
    expect(Array.from(positions.subarray(0, 3))).toEqual([-0.5, -0.5, 0]);
    scene.free();
  });

  it("panic_poisons_instance", () => {
    const scene = new raw.WasmScene();
    expect(codeOf(() => call(() => raw.__test_panic()))).toBe("INSTANCE_POISONED");
    // Every later call throws the same, the existing functions included.
    expect(codeOf(() => call(() => scene.can_undo()))).toBe("INSTANCE_POISONED");
    expect(codeOf(() => computeBounds([0, 0, 0]))).toBe("INSTANCE_POISONED");
  });
});
