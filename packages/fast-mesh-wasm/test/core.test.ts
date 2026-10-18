// Error codes and panic poisoning, against the raw bindings. Runs on the
// sources (not dist/) to reach the raw bindings, and needs the test build of
// the .wasm, which exports `__test_panic`: the package's `test` script builds
// it (FAST_MESH_TEST_BUILD=1).
// Poisoning is module-wide and vitest isolates test files, so the panic here
// does not leak into the other files.
import { describe, expect, it } from "vitest";
import { computeBounds, computeVertexNormals, Scene } from "../src/node.js";
import { call, FastMeshError, release, wasmMemory } from "../src/core.js";
import * as raw from "../src/wasm/fast_mesh_wasm.js";

// Only the test build exports it; the cast keeps `tsc` happy on a release build.
const testPanic = (raw as unknown as { __test_panic: () => void }).__test_panic;

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

  it("unknown_code_prefix_passes_through", () => {
    // Only known codes become a FastMeshError; anything else is rethrown as is.
    const foreign = new Error("NOT_A_CODE: something else");
    expect(() =>
      call(() => {
        throw foreign;
      }),
    ).toThrow(foreign);
    let err: unknown;
    try {
      call(() => {
        throw new Error("INVALID_INDEX: index 9");
      });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(FastMeshError);
    expect((err as FastMeshError).code).toBe("INVALID_INDEX");
    // A RangeError that is not stack exhaustion does not poison.
    const range = new RangeError("Invalid typed array length: -1");
    expect(() =>
      call(() => {
        throw range;
      }),
    ).toThrow(range);
    expect(call(() => 1)).toBe(1);
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
    const normals = new Float32Array(memory.buffer, scene.normals_ptr(i, g), scene.normals_len(i, g));
    expect(Array.from(normals)).toEqual(Array.from(computeVertexNormals(positions, indices)));
    scene.free();
  });

  it("panic_poisons_instance", () => {
    const scene = new raw.WasmScene();
    const other = new raw.WasmScene();
    const wrapped = new Scene();
    expect(codeOf(() => call(() => testPanic()))).toBe("INSTANCE_POISONED");
    // Every later call throws the same, the existing functions included.
    expect(codeOf(() => call(() => scene.can_undo()))).toBe("INSTANCE_POISONED");
    expect(codeOf(() => computeBounds([0, 0, 0]))).toBe("INSTANCE_POISONED");
    expect(codeOf(() => wrapped.canUndo())).toBe("INSTANCE_POISONED");
    expect(codeOf(() => new Scene())).toBe("INSTANCE_POISONED");
    // Freeing never enters the poisoned instance: a no-op that also detaches
    // the glue's finalizer (its pointer is cleared).
    wrapped.free();
    expect(() => release(other)).not.toThrow();
    expect((other as unknown as { __wbg_ptr: number }).__wbg_ptr).toBe(0);
  });
});
