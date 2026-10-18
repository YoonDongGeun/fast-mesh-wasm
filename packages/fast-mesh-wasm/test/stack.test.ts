// Stack exhaustion inside a call poisons the instance like a trap. In its own
// file because poisoning is module-wide (vitest isolates test files).
import { describe, expect, it } from "vitest";
import { call, FastMeshError } from "../src/core.js";
import { Scene } from "../src/node.js";

describe("stack exhaustion", () => {
  it("range_error_from_stack_exhaustion_poisons", () => {
    const scene = new Scene();
    let err: unknown;
    try {
      // What V8 throws when a call runs out of stack, possibly mid-WebAssembly.
      call(() => {
        throw new RangeError("Maximum call stack size exceeded");
      });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(FastMeshError);
    expect((err as FastMeshError).code).toBe("INSTANCE_POISONED");
    expect(() => scene.canUndo()).toThrow(FastMeshError);
    scene.free();
  });
});
