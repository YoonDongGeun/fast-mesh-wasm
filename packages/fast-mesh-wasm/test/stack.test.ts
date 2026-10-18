// Stack exhaustion inside a call poisons the instance like a trap. In its own
// file because poisoning is module-wide (vitest isolates test files); each
// test loads a fresh copy of the core module so it starts unpoisoned.
import { beforeEach, describe, expect, it, vi } from "vitest";

beforeEach(() => {
  vi.resetModules();
});

async function poisonCode(error: Error): Promise<string> {
  const { call, FastMeshError } = await import("../src/core.js");
  let err: unknown;
  try {
    call(() => {
      throw error;
    });
  } catch (e) {
    err = e;
  }
  expect(err).toBeInstanceOf(FastMeshError);
  // Poisoned: even a call that does nothing throws now.
  expect(() => call(() => 1)).toThrow(FastMeshError);
  return (err as InstanceType<typeof FastMeshError>).code;
}

describe("stack exhaustion", () => {
  it("range_error_from_stack_exhaustion_poisons", async () => {
    // What V8 and JavaScriptCore throw when a call runs out of stack.
    expect(await poisonCode(new RangeError("Maximum call stack size exceeded"))).toBe(
      "INSTANCE_POISONED",
    );
  });

  it("firefox_internal_error_poisons", async () => {
    // Firefox's `InternalError: too much recursion` (no such class elsewhere).
    const e = new Error("too much recursion");
    e.name = "InternalError";
    expect(await poisonCode(e)).toBe("INSTANCE_POISONED");
  });
});
