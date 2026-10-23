// The threaded engine's pre-flight probe (src/worker/probe.ts): a nested
// Worker grows a shared WebAssembly memory, then this thread fills a byte of
// the new page. WebKit 26 traps there (its bulk memory operations keep the
// size from before another thread's grow), so the pool would trap too. Node
// has no Web Worker: the fakes run the probe's Worker script on this thread,
// or stand in for a broken, silent or failing Worker.
import { describe, expect, it } from "vitest";
import { type ProbeWorker, probeSharedGrowth } from "../src/worker/probe.js";

/** A fake nested Worker; `reply` decides what it does with the probe's message. */
function fakeSpawn(reply: (w: FakeWorker, msg: unknown, source: string) => void) {
  const made: FakeWorker[] = [];
  const spawn = (source: string): ProbeWorker => {
    const w: FakeWorker = {
      onmessage: null,
      onerror: null,
      terminated: false,
      postMessage: (msg) => setTimeout(() => reply(w, msg, source), 0),
      terminate: () => {
        w.terminated = true;
      },
    };
    made.push(w);
    return w;
  };
  return { spawn, made };
}

interface FakeWorker extends ProbeWorker {
  terminated: boolean;
}

/** Runs the probe's Worker script with `self` posting back to `w`, on this thread. */
function runScript(w: FakeWorker, msg: unknown, source: string) {
  const self: { onmessage?: (e: { data: unknown }) => void; postMessage: (m: unknown) => void } = {
    postMessage: (m) => w.onmessage?.({ data: m }),
  };
  new Function("self", source)(self);
  self.onmessage?.({ data: msg });
}

describe("probeSharedGrowth", () => {
  it("healthy_engine_passes", async () => {
    // The script really grows the memory (another instance, same thread: Node
    // has no such bug), so the fill lands in the new page.
    const { spawn, made } = fakeSpawn(runScript);
    expect(await probeSharedGrowth({ spawn, timeoutMs: 1000 })).toBeUndefined();
    expect(made).toHaveLength(1);
    expect(made[0].terminated).toBe(true);
  });

  it("stale_size_after_grow_detected", async () => {
    // As WebKit 26: the other thread grew the memory (it reports the old
    // size, 1 page) but this thread's fill still sees 1 page and traps.
    const { spawn, made } = fakeSpawn((w) => w.onmessage?.({ data: 1 }));
    const why = await probeSharedGrowth({ spawn, timeoutMs: 1000 });
    expect(why).toMatch(/shared WebAssembly memory/);
    expect(made[0].terminated).toBe(true);
  });

  it("inconclusive_cases_allow_threads", async () => {
    // The probe only rules threads out when it saw the bug: a Worker that
    // cannot start, errors, cannot grow or never answers says nothing.
    const cases: Record<string, (w: FakeWorker) => void> = {
      error: (w) => w.onerror?.(new Event("error")),
      growFailed: (w) => w.onmessage?.({ data: -1 }),
      garbage: (w) => w.onmessage?.({ data: "?" }),
      silent: () => {},
    };
    for (const [name, reply] of Object.entries(cases)) {
      const { spawn, made } = fakeSpawn(reply);
      const t0 = performance.now();
      expect(await probeSharedGrowth({ spawn, timeoutMs: 50 }), name).toBeUndefined();
      expect(performance.now() - t0, name).toBeLessThan(1000);
      expect(made[0].terminated, name).toBe(true);
    }
    const throwing = () => {
      throw new DOMException("blocked by worker-src", "SecurityError");
    };
    expect(await probeSharedGrowth({ spawn: throwing, timeoutMs: 50 })).toBeUndefined();
  });

  it("no_nested_workers_skips_probe", async () => {
    // Node (the test harness) has no Web Worker: nothing to probe.
    expect(typeof (globalThis as { Worker?: unknown }).Worker).toBe("undefined");
    expect(await probeSharedGrowth()).toBeUndefined();
  });
});
