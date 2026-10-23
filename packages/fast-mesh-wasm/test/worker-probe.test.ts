// The threaded engine's pre-flight probe (src/worker/probe.ts): a nested
// Worker grows a shared WebAssembly memory, then this thread fills a byte of
// the new page. WebKit 26 traps there (its bulk memory operations keep the
// size from before another thread's grow), so the pool would trap too. Node
// has no Web Worker: the fakes run the probe's Worker script on this thread,
// or stand in for a broken, silent or failing Worker.
import { afterEach, describe, expect, it, vi } from "vitest";
import { type ProbeWorker, probeSharedGrowth } from "../src/worker/probe.js";

/** A fake nested Worker; `reply` decides what it does with the probe's message. */
function fakeSpawn(reply: (w: FakeWorker, msg: unknown, source: string) => void) {
  const made: FakeWorker[] = [];
  const spawn = (source: string): ProbeWorker => {
    const w: FakeWorker = {
      onmessage: null,
      onerror: null,
      onmessageerror: null,
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
  const self = scriptScope((m) => w.onmessage?.({ data: m }));
  new Function("self", source)(self);
  self.onmessage?.({ data: msg });
}

interface ScriptScope {
  onmessage?: (e: { data: unknown }) => void;
  onmessageerror?: () => void;
  postMessage: (m: unknown) => void;
}

function scriptScope(post: (m: unknown) => void): ScriptScope {
  return { postMessage: post };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

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
    const cases: Record<string, (w: FakeWorker, msg: unknown, source: string) => void> = {
      error: (w) => w.onerror?.(new Event("error")),
      // The memory or module could not be passed: answered by either side.
      messageerror: (w) => w.onmessageerror?.(new Event("messageerror")),
      scriptMessageerror: (w, _msg, source) => {
        const self = scriptScope((m) => w.onmessage?.({ data: m }));
        new Function("self", source)(self);
        self.onmessageerror?.();
      },
      growFailed: (w) => w.onmessage?.({ data: -1 }),
      garbage: (w) => w.onmessage?.({ data: "?" }),
      silent: () => {},
    };
    for (const [name, reply] of Object.entries(cases)) {
      const { spawn, made } = fakeSpawn(reply);
      const t0 = performance.now();
      // Answered right away, not after the timeout (except the silent one).
      const timeoutMs = name === "silent" ? 50 : 5000;
      expect(await probeSharedGrowth({ spawn, timeoutMs }), name).toBeUndefined();
      expect(performance.now() - t0, name).toBeLessThan(1000);
      expect(made[0].terminated, name).toBe(true);
    }
    const throwing = () => {
      throw new DOMException("blocked by worker-src", "SecurityError");
    };
    expect(await probeSharedGrowth({ spawn: throwing, timeoutMs: 50 })).toBeUndefined();
  });

  it("blob_url_workers_by_default", async () => {
    // A page's Worker scope: the probe starts a classic Worker on a Blob URL
    // of its script, and revokes the URL when the Worker is terminated.
    const revoked: string[] = [];
    const made: { url: string; terminated: boolean }[] = [];
    let blobs = 0;
    class FakeBlob {
      constructor(
        readonly parts: string[],
        readonly options: { type: string },
      ) {}
    }
    let fail = false;
    class PageWorker {
      onmessage: ((e: { data: unknown }) => void) | null = null;
      onerror: ((e: unknown) => void) | null = null;
      onmessageerror: ((e: unknown) => void) | null = null;
      readonly record: { url: string; terminated: boolean };
      constructor(url: string) {
        if (fail) {
          throw new DOMException("blocked by worker-src (test)", "SecurityError");
        }
        this.record = { url, terminated: false };
        made.push(this.record);
      }
      postMessage(msg: unknown) {
        setTimeout(() => runScript(this as unknown as Parameters<typeof runScript>[0], msg, source), 0);
      }
      terminate() {
        this.record.terminated = true;
      }
    }
    let source = "";
    vi.stubGlobal("Blob", FakeBlob);
    vi.stubGlobal("Worker", PageWorker);
    vi.stubGlobal("URL", {
      createObjectURL: (b: FakeBlob) => {
        source = b.parts.join("");
        expect(b.options.type).toBe("text/javascript");
        return `blob:test/${++blobs}`;
      },
      revokeObjectURL: (u: string) => revoked.push(u),
    });
    expect(await probeSharedGrowth({ timeoutMs: 1000 })).toBeUndefined();
    expect(made).toEqual([{ url: "blob:test/1", terminated: true }]);
    expect(revoked).toEqual(["blob:test/1"]);
    expect(source).toMatch(/onmessageerror/);
    // A Worker that cannot be constructed (a Content Security Policy): its
    // URL is revoked all the same, and the probe says nothing.
    fail = true;
    expect(await probeSharedGrowth({ timeoutMs: 1000 })).toBeUndefined();
    expect(made).toHaveLength(1);
    expect(revoked).toEqual(["blob:test/1", "blob:test/2"]);
  });

  it("no_nested_workers_skips_probe", async () => {
    // Node (the test harness) has no Web Worker: nothing to probe.
    expect(typeof (globalThis as { Worker?: unknown }).Worker).toBe("undefined");
    expect(await probeSharedGrowth()).toBeUndefined();
  });
});
