// The probe's verdict is cached per page: once a Worker reports that the
// threaded engine cannot run in this browser (`probeFailed` on its ready
// message), that scene's restarts and the page's later scenes start their
// Workers with `threads: 1`, so the probe never runs again; in development
// each scene still says why, once. Its own file: the cache is module state
// of the client, which must start empty here and is never reset.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import type { InitMessage } from "../src/worker/protocol.js";

/** A harness Worker (test/harness/node-worker.mjs) posing as an isolated page. */
class HarnessWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: unknown[] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];

  constructor(workerData: Record<string, unknown>) {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true, isolated: true, threaded: "fake", ...workerData },
    });
    this.worker.on("message", (data: { kind?: string }) => {
      if (data.kind === "pool" || data.kind === "watch") {
        return;
      }
      for (const cb of this.#message) {
        cb({ data });
      }
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.sent.push(msg);
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error" | "messageerror", cb: (e: any) => void): void {
    if (type === "message") {
      this.#message.push(cb);
    }
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const opened: WorkerScene[] = [];
afterEach(() => {
  for (const s of opened.splice(0)) {
    s.terminate();
  }
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function open(workerData: Record<string, unknown>, options: { threads?: number; debug?: boolean }) {
  const workers: HarnessWorker[] = [];
  const scene = await createWorkerScene({
    ...options,
    createWorker: () => {
      const w = new HarnessWorker(workerData);
      workers.push(w);
      return w;
    },
  });
  opened.push(scene);
  return { scene, workers };
}

const init = (w: HarnessWorker) => w.sent[0] as InitMessage;

describe("probe verdict cached per page", () => {
  it("later_workers_skip_probe", async () => {
    vi.stubGlobal("crossOriginIsolated", true);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    // A scene before the probe failed anywhere: threaded as asked.
    const before = await open({}, { threads: 4, debug: true });
    expect(before.scene.threads).toBe(4);
    expect(warn).not.toHaveBeenCalled();

    // The probe fails: single-threaded, one warning (the Worker's).
    const first = await open({ probeFails: true }, { threads: 4, debug: true });
    expect(first.scene.threads).toBe(1);
    expect(init(first.workers[0]).threads).toBe(4);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toMatch(/cannot run in this browser/);
    // Its restart asks for one thread: no probe, no second warning.
    await first.scene.restart();
    expect(first.workers).toHaveLength(2);
    expect(init(first.workers[1]).threads).toBe(1);
    expect(first.scene.threads).toBe(1);
    expect(warn).toHaveBeenCalledTimes(1);

    // A later scene on the page starts at one thread (its Worker would pass
    // the probe here: it is not asked), and says why, once.
    const later = await open({}, { threads: 4, debug: true });
    expect(init(later.workers[0]).threads).toBe(1);
    expect(later.scene.threads).toBe(1);
    expect(warn).toHaveBeenCalledTimes(2);
    expect(String(warn.mock.calls[1][0])).toMatch(/cannot run in this browser/);
    expect(String(warn.mock.calls[1][0])).toMatch(/Multithreading \(COOP\/COEP\)/);
    await later.scene.restart();
    expect(warn).toHaveBeenCalledTimes(2);

    // Default threads too. `threads: 1` has nothing to explain.
    const fallback = await open({}, {});
    expect(init(fallback.workers[0]).threads).toBe(1);
    const warned = warn.mock.calls.length;
    const explicit = await open({}, { threads: 1, debug: true });
    expect(init(explicit.workers[0]).threads).toBe(1);
    expect(warn).toHaveBeenCalledTimes(warned);
    // The scene opened before keeps its pool, across restarts too.
    await before.scene.restart();
    expect(before.scene.threads).toBe(4);
  });
});
