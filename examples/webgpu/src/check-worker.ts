// The Worker of the browser check page (check.ts) for the memory, restart,
// fallback and benchmark checks: the engine served as the package's
// `worker-entry.js` serves it (the same `serveEngine` and `startEngine`, the
// same two WebAssembly artifacts), plus probes the shipped script does not
// have. It imports the package's internal `worker-engine.js` and glue files by
// path, so it needs a built `packages/fast-mesh-wasm/dist`.
//
// Switched on by the script URL's query:
// - `stats`: every chunk of work the engine reports (`onChunk`) is timed from
//   the start of the turn or the end of the previous chunk, and a
//   `MessageChannel` loop records the longest gap between two of its turns
//   (the longest time this Worker's event loop was blocked), as the Node bench
//   harness (`packages/fast-mesh-wasm/bench/worker-harness.mjs`) does.
// - `failNested`: every nested Worker (wasm-bindgen-rayon's pool) loads a
//   script that does not exist, so the pool never starts.
//
// Messages besides the scene's: `{ checkStats: true }` is answered with
// `{ kind: "checkStats", chunks, maxGap, memory }` (chunks and gap since the
// last call; `memory` is the WebAssembly memory's size in bytes),
// `{ checkPoison: true }` throws an uncaught error here, as a crashed Worker
// would, which poisons the scene, and `{ checkPoolCrash: "panic" | "trap" }`
// crashes a thread of the pool (the threaded test build's `__test_panic_pool`
// or `__test_trap_pool`), which blocks this Worker for good: it is answered
// first with `{ kind: "checkPoolCrash", available }` (false when the threaded
// glue is not a test build, or was never loaded).
import { serveEngine, startEngine } from "../../../packages/fast-mesh-wasm/dist/worker-engine.js";

type ThreadedGlue = Awaited<ReturnType<Parameters<typeof startEngine>[1]["importThreaded"]>>;

/** The parts of a dedicated Worker's global scope used here (the example's types are the page's). */
const scope = self as unknown as {
  location: Location;
  postMessage(msg: unknown, transfer?: Transferable[]): void;
  addEventListener(type: "message", cb: (e: MessageEvent) => void): void;
};
const query = new URLSearchParams(scope.location.search);
const timed = query.has("stats");
const now = () => performance.now();

if (query.has("failNested")) {
  const Native = globalThis.Worker;
  globalThis.Worker = class extends Native {
    constructor(_url: string | URL, options?: WorkerOptions) {
      super(new URL("/missing-rayon-helper.js", scope.location.href), options);
    }
  };
}

let memory: WebAssembly.Memory | undefined;
/** The threaded glue, once imported (its test hooks crash the pool). */
let threadedGlue: Record<string, unknown> | undefined;
/** `glue` with its init recording the memory it instantiated with. */
function recorded<T extends { default(): Promise<{ memory: WebAssembly.Memory }> }>(glue: T): T {
  return {
    ...glue,
    default: async () => {
      const out = await glue.default();
      memory = out.memory;
      return out;
    },
  };
}

let chunks: { kind: string; ms: number }[] = [];
let last = now();
let maxGap = 0;
let probeLast = now();
if (timed) {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => {
    const t = now();
    maxGap = Math.max(maxGap, t - probeLast);
    probeLast = t;
    last = t; // nothing of the engine runs in this turn
    channel.port2.postMessage(0);
  };
  channel.port2.postMessage(0);
}

serveEngine(
  {
    post: (msg, transfer) => scope.postMessage(msg, transfer),
    onMessage: (cb) =>
      scope.addEventListener("message", (e) => {
        const data = e.data;
        if (data?.checkStats) {
          scope.postMessage({ kind: "checkStats", chunks, maxGap, memory: memory?.buffer.byteLength ?? 0 });
          chunks = [];
          maxGap = 0;
          probeLast = now();
          return;
        }
        if (data?.checkPoolCrash !== undefined) {
          const hook = data.checkPoolCrash === "trap" ? "__test_trap_pool" : "__test_panic_pool";
          const crash = threadedGlue?.[hook];
          scope.postMessage({ kind: "checkPoolCrash", available: typeof crash === "function" });
          if (typeof crash === "function") {
            // Synchronous on purpose: the reply above is already queued, and
            // yielding (setTimeout) lets the engine messages the page sends
            // next get processed first, so the "pending" calls resolve
            // instead of poisoning (poolCrash flake).
            crash();
          }
          return;
        }
        if (data?.checkPoison) {
          setTimeout(() => {
            throw new Error("check: the Worker crashed (on purpose)");
          });
          return;
        }
        last = now();
        cb(data);
      }),
  },
  (init) =>
    startEngine(init, {
      importThreaded: async () => {
        const glue = await import("../../../packages/fast-mesh-wasm/dist/wasm-threads/fast_mesh_wasm.js");
        threadedGlue = glue as unknown as Record<string, unknown>;
        return recorded(glue as unknown as ThreadedGlue);
      },
      importSingle: async () => recorded(await import("../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm.js")),
    }),
  timed
    ? {
        onChunk: ({ kind }) => {
          const t = now();
          chunks.push({ kind, ms: t - last });
          last = t;
        },
      }
    : {},
);
