// The Worker side of the Worker-scene benchmarks: the engine served on a
// `worker_threads` port as the browser Worker script serves it (`serveEngine`,
// a `ready` message first), with two probes for the bench:
//
// - every chunk of work the engine reports (`onChunk`: a request run, a field
//   built, a repair operation) is timed from the start of the turn or the end
//   of the previous chunk, so a chunk's time is an upper bound of how long
//   that one piece of work held the Worker;
// - a `setImmediate` loop records the longest gap between two of its turns,
//   which is the longest time the Worker's event loop was blocked.
//
// The bench asks for the numbers with the message `{ benchStats: true }`
// (never a request batch) and gets `{ kind: "benchStats", chunks, maxGap }`
// back, after which both probes start again.
//
// The engine yields with a `MessageChannel` tick. In a Node Worker a chain of
// those ticks starves `parentPort`: no batch is received, and no
// `setImmediate` runs, until the engine stops (measured: 0 of about 90
// messages sent at 60 Hz arrived during 1.5 s of 4 ms chunks and ticks). A
// browser Worker interleaves the two. So here the tick is a `setImmediate`
// turn, which lets messages in as a browser would: `MessageChannel` is
// replaced, before the engine is created, by a stand-in that delivers through
// `setImmediate`. The engine uses `MessageChannel` only for that tick.
import { parentPort } from "node:worker_threads";

globalThis.MessageChannel = class {
  constructor() {
    this.port1 = { onmessage: null, unref() {} };
    this.port2 = { postMessage: (data) => setImmediate(() => this.port1.onmessage?.({ data })) };
  }
};

await import("../dist/node.js");
const { serveEngine } = await import("../dist/worker-engine.js");

const now = () => performance.now();
let chunks = [];
let last = now();
let maxGap = 0;
let probeLast = now();

function probe() {
  const t = now();
  maxGap = Math.max(maxGap, t - probeLast);
  probeLast = t;
  last = t; // nothing of the engine runs in this turn
  setImmediate(probe);
}
setImmediate(probe);

const endpoint = {
  post: (msg, transfer) => parentPort.postMessage(msg, transfer),
  onMessage: (cb) =>
    parentPort.on("message", (data) => {
      if (data?.benchStats) {
        parentPort.postMessage({ kind: "benchStats", chunks, maxGap });
        chunks = [];
        maxGap = 0;
        probeLast = now();
        return;
      }
      last = now();
      cb(data);
    }),
};

serveEngine(endpoint, Promise.resolve(), {
  onChunk: ({ kind }) => {
    const t = now();
    chunks.push({ kind, ms: t - last });
    last = t;
  },
});
