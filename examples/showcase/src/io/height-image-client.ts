// The page's side of the height image Worker (#267): the jobs of
// `height-image.ts` as promises. The Worker starts with the first job, and a
// new one with the next job after it failed. Where there is no Worker (tests)
// the jobs run on the page.
import {
  type GridOptions,
  HeightImageEngine,
  type HeightImageInfo,
  type HeightImageReply,
  type HeightImageRequest,
} from "./height-image";

type Job = HeightImageRequest extends infer R ? (R extends { n: number } ? Omit<R, "n"> : never) : never;

interface Pending {
  resolve: (value: never) => void;
  reject: (error: Error) => void;
}

let worker: Worker | null = null;
let local: HeightImageEngine | null = null;
let next = 1;
const pending = new Map<number, Pending>();

function fail(failed: Worker, message: string): void {
  failed.terminate();
  if (worker !== failed) {
    return;
  }
  worker = null;
  for (const p of pending.values()) {
    p.reject(new Error(message));
  }
  pending.clear();
}

function workerFor(): Worker {
  if (worker === null) {
    const made = new Worker(new URL("./height-image.worker.ts", import.meta.url), { type: "module" });
    made.onmessage = (e: MessageEvent<HeightImageReply>) => {
      const p = pending.get(e.data.n);
      pending.delete(e.data.n);
      if (e.data.ok) {
        p?.resolve(e.data.value as never);
      } else {
        p?.reject(new Error(e.data.message));
      }
    };
    made.onerror = (e) => fail(made, e.message || "the image Worker failed");
    made.onmessageerror = () => fail(made, "an image Worker reply could not be read");
    worker = made;
  }
  return worker;
}

function post<T>(job: Job, transfer: Transferable[] = []): Promise<T> {
  const n = next++;
  return new Promise<T>((resolve, reject) => {
    const target = workerFor();
    pending.set(n, { resolve: resolve as never, reject });
    target.postMessage({ ...job, n }, transfer);
  });
}

function engine(): HeightImageEngine | null {
  if (typeof Worker !== "undefined") {
    return null;
  }
  local ??= new HeightImageEngine();
  return local;
}

/** Heights in `[0, range]` as a 16-bit PNG's bytes. `heights` is handed over: its buffer is detached. */
export function encodeHeightPng(
  heights: Float32Array,
  width: number,
  height: number,
  range: number,
): Promise<Uint8Array> {
  return (
    engine()?.encode(heights, width, height, range) ??
    post({ op: "encode", heights, width, height, range }, [heights.buffer as ArrayBuffer])
  );
}

/** Reads an image file; `closeHeightImage` frees it again. Rejects with a message for the user. */
export function openHeightImage(blob: Blob): Promise<HeightImageInfo> {
  return engine()?.open(blob) ?? post({ op: "open", blob });
}

/** The heights of a `w` by `h` grid over an open image. */
export function heightImageGrid(id: number, w: number, h: number, options: GridOptions): Promise<Float32Array> {
  const here = engine();
  if (here !== null) {
    return new Promise((resolve) => resolve(here.grid(id, w, h, options)));
  }
  return post({ op: "grid", id, w, h, options });
}

/** Frees an open image. */
export function closeHeightImage(id: number): void {
  const here = engine();
  if (here !== null) {
    here.close(id);
  } else if (worker !== null) {
    // A Worker that failed took its images with it.
    void post({ op: "close", id }).catch(() => {});
  }
}
