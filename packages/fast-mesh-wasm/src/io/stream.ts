/**
 * Streaming mesh format auto-detection, chunked bounded reading, progress
 * reporting, cancellation (`AbortSignal`), and `Scene` / `WorkerScene` loading.
 */
import type { MeshHandle } from "../handles.js";
import type { Scene } from "../scene.js";
import type { WorkerScene } from "../worker/client.js";
import {
  type GltfImportOptions,
  type ImportedMesh,
  importGLTFAsync,
} from "./index.js";
import { importOBJ } from "./obj.js";
import { importPLY } from "./ply.js";
import { importSTL } from "./stl.js";
import { import3MF } from "./threemf.js";
import { DEFAULT_MAX_ZIP_FILE_BYTES } from "./zip.js";

/** Normalized mesh format identifiers supported by the streaming importer. */
export type MeshFileFormat = "threemf" | "ply" | "stl" | "obj" | "gltf" | "glb";

/** Progress event emitted by {@link parseMeshStream} and {@link loadMeshStream}. */
export interface IoProgress {
  /** Bytes read from the input source so far. */
  bytesRead: number;
  /** Expected total byte length when known (e.g. `Blob`/`File`/`Uint8Array`), else `null`. */
  totalBytes: number | null;
  /** Current pipeline stage. */
  stage: "reading" | "parsing" | "building_bvh";
}

/** Supported input sources for {@link parseMeshStream} and {@link loadMeshStream}. */
export type MeshStreamSource =
  | ReadableStream<Uint8Array>
  | Blob
  | Uint8Array
  | ArrayBuffer;

/** Scene targets accepted by {@link loadMeshStream} (sync `Scene` or async `WorkerScene`). */
export type MeshLoadTarget = Scene | WorkerScene;

/** Unified options for {@link parseMeshStream} and {@link loadMeshStream}. */
export interface ImportOptions extends GltfImportOptions {
  /** Explicitly override format detection (accepts `"3mf"` alias for `"threemf"`). */
  format?: MeshFileFormat | "3mf";
  /** Filename or path hint used for extension-based format detection. */
  filename?: string;
  /** Maximum accepted payload size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
  /** Override 3MF millimeter unit scale factor. */
  unitScale?: number;
  /** Weld duplicate vertices when importing STL meshes. Default: `false`. */
  deduplicateVertices?: boolean;
  /** Merge OBJ `o`/`g` groups into a single mesh. Default: `false`. */
  mergeGroups?: boolean;
  /** Optional abort signal to cancel reading, parsing, or scene upload. */
  signal?: AbortSignal;
}

const DEFAULT_MAX_FILE_BYTES = DEFAULT_MAX_ZIP_FILE_BYTES;
const PROGRESS_INTERVAL_MS = 16;
const textDecoder = new TextDecoder("utf-8", { fatal: false });

function createAbortError(signal?: AbortSignal): Error {
  const reason = signal?.reason;
  if (reason instanceof Error) return reason;
  return new DOMException("The operation was aborted.", "AbortError");
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw createAbortError(signal);
  }
}

function combineSignals(
  a?: AbortSignal,
  b?: AbortSignal,
): { signal?: AbortSignal; cleanup: () => void } {
  if (!a && !b) return { signal: undefined, cleanup: () => {} };
  if (a && !b) return { signal: a, cleanup: () => {} };
  if (!a && b) return { signal: b, cleanup: () => {} };
  if (a === b) return { signal: a, cleanup: () => {} };

  const ctrl = new AbortController();
  const onAbortA = () => ctrl.abort(a!.reason);
  const onAbortB = () => ctrl.abort(b!.reason);

  if (a!.aborted) {
    ctrl.abort(a!.reason);
    return { signal: ctrl.signal, cleanup: () => {} };
  }
  if (b!.aborted) {
    ctrl.abort(b!.reason);
    return { signal: ctrl.signal, cleanup: () => {} };
  }

  a!.addEventListener("abort", onAbortA, { once: true });
  b!.addEventListener("abort", onAbortB, { once: true });
  return {
    signal: ctrl.signal,
    cleanup: () => {
      a!.removeEventListener("abort", onAbortA);
      b!.removeEventListener("abort", onAbortB);
    },
  };
}

function formatFromExtension(filename?: string): MeshFileFormat | null {
  if (!filename) return null;
  const clean = filename.split(/[?#]/, 1)[0].trim().toLowerCase();
  const dot = clean.lastIndexOf(".");
  if (dot < 0) return null;
  switch (clean.slice(dot)) {
    case ".3mf":
      return "threemf";
    case ".ply":
      return "ply";
    case ".stl":
      return "stl";
    case ".obj":
      return "obj";
    case ".gltf":
      return "gltf";
    case ".glb":
      return "glb";
    default:
      return null;
  }
}

function hasBinaryBytes(bytes: Uint8Array): boolean {
  const limit = Math.min(bytes.byteLength, 1024);
  for (let i = 0; i < limit; i++) {
    const c = bytes[i];
    if (c === 0 || (c < 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d)) {
      return true;
    }
  }
  return false;
}

/**
 * Detect a mesh file format from header magic bytes and/or filename extension.
 */
export function detectMeshFormat(
  bytesOrFilename?: Uint8Array | ArrayBuffer | string | null,
  filename?: string,
): MeshFileFormat | null {
  if (typeof bytesOrFilename === "string") {
    return formatFromExtension(bytesOrFilename);
  }
  const u8 =
    bytesOrFilename === undefined || bytesOrFilename === null
      ? null
      : bytesOrFilename instanceof Uint8Array
        ? bytesOrFilename
        : new Uint8Array(bytesOrFilename);

  if (u8 && u8.byteLength >= 4) {
    // ZIP local header `PK\x03\x04` or empty ZIP EOCD `PK\x05\x06`
    if (
      u8[0] === 0x50 &&
      u8[1] === 0x4b &&
      ((u8[2] === 0x03 && u8[3] === 0x04) || (u8[2] === 0x05 && u8[3] === 0x06))
    ) {
      return "threemf";
    }
    // Binary glTF magic `"glTF"`
    if (
      u8[0] === 0x67 &&
      u8[1] === 0x6c &&
      u8[2] === 0x54 &&
      u8[3] === 0x46
    ) {
      return "glb";
    }
    // PLY magic `ply\n` or `ply\r`
    if (
      u8[0] === 0x70 &&
      u8[1] === 0x6c &&
      u8[2] === 0x79 &&
      (u8[3] === 0x0a || u8[3] === 0x0d)
    ) {
      return "ply";
    }
  }

  const byExt = formatFromExtension(filename);
  if (byExt !== null) {
    return byExt;
  }

  if (!u8 || u8.byteLength === 0) {
    return null;
  }

  // Exact binary STL size signature (`84 + 50 * N === byteLength`)
  if (u8.byteLength >= 84) {
    const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const triCount = view.getUint32(80, true);
    if (84 + triCount * 50 === u8.byteLength) {
      return "stl";
    }
  }

  if (!hasBinaryBytes(u8)) {
    let head = textDecoder.decode(u8.subarray(0, Math.min(u8.byteLength, 2048)));
    if (head.charCodeAt(0) === 0xfeff) {
      head = head.slice(1);
    }
    const trimmed = head.trimStart();
    if (trimmed.startsWith("{")) {
      return "gltf";
    }
    if (
      trimmed.startsWith("solid") &&
      (trimmed.length === 5 || trimmed.charCodeAt(5) <= 32)
    ) {
      return "stl";
    }
    // Scan first few non-empty lines (OBJ often starts with `#` comments)
    const lines = trimmed.split(/\r?\n/, 20);
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i].trim();
      if (l === "" || l.startsWith("#")) continue;
      if (/^(?:v|vt|vn|vp|f|o|g|mtllib|usemtl|s)\s/.test(l)) {
        return "obj";
      }
      break;
    }
  }

  return null;
}

function normalizeFormatOption(format: MeshFileFormat | "3mf"): MeshFileFormat {
  return format === "3mf" ? "threemf" : format;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function createThrottledEmitter(
  onProgress?: (progress: IoProgress) => void,
): { emit: (progress: IoProgress, force?: boolean) => void } {
  if (!onProgress) {
    return { emit: () => {} };
  }
  let lastEmit = -Infinity;
  return {
    emit(progress: IoProgress, force = false) {
      const now = nowMs();
      if (force || now - lastEmit >= PROGRESS_INTERVAL_MS) {
        lastEmit = now;
        onProgress(progress);
      }
    },
  };
}

function isReadableStream(source: unknown): source is ReadableStream<Uint8Array> {
  return (
    typeof source === "object" &&
    source !== null &&
    typeof (source as ReadableStream<Uint8Array>).getReader === "function"
  );
}

function isBlobLike(source: unknown): source is Blob {
  return (
    typeof source === "object" &&
    source !== null &&
    typeof (source as Blob).size === "number" &&
    typeof (source as Blob).arrayBuffer === "function"
  );
}

async function readStreamSource(
  source: MeshStreamSource,
  maxFileBytes: number,
  emit: (p: IoProgress, force?: boolean) => void,
  signal?: AbortSignal,
): Promise<{ bytes: Uint8Array; bytesRead: number; totalBytes: number | null }> {
  throwIfAborted(signal);

  if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
    const u8 = source instanceof Uint8Array ? source : new Uint8Array(source);
    if (u8.byteLength > maxFileBytes) {
      throw new Error(
        `Mesh payload size (${u8.byteLength} bytes) exceeds limit of ${maxFileBytes} bytes`,
      );
    }
    emit({ bytesRead: u8.byteLength, totalBytes: u8.byteLength, stage: "reading" }, true);
    return { bytes: u8, bytesRead: u8.byteLength, totalBytes: u8.byteLength };
  }

  let totalBytes: number | null = null;
  let stream: ReadableStream<Uint8Array> | null = null;

  if (isBlobLike(source)) {
    totalBytes = source.size;
    if (totalBytes > maxFileBytes) {
      throw new Error(
        `Mesh file size (${totalBytes} bytes) exceeds limit of ${maxFileBytes} bytes`,
      );
    }
    if (typeof source.stream === "function") {
      stream = source.stream() as ReadableStream<Uint8Array>;
    } else {
      const buf = await source.arrayBuffer();
      throwIfAborted(signal);
      const u8 = new Uint8Array(buf);
      if (u8.byteLength > maxFileBytes) {
        throw new Error(
          `Mesh file size (${u8.byteLength} bytes) exceeds limit of ${maxFileBytes} bytes`,
        );
      }
      emit({ bytesRead: u8.byteLength, totalBytes, stage: "reading" }, true);
      return { bytes: u8, bytesRead: u8.byteLength, totalBytes };
    }
  } else if (isReadableStream(source)) {
    stream = source;
  } else {
    throw new Error(
      "parseMeshStream: unsupported source (expected ReadableStream, Blob, Uint8Array, or ArrayBuffer)",
    );
  }

  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let bytesRead = 0;

  const onAbort = () => {
    chunks.length = 0;
    void reader.cancel(createAbortError(signal)).catch(() => {});
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  emit({ bytesRead: 0, totalBytes, stage: "reading" }, true);

  try {
    while (true) {
      throwIfAborted(signal);
      const { value, done } = await reader.read();
      if (done) break;
      throwIfAborted(signal);
      if (value && value.byteLength > 0) {
        if (value.byteLength > maxFileBytes - bytesRead) {
          chunks.length = 0;
          await reader.cancel().catch(() => {});
          throw new Error(
            `Mesh stream exceeds maximum size limit of ${maxFileBytes} bytes`,
          );
        }
        bytesRead += value.byteLength;
        chunks.push(new Uint8Array(value));
        emit({ bytesRead, totalBytes, stage: "reading" });
      }
    }
  } catch (err) {
    chunks.length = 0;
    await reader.cancel(err).catch(() => {});
    throw err;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }

  throwIfAborted(signal);
  const resolvedTotal = totalBytes ?? bytesRead;
  emit({ bytesRead, totalBytes: resolvedTotal, stage: "reading" }, true);

  if (chunks.length === 1 && chunks[0].byteLength === bytesRead) {
    const single = chunks[0];
    chunks.length = 0;
    return { bytes: single, bytesRead, totalBytes: resolvedTotal };
  }

  const assembled = new Uint8Array(bytesRead);
  let off = 0;
  for (let i = 0; i < chunks.length; i++) {
    assembled.set(chunks[i], off);
    off += chunks[i].byteLength;
  }
  chunks.length = 0;
  return { bytes: assembled, bytesRead, totalBytes: resolvedTotal };
}

/**
 * Stream, detect format, and parse a mesh from a `ReadableStream`, `Blob`/`File`,
 * `Uint8Array`, or `ArrayBuffer` without loading Worker/Wasm state.
 */
export async function parseMeshStream(
  source: MeshStreamSource,
  options?: ImportOptions,
  onProgress?: (progress: IoProgress) => void,
  signal?: AbortSignal,
): Promise<ImportedMesh[]> {
  const maxFileBytes = options?.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
  const combined = combineSignals(options?.signal, signal);
  const activeSignal = combined.signal;
  const { emit } = createThrottledEmitter(onProgress);

  let rawBytes: Uint8Array | null = null;
  try {
    const read = await readStreamSource(source, maxFileBytes, emit, activeSignal);
    rawBytes = read.bytes;
    const totalBytes = read.totalBytes;

    throwIfAborted(activeSignal);
    const filename =
      options?.filename ??
      (typeof File !== "undefined" && source instanceof File ? source.name : undefined);
    const format = options?.format
      ? normalizeFormatOption(options.format)
      : detectMeshFormat(rawBytes, filename);
    if (format === null) {
      throw new Error(
        `Unable to detect mesh format${filename ? ` for "${filename}"` : ""}`,
      );
    }

    emit({ bytesRead: read.bytesRead, totalBytes, stage: "parsing" }, true);

    switch (format) {
      case "threemf":
        return await import3MF(rawBytes, {
          maxFileBytes,
          unitScale: options?.unitScale,
          signal: activeSignal,
        });
      case "ply":
        return [
          importPLY(rawBytes, {
            maxFileBytes,
            signal: activeSignal,
          }),
        ];
      case "stl":
        return [
          importSTL(rawBytes, {
            deduplicateVertices: options?.deduplicateVertices,
            maxFileBytes,
            signal: activeSignal,
          }),
        ];
      case "obj":
        return importOBJ(rawBytes, {
          mergeGroups: options?.mergeGroups,
          maxFileBytes,
          signal: activeSignal,
        });
      case "gltf":
      case "glb":
        return await importGLTFAsync(rawBytes, {
          resolveBuffer: options?.resolveBuffer,
          decoderModule: options?.decoderModule,
          wasmBinary: options?.wasmBinary,
        });
    }
  } finally {
    rawBytes = null;
    combined.cleanup();
  }
}

function isWorkerSceneTarget(scene: MeshLoadTarget): scene is WorkerScene {
  const candidate = scene as unknown as Record<string, unknown>;
  return typeof candidate.status === "function" && typeof candidate.idOf === "function";
}

async function raceWithAbort<T>(
  promise: Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  throwIfAborted(signal);
  if (!signal) return promise;

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      reject(createAbortError(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (val) => {
        signal.removeEventListener("abort", onAbort);
        resolve(val);
      },
      (err) => {
        signal.removeEventListener("abort", onAbort);
        reject(err);
      },
    );
  });
}

/**
 * Stream, parse, and upload meshes into either a synchronous {@link Scene} or
 * an asynchronous {@link WorkerScene}, emitting `"reading"`, `"parsing"`, and
 * `"building_bvh"` progress stages. Cleans up partial meshes on abort or error.
 */
export async function loadMeshStream(
  scene: MeshLoadTarget,
  source: MeshStreamSource,
  options?: ImportOptions,
  onProgress?: (progress: IoProgress) => void,
  signal?: AbortSignal,
): Promise<MeshHandle[]> {
  const combined = combineSignals(options?.signal, signal);
  const activeSignal = combined.signal;
  const { emit } = createThrottledEmitter(onProgress);

  let latestBytes = 0;
  let latestTotal: number | null = null;
  const forwardProgress = (p: IoProgress) => {
    latestBytes = p.bytesRead;
    latestTotal = p.totalBytes;
    onProgress?.(p);
  };

  const added: MeshHandle[] = [];
  const workerMode = isWorkerSceneTarget(scene);

  try {
    const parsed = await parseMeshStream(
      source,
      { ...options, signal: activeSignal },
      onProgress ? forwardProgress : undefined,
    );
    throwIfAborted(activeSignal);

    if (onProgress) {
      emit(
        {
          bytesRead: latestBytes,
          totalBytes: latestTotal ?? latestBytes,
          stage: "building_bvh",
        },
        true,
      );
    }

    for (let i = 0; i < parsed.length; i++) {
      throwIfAborted(activeSignal);
      const mesh = parsed[i];
      const handle = scene.addMesh(
        mesh.positions,
        mesh.indices,
        mesh.uvs !== null ? { uvs: mesh.uvs } : undefined,
      );
      added.push(handle);

      if (workerMode) {
        const ws = scene as WorkerScene;
        const measurePromise = ws.measure(handle);
        // Prevent unhandled rejection if raceWithAbort settles first on cancellation
        measurePromise.catch(() => {});
        try {
          await raceWithAbort(measurePromise, activeSignal);
        } catch (err) {
          if (activeSignal?.aborted) {
            throw err;
          }
          try {
            const st = ws.status(`mesh:${ws.idOf(handle)}`);
            if (st.status === "error" && st.error) {
              throw st.error;
            }
          } catch (statusErr) {
            if (statusErr !== err && statusErr instanceof Error) {
              throw statusErr;
            }
          }
          throw err;
        }
      }
    }

    throwIfAborted(activeSignal);
    return added;
  } catch (err) {
    for (let i = 0; i < added.length; i++) {
      try {
        scene.removeMesh(added[i]);
      } catch {
        // Handle may already be gone or never finished adding
      }
    }
    throw err;
  } finally {
    combined.cleanup();
  }
}
