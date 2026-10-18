// Typed-array views over a mesh's render buffers in wasm memory, and the debug
// guard that warns when a caller kept a view the memory growth detached.
import { call, wasmMemory } from "./core.js";
import type { WasmScene } from "./wasm/fast_mesh_wasm.js";

/**
 * Zero-copy views of one mesh's render buffers, slot-aligned.
 *
 * Each getter builds a fresh view over the current wasm memory. A view is
 * only valid until the next scene call that may allocate (any edit, add,
 * undo, ...): growing the memory detaches it and it then reads as empty.
 * Get views right before use and do not keep them.
 */
export interface MeshViews {
  /** Local positions, 3 floats per vertex slot; dead slots hold stale data. */
  positions(): Float32Array;
  /** Vertex normals, 3 floats per vertex slot, or `null` when normals are off. */
  normals(): Float32Array | null;
  /** RGBA8 colours, 4 bytes per vertex slot, or `null` without colouring. */
  colors(): Uint8Array | null;
  /** 3 vertex indices per triangle slot; a removed triangle reads `0, 0, 0`. */
  indices(): Uint32Array;
}

/**
 * Debug-mode tracking of the views a scene handed out. When the memory
 * buffer has changed since the last view, every tracked view that is still
 * alive and now detached (`byteLength === 0`) means a caller kept a view
 * across an allocating call; the first one found warns, once per guard.
 */
export class ViewGuard {
  #buffer: ArrayBuffer | undefined;
  #views: WeakRef<ArrayBufferView>[] = [];
  #warned = false;

  /** Checks the earlier views, then starts tracking `view`. */
  track<T extends ArrayBufferView>(view: T): T {
    const buffer = view.buffer as ArrayBuffer;
    if (buffer !== this.#buffer) {
      if (!this.#warned && this.#views.some((r) => r.deref()?.byteLength === 0)) {
        this.#warned = true;
        console.warn(
          "fast-mesh-wasm: a render view was kept across a scene call that grew the " +
            "WebAssembly memory, so it is detached and reads as empty. Get views right " +
            "before use (scene.views(mesh).positions() etc.) and do not keep them.",
        );
      }
      this.#buffer = buffer;
      this.#views = [];
    } else if (this.#views.length >= 1024) {
      // Bound the list between memory growths: drop collected views.
      this.#views = this.#views.filter((r) => r.deref() !== undefined);
    }
    this.#views.push(new WeakRef(view));
    return view;
  }
}

/**
 * The views of mesh `(index, generation)`. `scene` returns the live raw scene
 * (it throws once the scene is freed); every getter goes through it.
 */
export function meshViews(
  scene: () => WasmScene,
  index: number,
  generation: number,
  guard: ViewGuard | undefined,
): MeshViews {
  // `null` for a buffer that is off (pointer 0): normals and colours.
  const build = <T extends ArrayBufferView>(
    make: (buffer: ArrayBuffer, ptr: number, len: number) => T,
    ptr: (s: WasmScene) => number,
    len: (s: WasmScene) => number,
  ): T | null => {
    const s = scene();
    const p = call(() => ptr(s));
    if (p === 0) {
      return null;
    }
    const view = make(wasmMemory().buffer, p, call(() => len(s)));
    return guard === undefined ? view : guard.track(view);
  };
  const f32 = (b: ArrayBuffer, p: number, n: number) => new Float32Array(b, p, n);
  const u8 = (b: ArrayBuffer, p: number, n: number) => new Uint8Array(b, p, n);
  const u32 = (b: ArrayBuffer, p: number, n: number) => new Uint32Array(b, p, n);
  // Positions and indices always exist: a live slice never has address 0.
  return {
    positions: () =>
      build(
        f32,
        (s) => s.positions_ptr(index, generation),
        (s) => s.positions_len(index, generation),
      ) as Float32Array,
    normals: () =>
      build(
        f32,
        (s) => s.normals_ptr(index, generation),
        (s) => s.normals_len(index, generation),
      ),
    colors: () =>
      build(
        u8,
        (s) => s.colors_ptr(index, generation),
        (s) => s.colors_len(index, generation),
      ),
    indices: () =>
      build(
        u32,
        (s) => s.indices_ptr(index, generation),
        (s) => s.indices_len(index, generation),
      ) as Uint32Array,
  };
}
