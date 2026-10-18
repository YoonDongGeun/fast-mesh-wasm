// Typed-array views over a mesh's render buffers in wasm memory, and the debug
// guard that warns when a caller reads a view kept across a scene call.
import { call, wasmMemory } from "./core.js";
import type { WasmScene } from "./wasm/fast_mesh_wasm.js";

/**
 * Zero-copy views of one mesh's render buffers, slot-aligned.
 *
 * Each getter builds a fresh view over the current wasm memory. A view is
 * only valid until the next scene call that may allocate or move buffers
 * (any edit, add, undo, query, ...): the buffer may have moved, so a kept
 * view may read other data, or be empty once the memory grew. Re-acquire
 * views after every scene call and do not keep them. With
 * `new Scene({ debug: true })` reading a kept view warns.
 */
export interface MeshViews {
  /** Local positions, 3 floats per vertex slot; dead slots hold stale data. */
  positions(): Float32Array;
  /**
   * Vertex normals, 3 floats per vertex slot, or `null` when normals are off.
   * For a `"lazy"` or `"manual"` mesh it can be shorter than 3 · vertex slots
   * after a growth, until the mesh is updated ({@link Scene.update}).
   */
  normals(): Float32Array | null;
  /** RGBA8 colours, 4 bytes per vertex slot, or `null` without colouring. */
  colors(): Uint8Array | null;
  /** 3 vertex indices per triangle slot; a removed triangle reads `0, 0, 0`. */
  indices(): Uint32Array;
}

/**
 * Debug-mode check of kept views. The scene bumps the epoch on every call
 * that may allocate or move buffers; a guarded view is a `Proxy` over the
 * typed array that, on every access, warns (once per guard) when the epoch
 * moved since the view was made, or the memory growth detached it.
 */
export class ViewGuard {
  #epoch = 0;
  #warned = false;

  /** A scene call that may allocate or move buffers is about to run. */
  bump(): void {
    this.#epoch++;
  }

  /** `view`, checked on every access. */
  wrap<T extends ArrayBufferView>(view: T): T {
    const born = this.#epoch;
    const attached = view.byteLength > 0;
    const check = () => {
      if (!this.#warned && (this.#epoch !== born || (attached && view.byteLength === 0))) {
        this.#warned = true;
        console.warn(
          "fast-mesh-wasm: a render view was read after a scene call that may have moved " +
            "or freed its buffer, so it may read other data or be empty. Views are only " +
            "valid until the next scene call: re-acquire them right before use " +
            "(scene.views(mesh).positions() etc.) and do not keep them.",
        );
      }
    };
    return new Proxy(view, {
      get(target, key) {
        check();
        const value: unknown = Reflect.get(target, key, target);
        // Methods (subarray, slice, iteration, ...) need the real array as `this`.
        return typeof value === "function" ? value.bind(target) : value;
      },
      set(target, key, value) {
        check();
        return Reflect.set(target, key, value, target);
      },
      has(target, key) {
        check();
        return Reflect.has(target, key);
      },
    });
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
    return guard === undefined ? view : guard.wrap(view);
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
