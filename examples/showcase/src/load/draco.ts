// The draco3d decoder module (Google's WASM), loaded once and shared by
// `.drc` files and `KHR_draco_mesh_compression` glTF primitives in the
// loader Worker. The `.wasm` rides on Vite's `?url` import, so the build
// emits and hashes it (base-aware); under Vitest (Node) there is no asset
// server, so the fetch falls through to draco3d's own file lookup.
import type { DecoderModule } from "draco3d";
import wasmUrl from "draco3d/draco_decoder.wasm?url";

let cached: Promise<DecoderModule> | null = null;

/** The shared draco3d decoder module, initializing it on first use. */
export function decoderModule(): Promise<DecoderModule> {
  cached ??= load();
  return cached;
}

async function load(): Promise<DecoderModule> {
  const { createDecoderModule } = await import("draco3d");
  let wasmBinary: ArrayBuffer | undefined;
  try {
    const response = await fetch(wasmUrl);
    if (response.ok) {
      wasmBinary = await response.arrayBuffer();
    }
  } catch {
    // No asset server (Vitest, inlined glue): draco3d reads
    // draco_decoder.wasm next to its own files instead.
  }
  return createDecoderModule(wasmBinary === undefined ? {} : { wasmBinary });
}
