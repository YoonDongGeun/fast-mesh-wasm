// Decoder of the change message `WasmScene.take_changes()` returns. The
// layout is documented in crates/fast-mesh-wasm/src/render.rs; this mirrors it.

/** `[start, count]` in elements: vertex slots, or triangle slots for indices. */
export type Range = readonly [start: number, count: number];

const ADDED = 1;
const REMOVED = 2;
const PLACEMENT = 4;
const CAPACITY = 8;
const POSITIONS_ALL = 16;
const NORMALS_ALL = 32;
const COLORS_ALL = 64;
const INDICES_ALL = 128;

/** One mesh's record, with raw ids; the scene turns them into handles. */
export interface RawChange {
  index: number;
  generation: number;
  added: boolean;
  removed: boolean;
  placementChanged: boolean;
  capacityChanged: boolean;
  positions: Range[] | "all";
  normals: Range[] | "all";
  colors: Range[] | "all";
  indices: Range[] | "all";
  fields: number[];
  sections: number[];
}

/** Decodes `[record_count, record...]`. */
export function decodeChanges(msg: Uint32Array): RawChange[] {
  let at = 0;
  const next = (): number => {
    if (at >= msg.length) {
      throw new Error("fast-mesh-wasm: truncated change message");
    }
    return msg[at++];
  };
  // A range list, or "all" when the flag is set (its list is then empty).
  const ranges = (all: boolean): Range[] | "all" => {
    const n = next();
    const out: Range[] = [];
    for (let k = 0; k < n; k++) {
      const start = next();
      out.push([start, next()]);
    }
    return all ? "all" : out;
  };
  const ids = (): number[] => {
    const n = next();
    const out: number[] = [];
    for (let k = 0; k < n; k++) {
      out.push(next());
    }
    return out;
  };

  const count = next();
  const out: RawChange[] = [];
  for (let r = 0; r < count; r++) {
    const index = next();
    const generation = next();
    const flags = next();
    out.push({
      index,
      generation,
      added: (flags & ADDED) !== 0,
      removed: (flags & REMOVED) !== 0,
      placementChanged: (flags & PLACEMENT) !== 0,
      capacityChanged: (flags & CAPACITY) !== 0,
      positions: ranges((flags & POSITIONS_ALL) !== 0),
      normals: ranges((flags & NORMALS_ALL) !== 0),
      colors: ranges((flags & COLORS_ALL) !== 0),
      indices: ranges((flags & INDICES_ALL) !== 0),
      fields: ids(),
      sections: ids(),
    });
  }
  return out;
}
