// Polyline edits in the scene's undo history (#268, #282). The history is the
// engine's and knows meshes, not polylines, so each edit is an application
// entry in it: a step of its own or part of the open one, undone and redone in
// order with every other edit. An entry holds what its edit changed, by id,
// and makes that change to the list as it finds it. Display settings (name,
// colour, visibility) are not undone: a polyline that stays keeps its own.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { Polyline } from "./polyline";

/** The scene call the history makes. */
export type HistoryStore = Pick<SceneStore, "recordEntry">;

type List = readonly Polyline[];

/** Bytes an entry counts for each polyline in its lists, beside the points it changes. */
const REF_BYTES = 8;

/** `list` with what differs between `from` and `to` as `to` has it: polylines that went, came or changed shape. */
export function applyEdit(list: List, from: List, to: List): Polyline[] {
  const before = new Map(from.map((p) => [p.id, p]));
  const after = new Map(to.map((p) => [p.id, p]));
  const kept = list.flatMap((p): Polyline[] => {
    const [a, b] = [before.get(p.id), after.get(p.id)];
    if (a === undefined) {
      return [p];
    }
    if (b === undefined) {
      return [];
    }
    return [a.points === b.points && a.closed === b.closed ? p : { ...p, closed: b.closed, points: b.points }];
  });
  // Each that came goes after the one it follows in `to`, or first.
  const listed = new Set(kept.map((p) => p.id));
  const came = new Map<number | null, Polyline[]>([[null, []]]);
  let follows: number | null = null;
  for (const p of to) {
    if (listed.has(p.id)) {
      follows = p.id;
    } else if (!before.has(p.id)) {
      came.set(follows, [...(came.get(follows) ?? []), p]);
    }
  }
  return [...(came.get(null) ?? []), ...kept.flatMap((p) => [p, ...(came.get(p.id) ?? [])])];
}

/** About the bytes an edit's entry keeps alive: the points only one of its lists has, and the lists. */
export function editBytes(before: List, after: List): number {
  const only = (a: List, b: List) => {
    const shared = new Set(b.map((p) => p.points));
    return a.reduce((n, p) => n + (shared.has(p.points) ? 0 : p.points.byteLength), 0);
  };
  return only(before, after) + only(after, before) + REF_BYTES * (before.length + after.length);
}

export class PolylineHistory {
  /** Goes up when the list is replaced for good: entries recorded before do nothing. */
  #epoch = 0;

  /**
   * Makes the edit from `before` to `after` an undo step, or part of the open
   * one. Undo and redo call `change` with what to make of the list.
   */
  record(store: HistoryStore, before: List, after: List, change: (edit: (list: List) => Polyline[]) => void): void {
    const epoch = this.#epoch;
    // The lists around the edit, as it was last made. They change when it is
    // made again on another list: on a Worker, an undo can still be on its way
    // when the edit is recorded, and then comes before it.
    let [from, to] = [before, after];
    const run = (edit: (list: List) => Polyline[]) => () => {
      if (epoch === this.#epoch) {
        change(edit);
      }
    };
    store.recordEntry({
      undo: run((list) => applyEdit(list, to, from)),
      redo: run((list) => {
        const made = applyEdit(list, before, after);
        [from, to] = [list, made];
        return made;
      }),
      bytes: editBytes(before, after),
    });
  }

  /** The list is replaced for good, as by opening a project: no entry before this changes it any more. */
  reset(): void {
    this.#epoch++;
  }
}
