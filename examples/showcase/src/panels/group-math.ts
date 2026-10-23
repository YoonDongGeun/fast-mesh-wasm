// The collision panel's Group mode arithmetic: the pairs a contact group
// measures, editing the pair list, each pair's deepest penetration from the
// members' field values and nearest partners, and the partner behind one
// vertex (the hover tip).

/** `Scene.fieldNearest`'s "no target" index (an infinite or NaN value). */
const NONE = 0xffffffff;

/**
 * The pairs of `members` that are measured to each other, in member order
 * (the earlier member first), each once: every pair with `"all"`, else the
 * listed pairs, either way round, whose meshes are two different members.
 */
export function groupPairs<T>(members: readonly T[], pairs: readonly (readonly [T, T])[] | "all"): [T, T][] {
  const out: [T, T][] = [];
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) {
      const [a, b] = [members[i], members[j]];
      if (pairs === "all" || pairs.some(([x, y]) => (x === a && y === b) || (x === b && y === a))) {
        out.push([a, b]);
      }
    }
  }
  return out;
}

/** `pairs` with the pair of `a` and `b` (either way round) removed when listed, else added: a new list. */
export function togglePair<T>(pairs: readonly (readonly [T, T])[], a: T, b: T): [T, T][] {
  const listed = (p: readonly [T, T]) => (p[0] === a && p[1] === b) || (p[0] === b && p[1] === a);
  const copy = pairs.map(([x, y]): [T, T] => [x, y]);
  return pairs.some(listed) ? copy.filter((p) => !listed(p)) : [...copy, [a, b]];
}

/** A group member's colouring as read back: its field values and nearest partners (`null` when unread). */
export interface MemberValues<T> {
  mesh: T;
  values: Float32Array | null;
  nearest: Uint32Array | null;
  /** The field's targets in index order: what `nearest` indexes. */
  partners: readonly T[];
}

/**
 * Each pair's deepest penetration: `-min` over the negative values of
 * either member at vertices whose nearest partner is the other member (0
 * when there is none), or `null` when neither member's values were read.
 *
 * A member's field keeps the smallest value over all of its partners, so a
 * vertex inside two partners counts for the deeper one only: a pair's depth
 * can read shallower where three meshes overlap. A nearest index past the
 * partner list (a target removed since the read) counts for no pair.
 */
export function pairDepths<T>(entries: readonly MemberValues<T>[], pairs: readonly (readonly [T, T])[]): (number | null)[] {
  return pairs.map(([a, b]) => {
    let any = false;
    let min = 0;
    for (const [self, other] of [
      [a, b],
      [b, a],
    ]) {
      const e = entries.find((x) => x.mesh === self);
      if (e === undefined || e.values === null || e.nearest === null) {
        continue;
      }
      any = true;
      const k = e.partners.indexOf(other);
      if (k < 0) {
        continue;
      }
      const { values, nearest } = e;
      const n = Math.min(values.length, nearest.length);
      for (let v = 0; v < n; v++) {
        if (nearest[v] === k && values[v] < min) {
          min = values[v];
        }
      }
    }
    return any ? (min < 0 ? -min : 0) : null;
  });
}

/**
 * The partner whose surface gave vertex `slot` its value, with the value
 * (negative inside it), or `null` when the vertex has none (beyond the
 * field's bound, NaN, or an index past the partner list).
 */
export function nearestPartner<T>(
  values: Float32Array,
  nearest: Uint32Array,
  partners: readonly T[],
  slot: number,
): { partner: T; distance: number } | null {
  const k = nearest[slot];
  const distance = values[slot];
  if (k === undefined || k === NONE || k >= partners.length || !Number.isFinite(distance)) {
    return null;
  }
  return { partner: partners[k], distance };
}
