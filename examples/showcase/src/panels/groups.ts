// Named mesh groups: pure transitions over member lists, generic over the
// member identity (mesh handles in the app, strings in tests).
export interface NamedGroup<M> {
  id: number;
  name: string;
  members: readonly M[];
  /** Whether its contact colouring is applied (needs two live members). */
  colored: boolean;
}

/** Ticks (`on`) or unticks `m` in `checks`, kept in `order`. */
export function toggleCheck<M>(checks: readonly M[], order: readonly M[], m: M, on: boolean): readonly M[] {
  if (on) {
    return order.filter((o) => o === m || checks.includes(o));
  }
  return checks.filter((c) => c !== m);
}

/** Appends a group; a blank `name` falls back to `fallback`. Members deduped, order kept. */
export function createGroup<M>(
  groups: readonly NamedGroup<M>[],
  id: number,
  name: string,
  fallback: string,
  members: readonly M[],
): NamedGroup<M>[] {
  const kept: M[] = [];
  for (const m of members) {
    if (!kept.includes(m)) {
      kept.push(m);
    }
  }
  const trimmed = name.trim();
  return [...groups, { id, name: trimmed === "" ? fallback : trimmed, members: kept, colored: false }];
}

/** Removes the group `id` (a no-op when it is not there). */
export function removeGroup<M>(groups: readonly NamedGroup<M>[], id: number): NamedGroup<M>[] {
  return groups.filter((g) => g.id !== id);
}

/** Colours (`on`) or uncolours the group `id`; other groups untouched. */
export function setColored<M>(groups: readonly NamedGroup<M>[], id: number, on: boolean): NamedGroup<M>[] {
  return groups.map((g) => (g.id === id ? { ...g, colored: on } : g));
}

/** Drops `m` from every group (a removal); groups stay, even emptied. */
export function dropMember<M>(groups: readonly NamedGroup<M>[], m: M): NamedGroup<M>[] {
  return groups.map((g) => (g.members.includes(m) ? { ...g, members: g.members.filter((x) => x !== m) } : g));
}
