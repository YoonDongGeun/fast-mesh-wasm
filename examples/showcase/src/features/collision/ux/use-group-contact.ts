// The collision panel's Group mode: the ticked meshes as a contact group
// (`store.setContactGroup`), measured to each other in every pair or in the
// pairs ticked, each member coloured by its distance to the nearest of its
// partners; their per-pair readouts (gap, intersecting, deepest penetration)
// and topology warnings kept live, and the members' colourings read back.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo } from "react";
import type { AppState, GroupSetup, Item } from "../../../app/state/app-state";
import { type GroupReadouts, inspect } from "../../../app/state/readouts";
import { mayBeDeeper } from "./collision-math";
import { groupPairs, type MemberValues, pairDepths, togglePair } from "./group-math";
import { liveMembers } from "./groups";
import { meshesKey, useLiveReadouts, useTopologyWarnings } from "./live";
import type { Heatmap } from "./use-heatmap-settings";

/** The store's id of the panel's contact group. */
const GROUP_ID = "showcase";

/**
 * The targets of the field behind a mesh's colouring, in the order its
 * nearest-target indices refer to (`store.coloringTargets`: ascending scene
 * id, not the member order); empty without a colouring.
 */
export function partnersOf(store: SceneStore, mesh: MeshHandle): readonly MeshHandle[] {
  return store.coloringTargets(mesh) ?? [];
}

/**
 * A member's colouring read back: the values of its store-owned field
 * (`fetchColoringValues`) and the field's nearest partner per vertex
 * (`fetchFieldNearest` on a Worker scene, `fieldNearest` on a local one),
 * either `null` when it has no colouring or it changed meanwhile.
 */
export async function readMember(
  store: SceneStore,
  scene: Scene | WorkerScene,
  mesh: MeshHandle,
): Promise<MemberValues<MeshHandle>> {
  const partners = partnersOf(store, mesh);
  const field = scene.coloring(mesh)?.field;
  const [values, nearest] = await Promise.all([
    store.fetchColoringValues(mesh).catch(() => null),
    field === undefined
      ? null
      : Promise.resolve()
          .then(() => ("fetchFieldNearest" in scene ? scene.fetchFieldNearest(field) : scene.fieldNearest(field)))
          .catch(() => null),
  ]);
  return { mesh, values, nearest, partners };
}

/**
 * The contact group of `group`'s members still listed, coloured with `heat`
 * while it has two or more (`ready`), with its pairs, readouts and warnings.
 * `setAllPairs` measures every pair (or lists them all, to untick some), and
 * `togglePair` ticks or unticks one.
 */
export function useGroupContact(
  app: AppState,
  scene: Scene | WorkerScene,
  items: readonly Item[],
  group: GroupSetup,
  heat: Heatmap,
): {
  members: MeshHandle[];
  pairs: [MeshHandle, MeshHandle][];
  ready: boolean;
  bound: number;
  readouts: GroupReadouts | null;
  warnings: string[];
  nameOf: (m: MeshHandle) => string;
  setAllPairs: (on: boolean) => void;
  togglePair: (a: MeshHandle, b: MeshHandle) => void;
} {
  const store = useScene();
  // Members still listed (a removal updates the group in the same tick).
  const members = useMemo(() => liveMembers(group.members, items), [group.members, items]);
  const pairs = useMemo(() => groupPairs(members, group.pairs), [members, group.pairs]);
  const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? "the mesh";
  const key = meshesKey(items, members);
  const pairsKey = pairs.map(([a, b]) => meshesKey(items, [a, b])).join(";");

  // The group, set again when its members, pairs or heatmap change: a new
  // heatmap only recolours (the store keeps each member's field when its
  // bound and partners stay). Fewer than two members colour nothing.
  useEffect(() => {
    store.setContactGroup(
      GROUP_ID,
      members.length < 2 ? null : { meshes: members, pairs: group.pairs === "all" ? "all" : pairs, coloring: heat },
    );
    // `key` and `pairsKey` stand for `members` and `pairs` (new arrays on every list change).
  }, [store, key, pairsKey, heat]);
  // Leaving Group mode uncolours the members and drops their fields.
  useEffect(() => () => store.setContactGroup(GROUP_ID, null), [store]);

  const bound = heat.maxDistance;
  const ready = members.length >= 2;
  const { readouts, poke } = useLiveReadouts<GroupReadouts>(
    store,
    ready ? members : null,
    `${key}/${pairsKey}`,
    async () => {
      const entries = await Promise.all(members.map((m) => readMember(store, scene, m)));
      const each = await Promise.all(
        pairs.map(([a, b]) => Promise.all([Promise.resolve(scene.meshDistance(a, b)), Promise.resolve(scene.intersects(a, b))])),
      );
      const depths = pairDepths(entries, pairs);
      return {
        bound,
        pairs: pairs.map(([a, b], k) => ({
          a,
          b,
          gap: each[k][0] === null ? null : each[k][0].distance,
          intersects: each[k][1],
          depth: depths[k],
          atLeast: depths[k] !== null && mayBeDeeper(depths[k], bound, each[k][1]),
        })),
      };
    },
    inspect.group,
  );
  // A new bound changes how the depths read (the colours follow through change records).
  useEffect(() => poke(), [bound]);
  const warnings = useTopologyWarnings(scene, members, key, nameOf);
  useEffect(() => {
    inspect.group.warnings = warnings;
  });

  return {
    members,
    pairs,
    ready,
    bound,
    readouts,
    warnings,
    nameOf,
    setAllPairs: (on) => app.setGroupPairs(on ? "all" : groupPairs(members, "all")),
    togglePair: (a, b) => app.setGroupPairs(togglePair(group.pairs === "all" ? [] : group.pairs, a, b)),
  };
}
