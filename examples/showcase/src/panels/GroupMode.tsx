// The collision panel's Group mode: tick meshes into a contact group
// (`store.setContactGroup`), measured to each other in every pair or in the
// pairs ticked in a small matrix. Each member is coloured by its distance to
// the nearest of its partners, with the same heatmap as the pair; readouts
// are given per pair (gap, intersecting, deepest penetration).
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useStoreSelector } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo } from "react";
import type { AppState, GroupSetup, Item } from "../scene/app-state";
import type { Heatmap } from "./CollisionPanel";
import { formatDistance, mayBeDeeper } from "./collision-math";
import { groupPairs, type MemberValues, pairDepths, togglePair } from "./group-math";
import { type GroupReadouts, inspect } from "./inspect";
import { meshesKey, useLiveReadouts, useTopologyWarnings } from "./live";

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

export function GroupMode(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  group: GroupSetup;
  heat: Heatmap;
}) {
  const { app, scene, items, group, heat } = props;
  const store = useScene();
  // Members still listed (a removal updates the group in the same tick).
  const members = useMemo(
    () => group.members.filter((m) => items.some((i) => i.mesh === m)),
    [group.members, items],
  );
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

  const all = group.pairs === "all";
  const short = (m: MeshHandle) => nameOf(m).split(" ")[0];
  return (
    <div className="group-mode">
      <ul className="members" aria-label="Group members">
        {items.map((i) => (
          <li key={i.id}>
            <label className="check">
              <input
                type="checkbox"
                data-group-member={i.name}
                checked={members.includes(i.mesh)}
                onChange={(e) => app.setGroupMember(i.mesh, e.currentTarget.checked)}
              />
              <span className="swatch" style={{ background: `#${i.color.toString(16).padStart(6, "0")}` }} />
              {i.name}
            </label>
          </li>
        ))}
      </ul>
      <label className="check" title="Measure every member to every other (root boxes farther apart than the bound are skipped)">
        <input
          type="checkbox"
          data-control="all-pairs"
          checked={all}
          onChange={(e) => app.setGroupPairs(e.currentTarget.checked ? "all" : groupPairs(members, "all"))}
        />
        All pairs
      </label>
      {!all && members.length >= 2 ? (
        <table className="pair-matrix" aria-label="Pairs">
          <thead>
            <tr>
              <th />
              {members.slice(1).map((m) => (
                <th key={meshesKey(items, [m])} title={nameOf(m)}>
                  {short(m)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {members.slice(0, -1).map((a, i) => (
              <tr key={meshesKey(items, [a])}>
                <th title={nameOf(a)}>{short(a)}</th>
                {members.slice(1).map((b, j) => (
                  <td key={meshesKey(items, [b])}>
                    {j >= i ? (
                      <input
                        type="checkbox"
                        data-pair={`${nameOf(a)}|${nameOf(b)}`}
                        title={`${nameOf(a)} ↔ ${nameOf(b)}`}
                        checked={pairs.some(([x, y]) => x === a && y === b)}
                        onChange={() => app.setGroupPairs(togglePair(group.pairs === "all" ? [] : group.pairs, a, b))}
                      />
                    ) : null}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {!ready ? (
        <p className="muted small">Tick two or more meshes to colour them by contact.</p>
      ) : (
        <>
          <GroupStatus members={members} />
          <PairReadouts readouts={readouts} pairs={pairs} nameOf={short} bound={bound} />
          {warnings.map((w) => (
            <p key={w} className="warn small" data-warning>
              {w}
            </p>
          ))}
          <p className="muted small">Hover a coloured mesh to see which partner it is nearest to.</p>
        </>
      )}
    </div>
  );
}

/** "Colouring…" while any member's colouring is being computed. */
function GroupStatus(props: { members: readonly MeshHandle[] }) {
  const { members } = props;
  const busy = useStoreSelector((s) => members.some((m) => s.meshes.get(m)?.status === "computing"));
  return busy ? (
    <p className="busy small" data-collision="computing">
      Colouring… <span className="progress" aria-hidden="true" />
    </p>
  ) : null;
}

function PairReadouts(props: {
  readouts: GroupReadouts | null;
  pairs: readonly (readonly [MeshHandle, MeshHandle])[];
  nameOf: (m: MeshHandle) => string;
  bound: number;
}) {
  const { readouts: r, pairs, nameOf, bound } = props;
  const dash = "—";
  if (pairs.length === 0) {
    return <p className="muted small">No pairs: tick some in the matrix.</p>;
  }
  const within = r === null ? bound : r.bound;
  return (
    <>
      <table className="pair-readouts" data-readouts={r === null ? "none" : "ready"}>
        <thead>
          <tr>
            <th>Pair</th>
            <th title="meshDistance: 0 when the surfaces touch">Min gap</th>
            <th title="intersects">Hit</th>
            <th title="The deepest penetration at vertices nearest to the other mesh">Max depth</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map(([a, b]) => {
            const p = r?.pairs.find((x) => x.a === a && x.b === b);
            return (
              <tr key={`${nameOf(a)}|${nameOf(b)}`}>
                <td>
                  {nameOf(a)} ↔ {nameOf(b)}
                </td>
                <td>{p === undefined || p.gap === null ? dash : formatDistance(p.gap)}</td>
                <td>{p === undefined ? dash : p.intersects ? "yes" : "no"}</td>
                <td>
                  {p === undefined || p.depth === null
                    ? dash
                    : p.atLeast
                      ? `≥ ${formatDistance(Math.max(p.depth, within))}`
                      : formatDistance(p.depth)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {Number.isFinite(within) ? (
        <p className="muted small">
          Fast mode: distances are measured within {formatDistance(within)}; deeper penetration shows the mesh colour.
        </p>
      ) : null}
    </>
  );
}
