// The collision panel's Group mode: tick meshes into a contact group
// (`store.setContactGroup`), measured to each other in every pair or in the
// pairs ticked in a small matrix. Each member is coloured by its distance to
// the nearest of its partners, with the same heatmap as the pair; readouts
// are given per pair (gap, intersecting, deepest penetration).
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useStoreSelector } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { type AppState, type GroupSetup, type Item, useApp } from "../../../app/state/app-state";
import type { GroupReadouts } from "../../../app/state/readouts";
import { meshesKey } from "../ux/live";
import { useGroupContact } from "../ux/use-group-contact";
import type { Heatmap } from "../ux/use-heatmap-settings";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";

export function GroupMode(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  group: GroupSetup;
  heat: Heatmap;
}) {
  const { app, scene, items, group, heat } = props;
  const units = useApp(app, (s) => s.units);
  const { members, pairs, ready, bound, readouts, warnings, nameOf, setAllPairs, togglePair } = useGroupContact(
    app,
    scene,
    items,
    group,
    heat,
  );

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
          onChange={(e) => setAllPairs(e.currentTarget.checked)}
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
                        onChange={() => togglePair(a, b)}
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
          <PairReadouts units={units} readouts={readouts} pairs={pairs} nameOf={short} bound={bound} />
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
  units: LengthUnit;
}) {
  const { readouts: r, pairs, nameOf, bound, units } = props;
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
                <td>{p === undefined || p.gap === null ? dash : formatLength(p.gap, units)}</td>
                <td>{p === undefined ? dash : p.intersects ? "yes" : "no"}</td>
                <td>
                  {p === undefined || p.depth === null
                    ? dash
                    : p.atLeast
                      ? `≥ ${formatLength(Math.max(p.depth, within), units)}`
                      : formatLength(p.depth, units)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {Number.isFinite(within) ? (
        <p className="muted small">
          Fast mode: distances are measured within {formatLength(within, units)}; deeper penetration shows the mesh colour.
        </p>
      ) : null}
    </>
  );
}
