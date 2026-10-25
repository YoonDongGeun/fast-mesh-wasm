// Named mesh groups: group the meshes ticked in the list, then colour each
// group by its contacts (`store.setContactGroup`, one id per group), show or
// hide all its members, remove the group (its meshes stay) or remove its
// meshes (each its own undo entry). A colouring needs two live members.
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { contactHeatmap } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { clampTolerance, niceScale } from "./collision-math";
import type { Heatmap } from "./CollisionPanel";

/** The store's contact-group id of a named group. */
function contactId(id: number): string {
  return `showcase-named-${id}`;
}

/** The heatmap of a named group: the collision defaults at its members' scale. */
export function groupHeat(items: readonly Item[], members: readonly MeshHandle[]): Heatmap {
  const scale = niceScale(Math.max(0, ...items.filter((i) => members.includes(i.mesh)).map((i) => i.radius)));
  const step = scale / 1000;
  const depth = scale / 2;
  const gap = scale / 2;
  return contactHeatmap({ depth, gap, tolerance: clampTolerance(scale / 50, depth, gap, step), exactDepth: true });
}

export function GroupsBar(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const items = useApp(app, (s) => s.items);
  const checks = useApp(app, (s) => s.checks);
  const groups = useApp(app, (s) => s.meshGroups);
  const [name, setName] = useState("");
  // The contact-group ids currently applied: released when no longer wanted.
  const applied = useRef(new Set<number>());
  useEffect(() => {
    const want = new Map<number, MeshHandle[]>();
    for (const g of groups) {
      const live = g.members.filter((m) => items.some((i) => i.mesh === m));
      if (g.colored && live.length >= 2) {
        want.set(g.id, live);
      }
    }
    for (const [id, members] of want) {
      store.setContactGroup(contactId(id), { meshes: members, pairs: "all", coloring: groupHeat(items, members) });
    }
    for (const id of applied.current) {
      if (!want.has(id)) {
        store.setContactGroup(contactId(id), null);
      }
    }
    applied.current = new Set(want.keys());
  }, [store, groups, items]);
  useEffect(() => {
    const at = applied.current;
    return () => {
      for (const id of at) {
        store.setContactGroup(contactId(id), null);
      }
    };
  }, [store]);

  const setVisible = (members: readonly MeshHandle[], visible: boolean) => {
    for (const m of members) {
      store.setDisplay(m, { visible });
    }
  };
  const removeMeshes = (members: readonly MeshHandle[]) => {
    for (const item of items.filter((i) => members.includes(i.mesh))) {
      app.removeMesh(store, item);
    }
  };

  return (
    <section className="panel" aria-label="Groups">
      <header className="panel-head">
        <h2>Groups</h2>
      </header>
      <div className="group-create">
        <input
          type="text"
          value={name}
          placeholder={`Group ${groups.length + 1}`}
          aria-label="Group name"
          onChange={(e) => setName(e.currentTarget.value)}
        />
        <button
          type="button"
          data-action="group-checked"
          disabled={checks.length === 0}
          title={checks.length === 0 ? "Tick meshes in the list first" : `Group ${checks.length} ticked meshes`}
          onClick={() => {
            app.createGroup(name);
            setName("");
          }}
        >
          Group checked ({checks.length})
        </button>
      </div>
      {groups.length > 0 ? (
        <ul className="groups" aria-label="Named groups">
          {groups.map((g) => {
            const live = g.members.filter((m) => items.some((i) => i.mesh === m));
            return (
              <li key={g.id} className="named-group" data-group={g.name}>
                <div className="group-head">
                  <span className="group-name" title={`${live.length} meshes`}>
                    {g.name} ({live.length})
                  </span>
                  <label className="check" title={live.length < 2 ? "Needs two meshes" : "Colour by contacts"}>
                    <input
                      type="checkbox"
                      data-group-color={g.name}
                      checked={g.colored}
                      disabled={live.length < 2}
                      onChange={(e) => app.setGroupColored(g.id, e.currentTarget.checked)}
                    />
                    Color
                  </label>
                </div>
                <div className="group-controls">
                  <button type="button" onClick={() => setVisible(live, true)} title="Show every member">
                    Show
                  </button>
                  <button type="button" onClick={() => setVisible(live, false)} title="Hide every member">
                    Hide
                  </button>
                  <button type="button" onClick={() => app.removeGroup(g.id)} title="Remove the group (meshes stay)">
                    Ungroup
                  </button>
                  <button
                    type="button"
                    className="danger"
                    onClick={() => removeMeshes(live)}
                    title="Remove every member (each its own undo entry)"
                  >
                    Remove meshes
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted small">Tick meshes in the list, then group them.</p>
      )}
    </section>
  );
}
