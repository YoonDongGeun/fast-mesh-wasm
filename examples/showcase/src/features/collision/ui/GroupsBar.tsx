// Named mesh groups: group the meshes ticked in the list, then colour each
// group by its contacts (`store.setContactGroup`, one id per group), show or
// hide all its members, remove the group (its meshes stay) or remove its
// meshes (each its own undo entry). A colouring needs two live members.
import type { CollisionMode } from "fast-mesh-wasm";
import { useState } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { liveMembers } from "../ux/groups";
import { useNamedGroups } from "../ux/use-named-groups";

export function GroupsBar(props: { app: AppState }) {
  const { app } = props;
  const { items, groups, collision, groupRules, setGroupCollisionMode, setVisible, ungroup, removeGroupMeshes } =
    useNamedGroups(app);
  const checks = useApp(app, (s) => s.checks);
  const [name, setName] = useState("");

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
            const live = liveMembers(g.members, items);
            const overrideMode = groupRules[g.id];
            const effectiveMode = overrideMode ?? collision.mode;
            return (
              <li key={g.id} className="named-group" data-group={g.name}>
                <div className="group-head">
                  <span className="group-name" title={`${live.length} meshes`}>
                    {g.name} ({live.length})
                  </span>
                  <span
                    className="tree-group-badge"
                    data-group-collision-badge={g.name}
                    title={`Active collision response: ${effectiveMode}`}
                  >
                    {effectiveMode}
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
                  <label
                    className="field"
                    title={live.length < 2 ? "Needs two meshes" : "Inter-object collision response for pairs in this group"}
                  >
                    <span>Rule</span>
                    <select
                      data-group-collision={g.name}
                      value={overrideMode ?? "default"}
                      disabled={live.length < 2}
                      onChange={(e) =>
                        setGroupCollisionMode(g.id, live, e.currentTarget.value as CollisionMode | "default")
                      }
                    >
                      <option value="default">Default ({collision.mode})</option>
                      <option value="passThrough">passThrough</option>
                      <option value="rigid">rigid</option>
                      <option value="cut">cut</option>
                      <option value="softDeform">softDeform</option>
                    </select>
                  </label>
                  <button type="button" onClick={() => setVisible(live, true)} title="Show every member">
                    Show
                  </button>
                  <button type="button" onClick={() => setVisible(live, false)} title="Hide every member">
                    Hide
                  </button>
                  <button
                    type="button"
                    onClick={() => ungroup(g.id, live)}
                    title="Remove the group (meshes stay)"
                  >
                    Ungroup
                  </button>
                  <button
                    type="button"
                    className="danger"
                    onClick={() => removeGroupMeshes(g.id, live)}
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
