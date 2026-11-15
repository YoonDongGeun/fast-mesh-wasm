// Named mesh groups: group the meshes ticked in the list, then colour each
// group by its contacts (`store.setContactGroup`, one id per group), show or
// hide all its members, remove the group (its meshes stay) or remove its
// meshes (each its own undo entry). A colouring needs two live members.
import type { CollisionMode, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { contactHeatmap } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { buildCollisionRule, clampTolerance, collisionRuleSignature, niceScale } from "./collision-math";
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
  const collision = useApp(app, (s) => s.wave6.collision);
  const [groupRules, setGroupRules] = useState<Record<number, CollisionMode>>({});
  const lastGroupSigRef = useRef(new Map<number, string>());
  const lastSceneRef = useRef<object | null>(store.scene);

  if (lastSceneRef.current !== store.scene) {
    lastSceneRef.current = store.scene;
    lastGroupSigRef.current.clear();
  }

  function syncGroupPairRules(groupId: number, members: readonly MeshHandle[], mode: CollisionMode | "default") {
    if (mode === "default") {
      lastGroupSigRef.current.delete(groupId);
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          void store.scene.clearCollisionRule?.(members[i], members[j]);
        }
      }
      return;
    }
    const rule = buildCollisionRule(collision, { mode });
    const sig = `${members.join(",")}|${collisionRuleSignature(rule)}`;
    if (lastGroupSigRef.current.get(groupId) === sig) {
      return;
    }
    lastGroupSigRef.current.set(groupId, sig);
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        store.scene.setCollisionRule?.(members[i], members[j], rule);
      }
    }
  }

  function setGroupCollisionMode(groupId: number, members: readonly MeshHandle[], mode: CollisionMode | "default") {
    syncGroupPairRules(groupId, members, mode);
    setGroupRules((prev) => {
      if (mode === "default") {
        if (!(groupId in prev)) {
          return prev;
        }
        const next = { ...prev };
        delete next[groupId];
        return next;
      }
      return { ...prev, [groupId]: mode };
    });
  }

  useEffect(() => {
    const activeIds = new Set(groups.map((g) => g.id));
    for (const key of lastGroupSigRef.current.keys()) {
      if (!activeIds.has(key)) {
        lastGroupSigRef.current.delete(key);
      }
    }
    setGroupRules((prev) => {
      let changed = false;
      const next: Record<number, CollisionMode> = {};
      for (const [rawId, mode] of Object.entries(prev)) {
        const id = Number(rawId);
        if (activeIds.has(id)) {
          next[id] = mode;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [groups]);

  useEffect(() => {
    for (const g of groups) {
      const mode = groupRules[g.id];
      if (!mode) {
        continue;
      }
      const live = g.members.filter((m) => items.some((i) => i.mesh === m));
      if (live.length >= 2) {
        syncGroupPairRules(g.id, live, mode);
      }
    }
  }, [collision, groups, groupRules, items, store]);
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
                    onClick={() => {
                      setGroupCollisionMode(g.id, live, "default");
                      app.removeGroup(g.id);
                    }}
                    title="Remove the group (meshes stay)"
                  >
                    Ungroup
                  </button>
                  <button
                    type="button"
                    className="danger"
                    onClick={() => {
                      setGroupCollisionMode(g.id, live, "default");
                      removeMeshes(live);
                    }}
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
