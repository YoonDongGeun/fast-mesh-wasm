// The collision panel: the contact heatmap of meshes coloured by their
// signed distance to each other (`contactHeatmap`). The heatmap controls are
// shared by two modes: a pair, each mesh coloured against the other, with
// readouts of the deepest penetration, the gap, whether they intersect and
// the triangles in contact; and a contact group (`store.setContactGroup`),
// several meshes each coloured by its nearest partner, with readouts per pair.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useMeshDisplay, useScene } from "fast-mesh-wasm/react";
import { contactHeatmap, type DistanceColoring } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo, useState } from "react";
import { type AppState, type CollisionPair, type Item, useApp } from "../scene/app-state";
import { clampTolerance, deepestPenetration, formatDistance, niceScale, toleranceLimit } from "./collision-math";
import { GroupMode } from "./GroupMode";
import { inspect, type Readouts } from "./inspect";
import { meshesKey, useLiveReadouts, useTopologyWarnings } from "./live";

export type Heatmap = Omit<DistanceColoring, "distanceTo">;

interface HeatmapParams {
  depth: number;
  gap: number;
  tolerance: number;
}

export function CollisionPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const items = useApp(app, (s) => s.items);
  const pair = useApp(app, (s) => s.collision);
  const mode = useApp(app, (s) => s.collisionMode);
  const group = useApp(app, (s) => s.group);
  const shown = mode === "pair" ? [pair.a, pair.b] : group.members;

  // The sliders' scale follows the size of the meshes coloured: a file in
  // millimetres and the unit-sized demo meshes both get useful ranges.
  const scale = niceScale(Math.max(0, ...items.filter((i) => shown.includes(i.mesh)).map((i) => i.radius)));
  const step = scale / 1000;
  // `null` until a slider moves: the defaults then follow the scale.
  const [params, setParams] = useState<HeatmapParams | null>(null);
  const [fast, setFast] = useState(false);
  const softness = useApp(app, (s) => s.edgeSoftness);
  const p = params ?? { depth: scale / 2, gap: scale / 2, tolerance: scale / 50 };
  // Clamped to contactHeatmap's limits (spec §6.5, gap ≥ 0 from §7.1).
  const depth = Math.min(Math.max(p.depth, step), scale);
  const gap = Math.min(Math.max(p.gap, 0), scale);
  const tolerance = clampTolerance(p.tolerance, depth, gap, step);
  const tolMax = Math.max(0, toleranceLimit(depth, gap) - step);
  const set = (patch: Partial<HeatmapParams>) => setParams({ depth, gap, tolerance, ...patch });

  const heat = useMemo(
    () => contactHeatmap({ depth, gap, tolerance, exactDepth: !fast }),
    [depth, gap, tolerance, fast],
  );

  return (
    <section className="panel" aria-label="Collision">
      <h2>Collision</h2>
      <div className="modes" role="group" aria-label="Collision mode">
        <button type="button" data-mode="pair" aria-pressed={mode === "pair"} onClick={() => app.setCollisionMode("pair")}>
          Pair
        </button>
        <button
          type="button"
          data-mode="group"
          aria-pressed={mode === "group"}
          onClick={() => app.setCollisionMode("group")}
        >
          Group
        </button>
      </div>
      <p className="muted small">
        {mode === "pair"
          ? "Each mesh coloured by its distance to the other"
          : "Each mesh coloured by its distance to the nearest of its partners"}
        : penetration in reds, contact white, the gap green to blue.
      </p>
      <Slider
        label="Penetration depth"
        name="depth"
        min={step}
        max={scale}
        step={step}
        value={depth}
        onChange={(v) => set({ depth: v })}
        title="The penetration shown darkest; deeper clamps to it"
      />
      <Slider
        label="Colour from (mm)"
        name="gap"
        min={0}
        max={scale}
        step={step}
        value={gap}
        onChange={(v) => set({ gap: v })}
        title="The gap shown blue; farther shows the mesh colour. 0 colours penetration only"
      />
      <Slider
        label="Contact tolerance"
        name="tolerance"
        min={0}
        max={tolMax}
        step={step}
        value={tolerance}
        onChange={(v) => set({ tolerance: v })}
        title="Within ± this distance counts as contact (white), and for the contact count"
      />
      <Slider
        label="Boundary softness"
        name="softness"
        min={0}
        max={scale / 10}
        step={step}
        value={Math.min(softness, scale / 10)}
        onChange={(v) => app.setEdgeSoftness(v)}
        title="The width of the blend where colours meet: the coloured area's edge and the contact band's. 0 is a hard edge"
      />
      <Legend depth={depth} gap={gap} tolerance={tolerance} />
      <label
        className="check"
        title="Bound the fields at twice max(depth, colour from): much faster on large meshes, but deeper penetration shows the mesh colour"
      >
        <input type="checkbox" data-control="fast" checked={fast} onChange={(e) => setFast(e.currentTarget.checked)} />
        Fast mode (large meshes)
      </label>
      {mode === "pair" ? (
        <PairMode app={app} scene={scene} items={items} pair={pair} heat={heat} tolerance={tolerance} />
      ) : (
        <GroupMode app={app} scene={scene} items={items} group={group} heat={heat} />
      )}
    </section>
  );
}

/** Two picked meshes, each coloured against the other, and their readouts. */
function PairMode(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  pair: CollisionPair;
  heat: Heatmap;
  tolerance: number;
}) {
  const { app, scene, items, pair, heat, tolerance } = props;
  const store = useScene();
  const { a, b } = pair;
  const ready = a !== null && b !== null && a !== b;

  // Colour both, each against the other. Picking another mesh (or removing
  // one: AppState.removeMesh resets its picker) stops the old colouring.
  useEffect(() => {
    if (a === null || b === null || a === b) {
      return;
    }
    store.setDisplay(a, { coloring: { distanceTo: b, ...heat } });
    store.setDisplay(b, { coloring: { distanceTo: a, ...heat } });
    return () => {
      for (const m of [a, b]) {
        if (store.getSnapshot().meshes.has(m)) {
          store.setDisplay(m, { coloring: null });
        }
      }
    };
  }, [store, a, b, heat]);

  const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? "the mesh";
  // B's warning first: the side A's colours are measured to.
  const watched = ready ? [b, a] : [];
  const key = meshesKey(items, watched);
  const bound = heat.maxDistance;
  const { readouts, poke } = useLiveReadouts<Readouts>(
    store,
    ready ? watched : null,
    key,
    async () => {
      if (!ready) {
        throw new Error("no pair");
      }
      const [va, vb, distance, hit, contacts] = await Promise.all([
        // After a commit (this runs from change records): the colouring last
        // sent. A colouring replaced meanwhile rejects; read it as none.
        store.fetchColoringValues(a).catch(() => null),
        store.fetchColoringValues(b).catch(() => null),
        Promise.resolve(scene.meshDistance(a, b)),
        Promise.resolve(scene.intersects(a, b)),
        Promise.resolve(scene.findContacts({ margin: tolerance, pairs: [[a, b]] })),
      ]);
      const c = contacts[0];
      return {
        penetration: deepestPenetration([va, vb], bound, hit),
        bound,
        gap: distance === null ? null : distance.distance,
        intersects: hit,
        contacts:
          c === undefined
            ? { a: 0, b: 0 }
            : c.a === a
              ? { a: c.trianglesA.length, b: c.trianglesB.length }
              : { a: c.trianglesB.length, b: c.trianglesA.length },
      };
    },
    inspect.collision,
  );
  // A new tolerance changes the contact count, a new bound how the depth
  // reads (the colours follow through change records).
  useEffect(() => poke(), [tolerance, bound]);
  const warnings = useTopologyWarnings(scene, watched, key, nameOf);
  useEffect(() => {
    inspect.collision.warnings = warnings;
  });

  return (
    <div className="pair">
      <Picker label="A" side="a" app={app} items={items} value={a} />
      <Picker label="B" side="b" app={app} items={items} value={b} />
      {!ready ? (
        <p className="muted small">
          {a !== null && a === b ? "Pick two different meshes." : "Pick two meshes to colour."}
        </p>
      ) : (
        <>
          <PairStatus a={a} b={b} />
          <ReadoutList readouts={readouts} bound={heat.maxDistance} />
          {warnings.map((w) => (
            <p key={w} className="warn small" data-warning>
              {w}
            </p>
          ))}
        </>
      )}
    </div>
  );
}

function Picker(props: {
  label: string;
  side: keyof CollisionPair;
  app: AppState;
  items: readonly Item[];
  value: MeshHandle | null;
}) {
  const { label, side, app, items, value } = props;
  const id = items.find((i) => i.mesh === value)?.id;
  return (
    <label className="picker">
      <span>{label}</span>
      <select
        data-pick={side}
        value={id === undefined ? "" : String(id)}
        onChange={(e) => {
          const v = e.currentTarget.value;
          app.pick(side, items.find((i) => String(i.id) === v)?.mesh ?? null);
        }}
      >
        <option value="">—</option>
        {items.map((i) => (
          <option key={i.id} value={String(i.id)}>
            {i.name}
          </option>
        ))}
      </select>
    </label>
  );
}

/** "Colouring…" while either mesh's colouring is being computed. */
function PairStatus(props: { a: MeshHandle; b: MeshHandle }) {
  const [sa] = useMeshDisplay(props.a);
  const [sb] = useMeshDisplay(props.b);
  if (sa.status === "computing" || sb.status === "computing") {
    return (
      <p className="busy small" data-collision="computing">
        Colouring… <span className="progress" aria-hidden="true" />
      </p>
    );
  }
  return null;
}

function ReadoutList(props: { readouts: Readouts | null; bound: number }) {
  const { readouts: r, bound } = props;
  const dash = "—";
  const pen = r?.penetration ?? null;
  // A bounded field (fast mode) measures within its bound only; a readout
  // keeps the bound it was measured with.
  const within = r === null ? bound : r.bound;
  return (
    <>
      <dl className="readouts" data-readouts={r === null ? "none" : "ready"}>
        <dt>Max penetration</dt>
        <dd>
          {pen === null
            ? dash
            : pen.atLeast
              ? `≥ ${formatDistance(Math.max(pen.depth, within))}`
              : formatDistance(pen.depth)}
          {pen !== null && Number.isFinite(within) ? (
            <span className="muted"> (measured within {formatDistance(within)})</span>
          ) : null}
        </dd>
        <dt>Min gap</dt>
        <dd>{r === null || r.gap === null ? dash : formatDistance(r.gap)}</dd>
        <dt>Intersecting</dt>
        <dd>{r === null ? dash : r.intersects ? "yes" : "no"}</dd>
        <dt title="Triangles of each mesh within the contact tolerance of the other">In contact</dt>
        <dd>
          {r === null ? dash : `${r.contacts.a.toLocaleString("en")} · ${r.contacts.b.toLocaleString("en")} triangles`}
        </dd>
      </dl>
      {Number.isFinite(bound) ? (
        <p className="muted small">
          Fast mode: distances are measured within {formatDistance(bound)}; deeper penetration shows the mesh colour.
        </p>
      ) : null}
    </>
  );
}

/** The heatmap's colours from −depth to the gap, as `contactHeatmap` maps them by default. */
function Legend(props: { depth: number; gap: number; tolerance: number }) {
  const { depth, gap, tolerance } = props;
  const top = gap > 0 ? gap : tolerance;
  const at = (v: number) => `${(((v + depth) / (depth + top)) * 100).toFixed(2)}%`;
  const stops =
    gap > 0
      ? `#7f0000 0%, #ff6b6b ${at(-tolerance)}, #ffffff ${at(-tolerance)}, #ffffff ${at(tolerance)}, #4caf50 ${at(tolerance)}, #1e3a8a 100%`
      : `#7f0000 0%, #ff6b6b ${at(-tolerance)}, #ffffff ${at(-tolerance)}, #ffffff 100%`;
  return (
    <div className="legend" aria-hidden="true">
      <div className="legend-bar" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <div className="legend-labels">
        <span>−{formatDistance(depth)}</span>
        {gap > 0 ? (
          <span className="legend-zero" style={{ left: at(0) }}>
            0
          </span>
        ) : null}
        <span>{gap > 0 ? formatDistance(gap) : `${formatDistance(tolerance)}, then the mesh colour`}</span>
      </div>
    </div>
  );
}

function Slider(props: {
  label: string;
  name: string;
  min: number;
  max: number;
  step: number;
  value: number;
  title: string;
  onChange: (v: number) => void;
}) {
  const { label, name, min, max, step, value, title, onChange } = props;
  return (
    <label className="slider" title={title}>
      <span className="slider-head">
        <span>{label}</span>
        <output>{formatDistance(value)}</output>
      </span>
      <input
        type="range"
        data-control={name}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={max <= min}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
      />
    </label>
  );
}
