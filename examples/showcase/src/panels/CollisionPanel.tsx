// The collision panel: the contact heatmap of two meshes, each coloured by
// its signed distance to the other (`contactHeatmap`), with readouts of the
// deepest penetration, the gap, whether they intersect and the triangles in
// contact. The heatmap controls are shared; the pair below them is one mode
// (a later "Group" mode colours several meshes with the same controls).
import type { FieldHandle, MeshHandle, Scene } from "fast-mesh-wasm";
import { useMeshDisplay, useScene } from "fast-mesh-wasm/react";
import { contactHeatmap, type DistanceColoring, type SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo, useRef, useState } from "react";
import { type AppState, type CollisionPair, type Item, useApp } from "../scene/app-state";
import { view } from "../viewport/view";
import { clampTolerance, deepestPenetration, formatDistance, niceScale, toleranceLimit } from "./collision-math";
import { inspect, type Readouts } from "./inspect";

/** The readouts refresh at most this often (about 4 Hz), and once more after the last change. */
const READOUT_PERIOD_MS = 250;

type Heatmap = Omit<DistanceColoring, "distanceTo">;

interface HeatmapParams {
  depth: number;
  gap: number;
  tolerance: number;
}

export function CollisionPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const items = useApp(app, (s) => s.items);
  const pair = useApp(app, (s) => s.collision);
  const itemA = items.find((i) => i.mesh === pair.a) ?? null;
  const itemB = items.find((i) => i.mesh === pair.b) ?? null;

  // The sliders' scale follows the pair's size: a file in millimetres and
  // the unit-sized demo meshes both get useful ranges.
  const scale = niceScale(Math.max(itemA?.radius ?? 0, itemB?.radius ?? 0));
  const step = scale / 1000;
  // `null` until a slider moves: the defaults then follow the scale.
  const [params, setParams] = useState<HeatmapParams | null>(null);
  const [fast, setFast] = useState(false);
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
      <p className="muted small">
        Each mesh coloured by its distance to the other: penetration in reds, contact white, the gap green to blue.
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
      <Legend depth={depth} gap={gap} tolerance={tolerance} />
      <label
        className="check"
        title="Bound the fields at max(depth, colour from): much faster on large meshes, but deeper penetration shows the mesh colour"
      >
        <input type="checkbox" data-control="fast" checked={fast} onChange={(e) => setFast(e.currentTarget.checked)} />
        Fast mode (large meshes)
      </label>
      <PairMode app={app} scene={scene} items={items} pair={pair} heat={heat} tolerance={tolerance} />
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
  const readouts = useReadouts(store, scene, ready ? a : null, ready ? b : null, tolerance, heat.maxDistance, nameOf);

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
          <ReadoutList readouts={readouts} exact={heat.maxDistance === Infinity} bound={heat.maxDistance} />
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

function ReadoutList(props: { readouts: Readouts | null; exact: boolean; bound: number }) {
  const { readouts: r, exact, bound } = props;
  const dash = "—";
  const pen = r?.penetration ?? null;
  return (
    <>
      <dl className="readouts" data-readouts={r === null ? "none" : "ready"}>
        <dt>Max penetration</dt>
        <dd>{pen === null ? dash : pen.atBound ? `≥ ${formatDistance(bound)}` : formatDistance(pen.depth)}</dd>
        <dt>Min gap</dt>
        <dd>{r === null || r.gap === null ? dash : formatDistance(r.gap)}</dd>
        <dt>Intersecting</dt>
        <dd>{r === null ? dash : r.intersects ? "yes" : "no"}</dd>
        <dt title="Triangles of each mesh within the contact tolerance of the other">In contact</dt>
        <dd>
          {r === null ? dash : `${r.contacts.a.toLocaleString("en")} · ${r.contacts.b.toLocaleString("en")} triangles`}
        </dd>
      </dl>
      {exact ? null : (
        <p className="muted small">
          Fast mode: penetration beyond {formatDistance(bound)} is not measured (mesh colour).
        </p>
      )}
      {r?.warnings.map((w) => (
        <p key={w} className="warn small" data-warning>
          {w}
        </p>
      ))}
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

/**
 * The pair's readouts, refreshed when either mesh moves, changes or is
 * recoloured (its change records), at most every READOUT_PERIOD_MS and once
 * more after the last change (so a drag's end is always read). One query
 * at a time; `null` without a pair. A mesh removed mid-query rejects: that
 * refresh is dropped.
 */
function useReadouts(
  store: SceneStore,
  scene: Scene | WorkerScene,
  a: MeshHandle | null,
  b: MeshHandle | null,
  tolerance: number,
  bound: number,
  nameOf: (m: MeshHandle) => string,
): Readouts | null {
  const [readouts, setReadouts] = useState<Readouts | null>(null);
  const live = useRef({ tolerance, nameOf });
  live.current = { tolerance, nameOf };
  const poke = useRef<() => void>(() => {});

  // The latest field on each mesh, from the change records (the store owns
  // it; the app adds none): for `fieldStatus`. Kept across pair changes, as
  // a colouring turned back on reuses its field without recomputing it.
  const fields = useRef(new Map<MeshHandle, FieldHandle>());
  useEffect(
    () =>
      store.scene.onChanges((changes) => {
        for (const c of changes) {
          if (c.removed) {
            fields.current.delete(c.mesh);
          } else if (c.fieldsChanged.length > 0) {
            fields.current.set(c.mesh, c.fieldsChanged[c.fieldsChanged.length - 1]);
          }
        }
      }),
    [store],
  );

  useEffect(() => {
    if (a === null || b === null) {
      setReadouts(null);
      inspect.collision.readouts = null;
      return;
    }
    setReadouts(null);
    let alive = true;
    let dirty = true;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const s = scene;

    const status = async (on: MeshHandle) => {
      const f = fields.current.get(on);
      return f === undefined ? null : await Promise.resolve(s.fieldStatus(f)).catch(() => null);
    };
    const query = async (): Promise<Readouts> => {
      const { tolerance: margin, nameOf: name } = live.current;
      const [va, vb, distance, hit, contacts, onA, onB] = await Promise.all([
        // After a commit (this runs from change records): the colouring last
        // sent. A colouring replaced meanwhile rejects; read it as none.
        store.fetchColoringValues(a).catch(() => null),
        store.fetchColoringValues(b).catch(() => null),
        Promise.resolve(s.meshDistance(a, b)),
        Promise.resolve(s.intersects(a, b)),
        Promise.resolve(s.findContacts({ margin, pairs: [[a, b]] })),
        status(a),
        status(b),
      ]);
      const c = contacts[0];
      const warnings: string[] = [];
      // The field on A measures to B: its sign data describes B, and the other way round.
      for (const [st, target] of [
        [onA, b],
        [onB, a],
      ] as const) {
        if (st === null) {
          continue;
        }
        if (!st.sign.closed) {
          warnings.push(
            `${name(target)} is open: penetration side follows its surface normals; repair → orient/fill holes for a closed result`,
          );
        } else if (!st.sign.consistentOrientation) {
          warnings.push(
            `${name(target)} is inconsistently oriented: penetration side follows its surface normals; repair → orient for a closed result`,
          );
        }
      }
      return {
        penetration: deepestPenetration([va, vb], bound),
        gap: distance === null ? null : distance.distance,
        intersects: hit,
        contacts:
          c === undefined
            ? { a: 0, b: 0 }
            : c.a === a
              ? { a: c.trianglesA.length, b: c.trianglesB.length }
              : { a: c.trianglesB.length, b: c.trianglesA.length },
        warnings,
      };
    };
    const kick = () => {
      inspect.collision.settled = !dirty && !running;
      if (!alive || running || !dirty) {
        return;
      }
      const wait = last + READOUT_PERIOD_MS - performance.now();
      if (wait > 0) {
        timer ??= setTimeout(() => {
          timer = undefined;
          kick();
        }, wait);
        return;
      }
      dirty = false;
      running = true;
      inspect.collision.settled = false;
      last = performance.now();
      query()
        .then(
          (r) => {
            if (!alive) {
              return;
            }
            setReadouts(r);
            inspect.collision.readouts = r;
            inspect.collision.updates++;
            if (view.dragging) {
              inspect.collision.updatesWhileDragging++;
            }
          },
          () => {
            // A mesh removed (or the scene restarted) mid-query: skip this refresh.
          },
        )
        .finally(() => {
          running = false;
          kick();
        });
    };

    const off = store.scene.onChanges((changes) => {
      for (const c of changes) {
        if (c.mesh !== a && c.mesh !== b) {
          continue;
        }
        const changed = (r: readonly unknown[] | "all") => r === "all" || r.length > 0;
        if (c.placementChanged || changed(c.colors) || changed(c.indices) || c.fieldsChanged.length > 0) {
          dirty = true;
        }
      }
      kick();
    });
    poke.current = () => {
      dirty = true;
      kick();
    };
    kick();
    return () => {
      alive = false;
      off();
      clearTimeout(timer);
      poke.current = () => {};
      inspect.collision.readouts = null;
      inspect.collision.settled = true;
    };
  }, [store, scene, a, b, bound]);

  // A new tolerance changes the contact count only (the colours follow through change records).
  useEffect(() => poke.current(), [tolerance]);

  return a === null || b === null ? null : readouts;
}
