// The inspect panel (#110): in-viewport inspection tools. The ruler
// (click two vertices, world-space points that survive placement undo),
// click-to-query (the picked vertex's coloring value, through the
// thickness/curvature/distance colorings), a draggable section plane
// (wired to `setSectionPlane` through the store) and an axis-aligned clip
// box for looking inside meshes. Scalar mesh numbers reuse `measure`.
import type { FieldHandle, MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useRef, useState } from "react";
import {
  type AppState,
  type InspectTool,
  type Item,
  type SurfaceColormap,
  type SurfaceFieldMode,
  type SurfaceFieldState,
  useApp,
} from "../scene/app-state";
import {
  applySurfaceField,
  clearAppliedSurfaceField,
  isWorkerScene,
} from "../scene/workbench-ops";
import { formatCoord, readMeasure, rulerDistance } from "../viewport/inspect-tools";
import { inspect } from "./inspect";
import { LengthInput } from "./LengthInput";
import { formatArea, formatCoordLength, formatLength, formatPoint, formatVolume, type LengthUnit } from "./units";

const TOOLS: { tool: InspectTool; label: string; title: string }[] = [
  { tool: "ruler", label: "Ruler", title: "Click two vertices to measure their distance (snaps to vertices)" },
  { tool: "query", label: "Query", title: "Click a vertex to read its coloring value" },
  { tool: "section", label: "Section", title: "Cut the meshes with a draggable plane" },
  { tool: "clip", label: "Clip box", title: "Keep an axis-aligned world-space box, clip the rest away" },
];

type Axis = "x" | "y" | "z";
const AXES: Axis[] = ["x", "y", "z"];
const AXIS_NORMAL: Record<Axis, [number, number, number]> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

export function InspectPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const tool = useApp(app, (s) => s.inspectTool);
  const ruler = useApp(app, (s) => s.ruler);
  const query = useApp(app, (s) => s.inspectQuery);
  const section = useApp(app, (s) => s.inspectSection);
  const clip = useApp(app, (s) => s.clipBox);
  const units = useApp(app, (s) => s.units);
  const overlayValues = useApp(app, (s) => s.surfaceFieldValues);
  const item = items.find((i) => i.mesh === selection) ?? null;

  const distance = ruler !== null && ruler.b !== null ? rulerDistance(ruler.a, ruler.b) : null;

  // The query's coloring value (whatever the mesh is colored by, including client-side surface fields)
  // and the mesh's scalars, refreshed per hit. A mesh without a coloring has none.
  const [queryValue, setQueryValue] = useState<number | null>(null);
  const [queryMeasure, setQueryMeasure] = useState<{ area: number; volume: number | null } | null>(null);
  useEffect(() => {
    if (query === null) {
      setQueryValue(null);
      setQueryMeasure(null);
      return;
    }
    let alive = true;
    if (overlayValues !== null && overlayValues.mesh === query.mesh) {
      const val = overlayValues.values[query.slot];
      setQueryValue(val !== undefined && Number.isFinite(val) ? val : null);
    } else {
      store
        .fetchColoringValues(query.mesh)
        .then((values) => {
          if (alive) {
            setQueryValue(values === null ? null : (values[query.slot] ?? null));
          }
        })
        .catch(() => {
          if (alive) {
            setQueryValue(null);
          }
        });
    }
    void readMeasure(scene, query.mesh).then((m) => {
      if (alive) {
        setQueryMeasure(m);
      }
    });
    return () => {
      alive = false;
    };
  }, [store, scene, query, overlayValues]);

  // The test hook's view of the tools (names, not handles).
  useEffect(() => {
    const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? null;
    inspect.tools.tool = tool;
    inspect.tools.ruler =
      ruler === null
        ? { a: [0, 0, 0], b: null, distance: null }
        : { a: [...ruler.a], b: ruler.b === null ? null : [...ruler.b], distance };
    inspect.tools.query =
      query === null
        ? null
        : {
            mesh: nameOf(query.mesh) ?? "?",
            slot: query.slot,
            point: [...query.point],
            value: queryValue,
          };
    inspect.tools.section =
      section === null
        ? null
        : {
            plane: { origin: [...section.plane.origin], normal: [...section.plane.normal] },
            meshes: items.length,
          };
    inspect.tools.clip = clip === null ? null : { min: [...clip.min], max: [...clip.max] };
  });

  return (
    <section className="panel" aria-label="Inspect">
      <h2>Inspect</h2>
      <div className="modes" role="group" aria-label="Inspection tool">
        {TOOLS.map((t) => (
          <button
            key={t.tool}
            type="button"
            data-action={`inspect-${t.tool}`}
            aria-pressed={tool === t.tool}
            title={t.title}
            onClick={() => app.setInspectTool(tool === t.tool ? null : t.tool)}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tool === "ruler" ? (
        <p className="muted small">Click two vertices in the viewport; Esc clears the selection first.</p>
      ) : null}
      {tool === "query" ? (
        <p className="muted small">Click a vertex in the viewport to read its coloring value.</p>
      ) : null}
      <RulerReadout app={app} ruler={ruler} distance={distance} unit={units} />
      <QueryReadout query={query} items={items} value={queryValue} measure={queryMeasure} unit={units} />
      <SectionControls app={app} scene={scene} store={store} items={items} item={item} section={section} active={tool === "section"} unit={units} />
      {tool === "clip" || clip !== null ? <ClipControls app={app} scene={scene} items={items} item={item} clip={clip} unit={units} /> : null}
      <SurfaceFieldControls app={app} items={items} selection={selection} unit={units} />
    </section>
  );
}

const SURFACE_FIELD_MODES: readonly { id: SurfaceFieldMode; label: string; title: string }[] = [
  { id: "none", label: "Off", title: "Turn off surface field analysis" },
  { id: "distance", label: "Distance", title: "Contact distance field to another mesh" },
  { id: "thickness", label: "Thickness", title: "Wall thickness to opposite surface" },
  { id: "meanCurvature", label: "Mean κ", title: "Signed mean curvature (addCurvatureField)" },
  { id: "gaussianCurvature", label: "Gauss κ", title: "Discrete Gaussian curvature (angle defect / area)" },
];

const COLORMAPS: readonly { id: SurfaceColormap; label: string; gradient: string }[] = [
  { id: "turbo", label: "Turbo", gradient: "linear-gradient(90deg,#30123b,#287ced,#2ed673,#f9b724,#7a0403)" },
  { id: "viridis", label: "Viridis", gradient: "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)" },
  { id: "coolwarm", label: "Coolwarm", gradient: "linear-gradient(90deg,#3b4cc0,#dddddd,#b40426)" },
  { id: "grayscale", label: "Grayscale", gradient: "linear-gradient(90deg,#000000,#ffffff)" },
];

function SurfaceFieldControls(props: {
  app: AppState;
  items: readonly Item[];
  selection: MeshHandle | null;
  unit: LengthUnit;
}) {
  const { app, items, selection, unit } = props;
  const store = useScene();
  const sf = useApp(app, (s) => s.surfaceField);
  const overlayValues = useApp(app, (s) => s.surfaceFieldValues);

  const activeFieldRef = useRef<{ mesh: MeshHandle | null; field: FieldHandle | null }>({ mesh: null, field: null });
  const [summary, setSummary] = useState<{ min: number; max: number; mean: number; count: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inspectedMesh =
    (sf.mesh !== null && items.some((i) => i.mesh === sf.mesh) ? sf.mesh : null) ??
    (selection !== null && items.some((i) => i.mesh === selection) ? selection : null) ??
    items[0]?.mesh ??
    null;
  const targetMesh =
    (sf.target !== null && items.some((i) => i.mesh === sf.target) && sf.target !== inspectedMesh ? sf.target : null) ??
    items.find((i) => i.mesh !== inspectedMesh)?.mesh ??
    null;

  const computeAndApply = async (nextState: SurfaceFieldState, autoRange: boolean) => {
    clearAppliedSurfaceField(store, app, activeFieldRef.current);
    activeFieldRef.current = { mesh: null, field: null };
    setError(null);
    if (nextState.mode === "none" || nextState.mesh === null) {
      setSummary(null);
      return;
    }
    setBusy(true);
    try {
      const applied = await applySurfaceField({ store, app, state: nextState, autoRange });
      activeFieldRef.current = { mesh: applied.mesh, field: applied.field };
      setSummary(applied.summary);
    } catch (e) {
      setSummary(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const switchMode = (mode: SurfaceFieldMode) => {
    const nextState: SurfaceFieldState = {
      ...sf,
      mode,
      mesh: inspectedMesh,
      target: targetMesh,
    };
    app.setSurfaceField(nextState);
    void computeAndApply(nextState, true);
  };

  const updateAppearance = (patch: { map?: SurfaceColormap; min?: number; max?: number }) => {
    const nextMap = patch.map ?? sf.map;
    const nextMin = patch.min ?? sf.min;
    const nextMax = patch.max ?? sf.max;
    app.setSurfaceField({ map: nextMap, min: nextMin, max: nextMax });
    if (sf.mode === "none" || inspectedMesh === null) return;
    const lo = nextMin;
    const hi = nextMax > nextMin ? nextMax : nextMin + 1;

    if (overlayValues !== null && overlayValues.mesh === inspectedMesh) {
      app.setSurfaceFieldValues({ ...overlayValues, map: nextMap, min: lo, max: hi });
      return;
    }
    if (sf.mode === "distance" && isWorkerScene(store.scene) && targetMesh !== null && targetMesh !== inspectedMesh) {
      store.setDisplay(inspectedMesh, {
        coloring: { distanceTo: targetMesh, maxDistance: Math.max(sf.maxDistance, 1e-4), map: nextMap, range: [lo, hi] },
      });
      return;
    }
    if (!isWorkerScene(store.scene) && activeFieldRef.current.field !== null) {
      store.scene.inspect?.setColoring(inspectedMesh, {
        field: activeFieldRef.current.field,
        map: nextMap,
        range: [lo, hi],
      });
    }
  };

  const mapMeta = COLORMAPS.find((c) => c.id === sf.map) ?? COLORMAPS[0];

  return (
    <div className="op surface-field-panel">
      <h3>Surface Field Analysis</h3>
      <div className="modes" role="group" aria-label="Surface field mode">
        {SURFACE_FIELD_MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            data-inspect-field-mode={m.id}
            aria-pressed={sf.mode === m.id}
            title={m.title}
            disabled={busy || (m.id !== "none" && items.length === 0) || (m.id === "distance" && items.length < 2)}
            onClick={() => switchMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {sf.mode !== "none" && items.length > 0 ? (
        <>
          <div className="picker">
            <span>On</span>
            <select
              aria-label="Field source mesh"
              data-inspect-field-mesh
              value={inspectedMesh === null ? "" : String(items.findIndex((i) => i.mesh === inspectedMesh))}
              disabled={busy}
              onChange={(e) => {
                const nextMesh = items[Number(e.currentTarget.value)]?.mesh ?? null;
                const nextTarget = targetMesh === nextMesh ? (items.find((i) => i.mesh !== nextMesh)?.mesh ?? null) : targetMesh;
                const nextState = { ...sf, mesh: nextMesh, target: nextTarget };
                app.setSurfaceField(nextState);
                void computeAndApply(nextState, true);
              }}
            >
              {items.map((it, idx) => (
                <option key={it.id} value={String(idx)}>
                  {it.name}
                </option>
              ))}
            </select>
          </div>

          {sf.mode === "distance" ? (
            <div className="picker">
              <span>To</span>
              <select
                aria-label="Distance target mesh"
                data-inspect-field-target
                value={targetMesh === null ? "" : String(items.findIndex((i) => i.mesh === targetMesh))}
                disabled={busy}
                onChange={(e) => {
                  const nextTarget = items[Number(e.currentTarget.value)]?.mesh ?? null;
                  const nextState = { ...sf, mesh: inspectedMesh, target: nextTarget };
                  app.setSurfaceField(nextState);
                  void computeAndApply(nextState, true);
                }}
              >
                {items
                  .filter((it) => it.mesh !== inspectedMesh)
                  .map((it) => {
                    const idx = items.indexOf(it);
                    return (
                      <option key={it.id} value={String(idx)}>
                        {it.name}
                      </option>
                    );
                  })}
              </select>
            </div>
          ) : null}

          <div className="op-row">
            <label className="field">
              colormap
              <select
                aria-label="Field colormap"
                data-inspect-field-map
                value={sf.map}
                onChange={(e) => updateAppearance({ map: e.currentTarget.value as SurfaceColormap })}
              >
                {COLORMAPS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
            {(sf.mode === "distance" || sf.mode === "thickness") ? (
              <label className="field">
                limit ({unit})
                <LengthInput
                  aria-label="Field max distance"
                  data-inspect-field-maxdist
                  min={0.01}
                  mm={sf.maxDistance}
                  unit={unit}
                  onMm={(mm) => {
                    const nextMax = Math.max(0.01, mm);
                    const nextState = { ...sf, mesh: inspectedMesh, target: targetMesh, maxDistance: nextMax };
                    app.setSurfaceField({ maxDistance: nextMax });
                    void computeAndApply(nextState, false);
                  }}
                />
              </label>
            ) : null}
          </div>

          <div className="op-row">
            <label className="field">
              min
              <input
                type="number"
                step="any"
                aria-label="Field range minimum"
                data-inspect-field-min
                value={Number(sf.min.toFixed(4))}
                onChange={(e) => {
                  const v = Number(e.currentTarget.value);
                  if (Number.isFinite(v)) updateAppearance({ min: v });
                }}
              />
            </label>
            <label className="field">
              max
              <input
                type="number"
                step="any"
                aria-label="Field range maximum"
                data-inspect-field-max
                value={Number(sf.max.toFixed(4))}
                onChange={(e) => {
                  const v = Number(e.currentTarget.value);
                  if (Number.isFinite(v)) updateAppearance({ max: v });
                }}
              />
            </label>
            <button
              type="button"
              data-inspect-field-auto
              disabled={summary === null}
              onClick={() => {
                if (summary === null) return;
                const lo = summary.min;
                const hi = summary.max > summary.min ? summary.max : summary.min + 1;
                updateAppearance({ min: lo, max: hi });
              }}
            >
              Auto
            </button>
          </div>

          <div className="legend">
            <div className="legend-bar" style={{ background: mapMeta.gradient }} />
            <div className="legend-labels">
              <span>{sf.min.toPrecision(3)}</span>
              <span>{sf.max.toPrecision(3)}</span>
            </div>
          </div>

          {summary !== null ? (
            <p className="muted" data-inspect-field-stats>
              min {summary.min.toPrecision(3)} · mean {summary.mean.toPrecision(3)} · max {summary.max.toPrecision(3)}
            </p>
          ) : null}
        </>
      ) : null}

      {busy ? <p className="muted busy">Computing surface field...</p> : null}
      {error !== null ? <p className="report error">{error}</p> : null}
    </div>
  );
}

function RulerReadout(props: {
  app: AppState;
  ruler: { a: [number, number, number]; b: [number, number, number] | null } | null;
  distance: number | null;
  unit: LengthUnit;
}) {
  const { app, ruler, distance, unit } = props;
  if (ruler === null) {
    return null;
  }
  return (
    <p className="readout" data-readout="ruler">
      A {formatPoint(ruler.a, unit)}
      {ruler.b === null ? " · click a second vertex" : ` → B ${formatPoint(ruler.b, unit)}`}{" "}
      {distance === null ? null : (
        <strong>= {formatLength(distance, unit)}</strong>
      )}{" "}
      <button type="button" data-action="ruler-clear" onClick={() => app.setRuler(null)} title="Clear the ruler">
        Clear
      </button>
    </p>
  );
}

function QueryReadout(props: {
  query: { mesh: MeshHandle; slot: number; point: [number, number, number] } | null;
  items: readonly Item[];
  value: number | null;
  measure: { area: number; volume: number | null } | null;
  unit: LengthUnit;
}) {
  const { query, items, value, measure, unit } = props;
  if (query === null) {
    return null;
  }
  const name = items.find((i) => i.mesh === query.mesh)?.name ?? "a mesh";
  return (
    <div className="readout" data-readout="query">
      <p>
        {name} vertex {query.slot} at {formatPoint(query.point, unit)}
      </p>
      <p>
        coloring value: {value === null ? "no coloring" : formatCoordLength(value, unit)}
        {measure === null ? null : (
          <>
            {" · "}area {formatArea(measure.area, unit)}
            {measure.volume === null ? " · open mesh (no volume)" : ` · volume ${formatVolume(measure.volume, unit)}`}
          </>
        )}
      </p>
    </div>
  );
}

function SectionControls(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  store: ReturnType<typeof useScene>;
  items: readonly Item[];
  item: Item | null;
  section: { handle: unknown; plane: { origin: readonly [number, number, number]; normal: readonly [number, number, number] } } | null;
  active: boolean;
  unit: LengthUnit;
}) {
  const { app, scene, store, items, item, section, active, unit } = props;
  if (!active && section === null) {
    return null;
  }
  const center = sectionCenter(scene, section, item);
  const normal = section?.plane.normal ?? ([0, 0, 1] as readonly [number, number, number]);
  const axis: Axis = Math.abs(normal[0]) >= Math.abs(normal[1]) && Math.abs(normal[0]) >= Math.abs(normal[2]) ? "x" : Math.abs(normal[1]) >= Math.abs(normal[2]) ? "y" : "z";
  const range = Math.max(item?.radius ?? 1, 1e-6);
  const offset =
    section === null || center === null
      ? 0
      : (section.plane.origin[0] - center[0]) * normal[0] +
        (section.plane.origin[1] - center[1]) * normal[1] +
        (section.plane.origin[2] - center[2]) * normal[2];

  const enable = () => {
    if (items.length === 0) {
      return;
    }
    const c = center ?? [0, 0, 0];
    const plane = { origin: [...c] as [number, number, number], normal: [0, 0, 1] as [number, number, number] };
    const handle = store.addSection({ plane, meshes: items.map((i) => i.mesh), fill: false });
    app.setInspectSection({ handle, plane });
  };
  const move = (plane: { origin: [number, number, number]; normal: [number, number, number] }) => {
    if (section === null) {
      return;
    }
    const handle = section.handle as import("fast-mesh-wasm").SectionHandle;
    try {
      store.setSection(handle, { plane });
    } catch {
      // Removed meanwhile: the gizmo cleans up.
    }
    app.setInspectSection({ handle, plane });
  };

  return (
    <div className="readout" data-readout="section">
      {section === null ? (
        <p>
          <button type="button" data-action="section-enable" disabled={items.length === 0} onClick={enable} title="Cut the meshes with a plane">
            Cut with a plane
          </button>
        </p>
      ) : (
        <>
          <p className="muted small">
            origin {formatPoint(section.plane.origin, unit)} · normal (
            {section.plane.normal.map(formatCoord).join(", ")}) · drag the handle in the viewport
          </p>
          <div className="modes" role="group" aria-label="Section axis">
            {AXES.map((a) => (
              <button
                key={a}
                type="button"
                data-axis={a}
                aria-pressed={axis === a}
                onClick={() => {
                  const n = AXIS_NORMAL[a];
                  const c = center ?? [0, 0, 0];
                  move({ origin: [...c] as [number, number, number], normal: [...n] });
                }}
              >
                {a.toUpperCase()}
              </button>
            ))}
            <button
              type="button"
              data-action="section-flip"
              title="Flip the kept side"
              onClick={() => {
                const n = section.plane.normal;
                move({
                  origin: [...section.plane.origin] as [number, number, number],
                  normal: [-n[0], -n[1], -n[2]] as [number, number, number],
                });
              }}
            >
              Flip
            </button>
          </div>
          <label className="slider">
            offset <output>{formatCoordLength(offset, unit)}</output>
            <input
              type="range"
              min={-range}
              max={range}
              step={range / 200}
              value={Math.min(range, Math.max(-range, offset))}
              disabled={center === null}
              onChange={(e) => {
                const c = center;
                if (c === null) {
                  return;
                }
                const t = Number(e.currentTarget.value);
                move({
                  origin: [c[0] + normal[0] * t, c[1] + normal[1] * t, c[2] + normal[2] * t],
                  normal: [...normal] as [number, number, number],
                });
              }}
            />
          </label>
          <p>
            <button
              type="button"
              data-action="section-disable"
              onClick={() => {
                try {
                  store.removeSection(section.handle as import("fast-mesh-wasm").SectionHandle);
                } catch {
                  // Removed meanwhile.
                }
                app.setInspectSection(null);
              }}
            >
              Remove plane
            </button>
          </p>
        </>
      )}
    </div>
  );
}

/**
 * The section's anchor: the selected mesh's world center (its local center
 * through the placement, so the offset slider follows gizmo moves).
 */
function sectionCenter(
  scene: Scene | WorkerScene,
  section: { plane: { origin: readonly [number, number, number] } } | null,
  item: Item | null,
): [number, number, number] | null {
  void section;
  if (item === null) {
    return null;
  }
  try {
    const m = scene.matrix4(item.mesh);
    return [
      m[0] * item.center[0] + m[4] * item.center[1] + m[8] * item.center[2] + m[12],
      m[1] * item.center[0] + m[5] * item.center[1] + m[9] * item.center[2] + m[13],
      m[2] * item.center[0] + m[6] * item.center[1] + m[10] * item.center[2] + m[14],
    ];
  } catch {
    return null; // removed meanwhile
  }
}

function ClipControls(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  item: Item | null;
  clip: { min: [number, number, number]; max: [number, number, number] } | null;
  unit: LengthUnit;
}) {
  const { app, scene, items, item, clip, unit } = props;
  if (clip === null) {
    return (
      <p>
        <button
          type="button"
          data-action="clip-fit"
          disabled={item === null}
          onClick={() => {
            const fit = item === null ? null : fitWorldBox(scene, item);
            if (fit !== null) {
              app.setClipBox(fit);
            }
          }}
          title="Keep the selected mesh's middle half"
        >
          Clip to selection
        </button>
      </p>
    );
  }
  const span: [number, number, number] = [
    Math.max(clip.max[0] - clip.min[0], 1e-6),
    Math.max(clip.max[1] - clip.min[1], 1e-6),
    Math.max(clip.max[2] - clip.min[2], 1e-6),
  ];
  const lo: [number, number, number] = [clip.min[0] - span[0], clip.min[1] - span[1], clip.min[2] - span[2]];
  const hi: [number, number, number] = [clip.max[0] + span[0], clip.max[1] + span[1], clip.max[2] + span[2]];
  void items;
  const set = (axis: number, end: 0 | 1, v: number) => {
    const min: [number, number, number] = [...clip.min];
    const max: [number, number, number] = [...clip.max];
    if (end === 0) {
      min[axis] = Math.min(v, max[axis]);
    } else {
      max[axis] = Math.max(v, min[axis]);
    }
    app.setClipBox({ min, max });
  };
  return (
    <div className="readout" data-readout="clip">
      {AXES.map((a, i) => (
        <div key={a}>
          <label className="slider">
            {a} min <output>{formatCoordLength(clip.min[i], unit)}</output>
            <input
              type="range"
              min={lo[i]}
              max={hi[i]}
              step={(hi[i] - lo[i]) / 200}
              value={clip.min[i]}
              onChange={(e) => set(i, 0, Number(e.currentTarget.value))}
            />
          </label>
          <label className="slider">
            {a} max <output>{formatCoordLength(clip.max[i], unit)}</output>
            <input
              type="range"
              min={lo[i]}
              max={hi[i]}
              step={(hi[i] - lo[i]) / 200}
              value={clip.max[i]}
              onChange={(e) => set(i, 1, Number(e.currentTarget.value))}
            />
          </label>
        </div>
      ))}
      <p>
        <button type="button" data-action="clip-clear" onClick={() => app.setClipBox(null)}>
          Clear box
        </button>
      </p>
    </div>
  );
}

/**
 * The selected mesh's middle half in world coordinates: its local center
 * through the placement (scale-aware), half the world radius each way.
 */
function fitWorldBox(
  scene: Scene | WorkerScene,
  item: Item,
): { min: [number, number, number]; max: [number, number, number] } | null {
  try {
    const m = scene.matrix4(item.mesh);
    const cx = m[0] * item.center[0] + m[4] * item.center[1] + m[8] * item.center[2] + m[12];
    const cy = m[1] * item.center[0] + m[5] * item.center[1] + m[9] * item.center[2] + m[13];
    const cz = m[2] * item.center[0] + m[6] * item.center[1] + m[10] * item.center[2] + m[14];
    // The placement's axis scales (columns' lengths): non-uniform scale (#103) stretches the box.
    const sx = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    const sy = Math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6]);
    const sz = Math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
    const r = Math.max(item.radius, 1e-6) / 2;
    return {
      min: [cx - sx * r, cy - sy * r, cz - sz * r],
      max: [cx + sx * r, cy + sy * r, cz + sz * r],
    };
  } catch {
    return null; // removed meanwhile
  }
}
