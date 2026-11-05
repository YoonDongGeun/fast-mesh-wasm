// The inspect panel (#110): in-viewport inspection tools. The ruler
// (click two vertices, world-space points that survive placement undo),
// click-to-query (the picked vertex's coloring value, through the
// thickness/curvature/distance colorings), a draggable section plane
// (wired to `setSectionPlane` through the store) and an axis-aligned clip
// box for looking inside meshes. Scalar mesh numbers reuse `measure`.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useState } from "react";
import { type AppState, type InspectTool, type Item, useApp } from "../scene/app-state";
import { formatCoord, readMeasure, rulerDistance } from "../viewport/inspect-tools";
import { inspect } from "./inspect";
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
  const item = items.find((i) => i.mesh === selection) ?? null;

  const distance = ruler !== null && ruler.b !== null ? rulerDistance(ruler.a, ruler.b) : null;

  // The query's coloring value (whatever the mesh is colored by) and the
  // mesh's scalars, refreshed per hit. A mesh without a coloring has none.
  const [queryValue, setQueryValue] = useState<number | null>(null);
  const [queryMeasure, setQueryMeasure] = useState<{ area: number; volume: number | null } | null>(null);
  useEffect(() => {
    if (query === null) {
      setQueryValue(null);
      setQueryMeasure(null);
      return;
    }
    let alive = true;
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
    void readMeasure(scene, query.mesh).then((m) => {
      if (alive) {
        setQueryMeasure(m);
      }
    });
    return () => {
      alive = false;
    };
  }, [store, scene, query]);

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
    </section>
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
