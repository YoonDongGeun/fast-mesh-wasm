// The sculpt toolbox: a floating strip over the viewport for the selected
// mesh. Four tools arm the sculpt brush (inflate, deflate, smooth, grab),
// with the brush radius and per-tool strength on sliders and the
// stroke-end remesh toggle; the viewport paints the strokes.
import type { MeshHandle } from "fast-mesh-wasm";
import { useSelection } from "fast-mesh-wasm/react";
import { useEffect, useState } from "react";
import { type AppState, useApp } from "../scene/app-state";
import { REMESH_TRI_LIMIT, remeshTooBig } from "./sculpt";

export type SculptTool = "inflate" | "deflate" | "smooth" | "grab";

/**
 * The repair op a tool paints (deflate is inflate with a flipped strength;
 * smooth paints plain-Laplacian relax: Taubin λ/μ converges too slowly to
 * melt under a brush).
 */
const TOOL_OP: Record<SculptTool, "inflate" | "relax" | "grab"> = {
  inflate: "inflate",
  deflate: "inflate",
  smooth: "relax",
  grab: "grab",
};

const TOOLS: { tool: SculptTool; label: string; title: string }[] = [
  { tool: "inflate", label: "Inflate", title: "Puff the surface up along its normals" },
  { tool: "deflate", label: "Deflate", title: "Shrink the surface back along its normals" },
  { tool: "smooth", label: "Smooth", title: "Relax the surface toward its neighbours" },
  { tool: "grab", label: "Grab", title: "Drag the surface along with the pointer" },
];

/**
 * Each tool remembers its own strength (grab at 1 follows the pointer
 * exactly). Inflate stays small: dabs overlap heavily along a sweep, so a
 * dab-sized 0.05 already builds a visible bump without folding.
 */
const DEFAULT_STRENGTHS: Record<SculptTool, number> = { inflate: 0.05, deflate: 0.05, smooth: 0.5, grab: 1 };
/** Laplacian passes per smooth dab (each melts the full weight toward the centroid). */
const RELAX_PASSES = 2;

const fmt = (v: number): string => String(Number(v.toPrecision(3)));

export function SculptToolbox(props: { app: AppState }) {
  const { app } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const sculpt = useApp(app, (s) => s.sculpt);
  const remesh = useApp(app, (s) => s.sculptRemesh);
  const liveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const item = items.find((i) => i.mesh === selection) ?? null;
  const [tool, setTool] = useState<SculptTool>("inflate");
  const [radius, setRadius] = useState(0.25);
  const [strengths, setStrengths] = useState<Record<SculptTool, number>>(DEFAULT_STRENGTHS);
  const [falloff, setFalloff] = useState<"linear" | "smooth">("smooth");
  const armed = sculpt !== null && sculpt.mesh === selection;
  // The radius follows the mesh; the rest are preferences and persist.
  useEffect(() => {
    if (item !== null) {
      setRadius(item.radius / 4);
    }
  }, [item]);
  // While armed, the toolbox drives the sculpt brush live (sliders stay
  // adjustable mid-stroke).
  useEffect(() => {
    if (armed && selection !== null) {
      arm(app, selection, tool, radius, strengths[tool], falloff);
    }
  }, [app, armed, selection, tool, radius, strengths, falloff]);

  if (item === null || selection === null) {
    return null;
  }
  // Remeshes stand down on huge meshes, where one would stall the worker.
  const tooBig = remeshTooBig(item.triangles);
  const remeshTitle = tooBig
    ? `Stands down past ${REMESH_TRI_LIMIT.toLocaleString("en")} triangles: a remesh would stall the worker`
    : null;
  const strength = strengths[tool];
  return (
    <div className="toolbox" role="toolbar" aria-label="Sculpt tools">
      <div className="group">
        {TOOLS.map((t) => (
          <button
            key={t.tool}
            type="button"
            data-action={`sculpt-tool-${t.tool}`}
            aria-pressed={armed && tool === t.tool}
            title={t.title}
            onClick={() => {
              if (armed && tool === t.tool) {
                app.setSculpt(null);
                return;
              }
              setTool(t.tool);
              arm(app, selection, t.tool, radius, strengths[t.tool], falloff);
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      <label className="slider" title="Brush radius in mesh units">
        radius <output>{fmt(radius)}</output>
        <input
          type="range"
          min={0.01}
          max={Math.max(item.radius, 0.02)}
          step={0.01}
          value={Math.min(radius, Math.max(item.radius, 0.02))}
          onChange={(e) => setRadius(Number(e.currentTarget.value))}
        />
      </label>
      <label className="slider" title={tool === "grab" ? "How closely the surface follows the pointer" : "Displacement per dab"}>
        strength <output>{fmt(strength)}</output>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={strength}
          onChange={(e) => setStrengths({ ...strengths, [tool]: Number(e.currentTarget.value) })}
        />
      </label>
      <label className="field" title="How the strength falls off with distance">
        falloff
        <select value={falloff} onChange={(e) => setFalloff(e.currentTarget.value as "linear" | "smooth")}>
          <option value="smooth">smooth</option>
          <option value="linear">linear</option>
        </select>
      </label>
      <label
        className="check"
        title={remeshTitle ?? "Even the stroked region out with a remesh when the stroke ends"}
      >
        <input
          type="checkbox"
          data-action="sculpt-remesh"
          checked={remesh}
          disabled={tooBig}
          onChange={(e) => app.setSculptRemesh(e.currentTarget.checked)}
        />
        even out after stroke
      </label>
      <label
        className="check"
        title={
          remeshTitle ??
          "Split stretched triangles while the stroke paints (never collapses: the count only grows mid-stroke)"
        }
      >
        <input
          type="checkbox"
          data-action="sculpt-live-remesh"
          checked={liveRemesh}
          disabled={tooBig}
          onChange={(e) => app.setSculptLiveRemesh(e.currentTarget.checked)}
        />
        even out while painting
      </label>
    </div>
  );
}

/** Arms (or steers) the sculpt brush for `tool` with a dab-sized strength. */
function arm(
  app: AppState,
  mesh: MeshHandle,
  tool: SculptTool,
  radius: number,
  strength: number,
  falloff: "linear" | "smooth",
): void {
  app.setSculpt({
    mesh,
    op: TOOL_OP[tool],
    radius,
    strength: tool === "deflate" ? -Math.abs(strength) : Math.abs(strength),
    falloff,
    cycles: RELAX_PASSES,
    direction: [0, 0, 0],
  });
}
