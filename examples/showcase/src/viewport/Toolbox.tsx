// The sculpt toolbox: a floating strip over the viewport for the selected
// mesh. Four tools arm the sculpt brush (inflate, deflate, smooth, grab),
// with the brush radius and per-tool strength on sliders, the falloff, and
// the stroke-end and live remeshes toggles; the viewport paints the strokes.
// The mask group paints the protected region the brushes skip (invert and
// clear included); mask painting disarms the sculpt brush and vice versa.
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useState } from "react";
import { type AppState, useApp } from "../scene/app-state";
import { inspect } from "../panels/inspect";
import {
  DEFAULT_STRENGTHS,
  REMESH_TRI_LIMIT,
  checkMask,
  invertMaskIds,
  remeshTooBig,
  sculptBrush,
  type SculptTool,
} from "./sculpt";

export type { SculptTool };

const TOOLS: { tool: SculptTool; label: string; title: string }[] = [
  { tool: "inflate", label: "Inflate", title: "Puff the surface up along its normals" },
  { tool: "deflate", label: "Deflate", title: "Shrink the surface back along its normals" },
  { tool: "smooth", label: "Smooth", title: "Relax the surface toward its neighbours" },
  { tool: "grab", label: "Grab", title: "Drag the surface along with the pointer" },
];

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
              // One pointer owner: sculpting disarms the mask paint.
              app.setMaskPaint(false);
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
          data-action="sculpt-radius"
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
          data-action="sculpt-strength"
          min={0}
          max={1}
          step={0.01}
          value={strength}
          onChange={(e) => setStrengths({ ...strengths, [tool]: Number(e.currentTarget.value) })}
        />
      </label>
      <label className="field" title="How the strength falls off with distance">
        falloff
        <select
          data-action="sculpt-falloff"
          value={falloff}
          onChange={(e) => setFalloff(e.currentTarget.value as "linear" | "smooth")}
        >
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
      <MaskTools app={app} selection={selection} radius={item.radius} />
    </div>
  );
}

/** Arms (or steers) the sculpt brush for `tool` (see `sculptBrush`). */
function arm(
  app: AppState,
  mesh: MeshHandle,
  tool: SculptTool,
  radius: number,
  strength: number,
  falloff: "linear" | "smooth",
): void {
  app.setSculpt(sculptBrush(tool, mesh, radius, strength, falloff));
}

/**
 * The mask group: paint the protected region the deform brushes skip, with
 * invert and clear. Paint accumulates sphere-brush picks on the selected
 * mesh (a new mesh replaces the mask); invert and clear work in place over
 * the mirrored views, failing loudly when a topology edit took a masked
 * vertex. The counts come from the app; the notice is the loud half of the
 * mask contract (see `viewport/sculpt.ts`).
 */
function MaskTools(props: { app: AppState; selection: MeshHandle; radius: number }) {
  const { app, selection, radius } = props;
  const store = useScene();
  const mask = useApp(app, (s) => s.sculptMask);
  const painting = useApp(app, (s) => s.maskPaint);
  const maskRadius = useApp(app, (s) => s.maskRadius);
  const notice = useApp(app, (s) => s.sculptMaskNotice);
  // The paint radius follows the mesh, like the sculpt radius.
  useEffect(() => {
    app.setMaskRadius(radius / 4);
  }, [app, radius]);
  const sameMesh = mask !== null && mask.mesh === selection;

  /** Clears a mask its mesh outlived: loud, with the notice, never silent. */
  const staleClear = () => {
    app.setSculptMask(null);
    app.setSculptMaskNotice("The mask went stale after a topology edit and was cleared — repaint it.");
    inspect.sculpt.maskStale++;
  };

  const onInvert = () => {
    if (mask === null || !sameMesh) {
      return;
    }
    try {
      const v = store.scene.views(selection);
      const slots = v.positions().length / 3;
      checkMask(mask, selection, v.indices(), slots);
      app.setSculptMask({ mesh: selection, vertices: invertMaskIds(mask.vertices, v.indices(), slots), slots });
    } catch {
      // Removed meanwhile, or stale: a stale mask never silently reinterprets.
      staleClear();
    }
  };

  return (
    <>
      <div className="group" aria-label="Mask tools">
        <button
          type="button"
          data-action="sculpt-mask-paint"
          aria-pressed={painting}
          title="Paint the protected region the brushes skip (disarms the sculpt brush)"
          onClick={() => {
            const on = !painting;
            if (on) {
              // One pointer owner: mask painting disarms the sculpt brush.
              app.setSculpt(null);
            }
            app.setMaskPaint(on);
          }}
        >
          Mask
        </button>
        <button
          type="button"
          data-action="sculpt-mask-invert"
          title="Protect everything outside the mask instead"
          disabled={!sameMesh}
          onClick={onInvert}
        >
          Invert
        </button>
        <button
          type="button"
          data-action="sculpt-mask-clear"
          title="Drop the protected region"
          disabled={mask === null}
          onClick={() => app.setSculptMask(null)}
        >
          Clear
        </button>
        <span title={sameMesh ? `${mask.vertices.length} protected vertices` : undefined}>
          {sameMesh ? `${mask.vertices.length} masked` : mask === null ? "no mask" : "masked elsewhere"}
        </span>
      </div>
      <label className="slider" title="Mask paint radius in mesh units">
        mask radius <output>{fmt(maskRadius)}</output>
        <input
          type="range"
          data-action="sculpt-mask-radius"
          min={0.01}
          max={Math.max(radius, 0.02)}
          step={0.01}
          value={Math.min(maskRadius, Math.max(radius, 0.02))}
          onChange={(e) => app.setMaskRadius(Number(e.currentTarget.value))}
        />
      </label>
      {notice !== null ? (
        <span role="status" title="The mask contract's loud half">
          {notice}
        </span>
      ) : null}
    </>
  );
}
