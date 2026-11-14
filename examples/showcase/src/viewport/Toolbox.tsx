// The sculpt toolbox: a floating strip over the viewport for the selected
// mesh. Twelve tools arm the sculpt brush, with per-brush algorithm pickers,
// composable surface feature filters, mirror symmetry toggles, brush radius
// and per-tool strength sliders, falloff, and stretch-split / stroke-end
// remesh toggles; the viewport paints the strokes.
// The mask group paints the protected region the brushes hold (invert and
// clear included); mask painting disarms the sculpt brush and vice versa.
import type { GrabMethod, InflateMethod, MeshHandle, SmoothMethod, TwistMethod } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useState } from "react";
import { inspect } from "../panels/inspect";
import type {
  SculptFilterKind,
  SculptFilterSettings,
  SculptMethods,
  SculptSymmetrySettings,
} from "../panels/settings";
import { formatLength } from "../panels/units";
import { type AppState, useApp } from "../scene/app-state";
import {
  DEFAULT_STRENGTHS,
  FILTER_KIND_OPTIONS,
  GRAB_METHOD_OPTIONS,
  INFLATE_METHOD_OPTIONS,
  REMESH_TRI_LIMIT,
  SMOOTH_METHOD_OPTIONS,
  TWIST_METHOD_OPTIONS,
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
  { tool: "flatten", label: "Flatten", title: "Project peaks and valleys toward the brush plane" },
  { tool: "scrape", label: "Scrape", title: "Shave convex peaks down to the brush plane" },
  { tool: "fill", label: "Fill", title: "Lift concave valleys up to the brush plane" },
  { tool: "crease", label: "Crease", title: "Carve a sharp groove while pinching vertices inward" },
  { tool: "pinch", label: "Pinch", title: "Draw vertices radially toward the brush center" },
  { tool: "clay", label: "Clay", title: "Build flat-topped round clay dabs" },
  { tool: "clayStrips", label: "Strips", title: "Lay oriented square clay strips along the stroke" },
  { tool: "twist", label: "Twist", title: "Swirl vertices around the brush normal" },
];

const fmt = (v: number): string => String(Number(v.toPrecision(3)));

export function SculptToolbox(props: { app: AppState; selection?: MeshHandle | null }) {
  const { app } = props;
  const [storeSelection] = useSelection();
  const selection = props.selection !== undefined ? props.selection : storeSelection;
  const items = useApp(app, (s) => s.items);
  const sculpt = useApp(app, (s) => s.sculpt);
  const remesh = useApp(app, (s) => s.sculptRemesh);
  const liveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const methods = useApp(app, (s) => s.sculptMethods);
  const grabPoisson = useApp(app, (s) => s.sculptGrabPoisson);
  const filter = useApp(app, (s) => s.sculptFilter);
  const symmetry = useApp(app, (s) => s.sculptSymmetry);
  const units = useApp(app, (s) => s.units);
  const item = items.find((i) => i.mesh === selection) ?? null;
  const [tool, setTool] = useState<SculptTool>(() => sculpt?.tool ?? "inflate");
  const [radius, setRadius] = useState(0.25);
  const [strengths, setStrengths] = useState<Record<SculptTool, number>>(DEFAULT_STRENGTHS);
  const [falloff, setFalloff] = useState<"linear" | "smooth">("smooth");
  const armed = sculpt !== null && sculpt.mesh === selection;
  const activeTool = armed && sculpt.tool !== undefined ? sculpt.tool : tool;

  useEffect(() => {
    if (armed && sculpt.tool !== undefined && sculpt.tool !== tool) {
      setTool(sculpt.tool);
    }
  }, [armed, sculpt?.tool, tool]);

  // The radius follows the mesh; the rest are preferences and persist.
  useEffect(() => {
    if (item !== null) {
      setRadius(item.radius / 4);
    }
  }, [item]);

  // While armed, the toolbox drives the sculpt brush live (sliders and pickers
  // stay adjustable mid-stroke).
  useEffect(() => {
    if (armed && selection !== null) {
      arm(app, selection, activeTool, radius, strengths[activeTool], falloff, {
        methods,
        grabPoisson,
        filter,
        symmetry,
      });
    }
  }, [app, armed, selection, activeTool, radius, strengths, falloff, methods, grabPoisson, filter, symmetry]);

  if (item === null || selection === null) {
    return null;
  }
  // The optional remesh stands down on huge meshes, where it would stall the worker.
  const tooBig = remeshTooBig(item.triangles);
  const remeshTitle = tooBig
    ? `Stands down past ${REMESH_TRI_LIMIT.toLocaleString("en")} triangles: a remesh would stall the worker`
    : null;
  const strength = strengths[activeTool];
  const methodUi = methodConfigFor(activeTool, methods, app);

  return (
    <div className="toolbox" role="toolbar" aria-label="Sculpt tools">
      <div className="group">
        {TOOLS.map((t) => (
          <button
            key={t.tool}
            type="button"
            data-action={`sculpt-tool-${t.tool}`}
            aria-pressed={armed && activeTool === t.tool}
            title={t.title}
            onClick={() => {
              if (armed && activeTool === t.tool) {
                app.setSculpt(null);
                return;
              }
              setTool(t.tool);
              // One pointer owner: sculpting disarms the mask paint.
              app.setMaskPaint(false);
              arm(app, selection, t.tool, radius, strengths[t.tool], falloff, {
                methods,
                grabPoisson,
                filter,
                symmetry,
              });
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      {methodUi !== null ? (
        <div className="toolbox-section">
          <label className="field" title="Brush kernel algorithm">
            method
            <select
              data-action="sculpt-method"
              value={methodUi.value}
              onChange={(e) => methodUi.onChange(e.currentTarget.value)}
            >
              {methodUi.options.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <p className="method-desc" data-role="sculpt-method-description">
            {methodUi.description}
          </p>
          {activeTool === "grab" && methods.grab === "kelvinlet" ? (
            <label className="slider" title="Elastic Poisson ratio (0 compressible .. 0.5 incompressible)">
              poisson <output>{fmt(grabPoisson)}</output>
              <input
                type="range"
                data-action="sculpt-grab-poisson"
                min={0}
                max={0.5}
                step={0.01}
                value={grabPoisson}
                onChange={(e) => app.setSculptGrabPoisson(Number(e.currentTarget.value))}
              />
            </label>
          ) : null}
        </div>
      ) : null}
      <label className="slider" title="Brush radius">
        radius <output>{formatLength(radius, units)}</output>
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
      <label
        className="slider"
        title={
          activeTool === "grab"
            ? "How closely the surface follows the pointer"
            : activeTool === "twist"
              ? "Twist angle per dab (radians)"
              : "Strength of one dab"
        }
      >
        strength <output>{fmt(strength)}</output>
        <input
          type="range"
          data-action="sculpt-strength"
          min={0}
          max={1}
          step={0.01}
          value={strength}
          onChange={(e) => setStrengths({ ...strengths, [activeTool]: Number(e.currentTarget.value) })}
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
      <div className="toolbox-section">
        <label className="field" title="Composable per-vertex surface feature filter">
          filter
          <select
            data-action="sculpt-filter"
            value={filter.kind}
            onChange={(e) => app.setSculptFilter({ kind: e.currentTarget.value as SculptFilterKind })}
          >
            {FILTER_KIND_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        {filter.kind === "peaksOnly" || filter.kind === "valleysOnly" ? (
          <>
            <label className="slider" title="Minimum prominence threshold before the brush engages">
              threshold <output>{fmt(filter.threshold)}</output>
              <input
                type="range"
                data-action="sculpt-filter-threshold"
                min={0}
                max={1}
                step={0.01}
                value={filter.threshold}
                onChange={(e) => app.setSculptFilter({ threshold: Number(e.currentTarget.value) })}
              />
            </label>
            <label className="slider" title="Sharpness of the filter ramp">
              contrast <output>{fmt(filter.contrast)}</output>
              <input
                type="range"
                data-action="sculpt-filter-contrast"
                min={0}
                max={1}
                step={0.01}
                value={filter.contrast}
                onChange={(e) => app.setSculptFilter({ contrast: Number(e.currentTarget.value) })}
              />
            </label>
          </>
        ) : null}
        {filter.kind === "normalAngle" ? (
          <label className="slider" title="Maximum surface normal cone half-angle (degrees)">
            angle <output>{fmt(filter.maxAngleDeg)}°</output>
            <input
              type="range"
              data-action="sculpt-filter-angle"
              min={5}
              max={180}
              step={1}
              value={filter.maxAngleDeg}
              onChange={(e) => app.setSculptFilter({ maxAngleDeg: Number(e.currentTarget.value) })}
            />
          </label>
        ) : null}
      </div>
      <div className="group-3" role="group" aria-label="Mirror symmetry">
        <button
          type="button"
          data-action="sculpt-sym-x"
          aria-pressed={symmetry.x}
          title="Mirror stroke across local X = 0 plane"
          onClick={() => app.setSculptSymmetry({ x: !symmetry.x })}
        >
          Sym X
        </button>
        <button
          type="button"
          data-action="sculpt-sym-y"
          aria-pressed={symmetry.y}
          title="Mirror stroke across local Y = 0 plane"
          onClick={() => app.setSculptSymmetry({ y: !symmetry.y })}
        >
          Sym Y
        </button>
        <button
          type="button"
          data-action="sculpt-sym-z"
          aria-pressed={symmetry.z}
          title="Mirror stroke across local Z = 0 plane"
          onClick={() => app.setSculptSymmetry({ z: !symmetry.z })}
        >
          Sym Z
        </button>
      </div>
      <label
        className="check"
        title={
          remeshTitle ??
          "Remesh the stroked region when the stroke ends. Rewrites everything in the region, including what the stroke did not touch"
        }
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
        title="Split triangles once the stroke stretches them past their limit. Everything else stays exactly as it is"
      >
        <input
          type="checkbox"
          data-action="sculpt-live-remesh"
          checked={liveRemesh}
          onChange={(e) => app.setSculptLiveRemesh(e.currentTarget.checked)}
        />
        split stretched triangles
      </label>
      <MaskTools app={app} selection={selection} radius={item.radius} />
    </div>
  );
}

function methodConfigFor(
  tool: SculptTool,
  methods: SculptMethods,
  app: AppState,
): {
  value: string;
  options: readonly { value: string; label: string; description: string }[];
  description: string;
  onChange: (next: string) => void;
} | null {
  if (tool === "inflate" || tool === "deflate") {
    const desc = INFLATE_METHOD_OPTIONS.find((o) => o.value === methods.inflate)?.description ?? "";
    return {
      value: methods.inflate,
      options: INFLATE_METHOD_OPTIONS,
      description: desc,
      onChange: (next) => app.setSculptMethods({ inflate: next as InflateMethod }),
    };
  }
  if (tool === "smooth") {
    const desc = SMOOTH_METHOD_OPTIONS.find((o) => o.value === methods.smooth)?.description ?? "";
    return {
      value: methods.smooth,
      options: SMOOTH_METHOD_OPTIONS,
      description: desc,
      onChange: (next) => app.setSculptMethods({ smooth: next as SmoothMethod }),
    };
  }
  if (tool === "grab") {
    const desc = GRAB_METHOD_OPTIONS.find((o) => o.value === methods.grab)?.description ?? "";
    return {
      value: methods.grab,
      options: GRAB_METHOD_OPTIONS,
      description: desc,
      onChange: (next) => app.setSculptMethods({ grab: next as GrabMethod }),
    };
  }
  if (tool === "twist") {
    const desc = TWIST_METHOD_OPTIONS.find((o) => o.value === methods.twist)?.description ?? "";
    return {
      value: methods.twist,
      options: TWIST_METHOD_OPTIONS,
      description: desc,
      onChange: (next) => app.setSculptMethods({ twist: next as TwistMethod }),
    };
  }
  return null;
}

/** Arms (or steers) the sculpt brush for `tool` (see `sculptBrush`). */
function arm(
  app: AppState,
  mesh: MeshHandle,
  tool: SculptTool,
  radius: number,
  strength: number,
  falloff: "linear" | "smooth",
  extras: {
    methods: SculptMethods;
    grabPoisson: number;
    filter: SculptFilterSettings;
    symmetry: SculptSymmetrySettings;
  },
): void {
  app.setSculpt(sculptBrush(tool, mesh, radius, strength, falloff, extras));
}

/**
 * The mask group: paint the protected region the deform brushes hold, with
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
  const shown = useApp(app, (s) => s.sculptMaskShown);
  const units = useApp(app, (s) => s.units);
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
        <button
          type="button"
          data-action="sculpt-mask-show"
          title="Show the mask's tint on the surface while a tool is armed"
          aria-pressed={shown}
          onClick={() => app.setSculptMaskShown(!shown)}
        >
          Show
        </button>
        <span title={sameMesh ? `${mask.vertices.length} protected vertices` : undefined}>
          {sameMesh ? `${mask.vertices.length} masked` : mask === null ? "no mask" : "masked elsewhere"}
        </span>
      </div>
      <label className="slider" title="Mask paint radius">
        mask radius <output>{formatLength(maskRadius, units)}</output>
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
