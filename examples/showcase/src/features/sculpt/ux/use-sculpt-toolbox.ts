// The sculpt toolbox's behaviour: the brush picker table, the mesh it targets,
// the brush state it keeps (tool, radius, per-tool strength, falloff) and arms
// the sculpt brush with live, the per-brush method pickers, the Dyntopo ROI
// remesh, and the mask group's paint toggle, invert and stale clear. One
// pointer owner: arming a brush disarms the mask paint and vice versa.
import type { GrabMethod, InflateMethod, MeshHandle, SmoothMethod, TwistMethod } from "fast-mesh-wasm";
import { useScene, useSelection, useStoreSelector } from "fast-mesh-wasm/react";
import { meshState } from "fast-mesh-wasm/store";
import { useEffect, useMemo, useRef, useState } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import type { SculptFilterSettings, SculptMethods, SculptSymmetrySettings } from "../../../shared/ux/settings";
import {
  applyDyntopoRoi,
  checkMask,
  DEFAULT_STRENGTHS,
  GRAB_METHOD_OPTIONS,
  INFLATE_METHOD_OPTIONS,
  invertMaskIds,
  sculptBrush,
  SMOOTH_METHOD_OPTIONS,
  type SculptTool,
  TWIST_METHOD_OPTIONS,
} from "./sculpt";
import { resolveTargetFromEligible } from "./sculpt-target";

/** The brush picker's buttons, in toolbox order. */
export const SCULPT_TOOLS: { tool: SculptTool; label: string; title: string }[] = [
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

/** A slider readout, to three significant digits. */
export const formatReadout = (v: number): string => String(Number(v.toPrecision(3)));

/** The method picker for `tool`'s brush kernel, or null when the brush has one kernel. */
export function methodConfigFor(
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
export function armSculptBrush(
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
 * The toolbox's state and actions. `selectionProp` overrides the store's
 * selection when given. `eligibleItems` is empty when nothing can be sculpted.
 */
export function useSculptToolbox(app: AppState, selectionProp: MeshHandle | null | undefined) {
  const store = useScene();
  const [storeSelection] = useSelection();
  const selection = selectionProp !== undefined ? selectionProp : storeSelection;
  const items = useApp(app, (s) => s.items);
  const sculpt = useApp(app, (s) => s.sculpt);
  const mask = useApp(app, (s) => s.sculptMask);
  const brushPick = useApp(app, (s) => s.brushPick);
  const pinToActive = useApp(app, (s) => s.sculptPinToActive);
  const remesh = useApp(app, (s) => s.sculptRemesh);
  const liveRemesh = useApp(app, (s) => s.sculptLiveRemesh);
  const dyntopo = useApp(app, (s) => s.sculptDyntopo);
  const dyntopoMode = useApp(app, (s) => s.sculptDyntopoMode);
  const dyntopoDetail = useApp(app, (s) => s.sculptDyntopoDetail);
  const methods = useApp(app, (s) => s.sculptMethods);
  const grabPoisson = useApp(app, (s) => s.sculptGrabPoisson);
  const filter = useApp(app, (s) => s.sculptFilter);
  const symmetry = useApp(app, (s) => s.sculptSymmetry);
  const units = useApp(app, (s) => s.units);
  const visMask = useStoreSelector((s) =>
    items.map((i) => (!i.locked && meshState(s, i.mesh).desired.visible ? "1" : "0")).join(""),
  );
  const eligibleItems = useMemo(
    () => items.filter((_, idx) => visMask[idx] === "1"),
    [items, visMask],
  );
  const preferredCurrent = sculpt?.mesh ?? mask?.mesh ?? null;
  const targetItem = useMemo(
    () => resolveTargetFromEligible(eligibleItems, selection, preferredCurrent, pinToActive),
    [eligibleItems, selection, preferredCurrent, pinToActive],
  );
  const targetMesh = targetItem?.mesh ?? null;
  const [tool, setTool] = useState<SculptTool>(() => sculpt?.tool ?? "inflate");
  const [radius, setRadius] = useState(0.25);
  const [strengths, setStrengths] = useState<Record<SculptTool, number>>(DEFAULT_STRENGTHS);
  const [falloff, setFalloff] = useState<"linear" | "smooth">("smooth");
  const armed = sculpt !== null && targetMesh !== null && sculpt.mesh === targetMesh;
  const activeTool = armed && sculpt.tool !== undefined ? sculpt.tool : tool;
  const lastTargetMeshRef = useRef<MeshHandle | null>(null);

  useEffect(() => {
    if (armed && sculpt.tool !== undefined && sculpt.tool !== tool) {
      setTool(sculpt.tool);
    }
  }, [armed, sculpt?.tool, tool]);

  // If all eligible meshes disappear (hidden/locked/removed), disarm sculpting.
  useEffect(() => {
    if (targetMesh === null && sculpt !== null) {
      app.setSculpt(null);
    }
  }, [app, targetMesh, sculpt]);

  // The radius follows the target mesh (adopting hover-scaled radius if already armed there).
  useEffect(() => {
    if (targetItem === null) {
      lastTargetMeshRef.current = null;
      return;
    }
    if (lastTargetMeshRef.current !== targetItem.mesh) {
      lastTargetMeshRef.current = targetItem.mesh;
      if (sculpt !== null && sculpt.mesh === targetItem.mesh) {
        setRadius(sculpt.radius);
        return;
      }
      setRadius(targetItem.radius / 4);
    }
  }, [targetItem?.mesh, targetItem?.radius, sculpt]);

  // While armed, the toolbox drives the sculpt brush live (sliders and pickers
  // stay adjustable mid-stroke).
  useEffect(() => {
    if (armed && targetMesh !== null) {
      armSculptBrush(app, targetMesh, activeTool, radius, strengths[activeTool], falloff, {
        methods,
        grabPoisson,
        filter,
        symmetry,
      });
    }
  }, [app, armed, targetMesh, activeTool, radius, strengths, falloff, methods, grabPoisson, filter, symmetry]);

  /** Arms `next` on the target, or disarms it when it is the armed tool. */
  const pickTool = (next: SculptTool) => {
    if (armed && activeTool === next) {
      app.setSculpt(null);
      return;
    }
    if (targetMesh === null) {
      return;
    }
    setTool(next);
    // One pointer owner: sculpting disarms the mask paint.
    app.setMaskPaint(false);
    armSculptBrush(app, targetMesh, next, radius, strengths[next], falloff, {
      methods,
      grabPoisson,
      filter,
      symmetry,
    });
  };

  /** Sets the active tool's strength. */
  const setStrength = (value: number) => setStrengths({ ...strengths, [activeTool]: value });

  /** Cleans and subdivides the Dyntopo ROI around the brush center, or the mesh center. */
  const remeshRoi = () => {
    if (targetItem === null) {
      return;
    }
    void applyDyntopoRoi({
      scene: store.scene,
      store,
      target: targetItem,
      brushPick,
      mask,
      mode: dyntopoMode,
      detail: dyntopoDetail,
      brushRadius: radius,
    });
  };

  return {
    eligibleItems,
    targetItem,
    targetMesh,
    armed,
    activeTool,
    radius,
    setRadius,
    strengths,
    setStrength,
    falloff,
    setFalloff,
    pinToActive,
    remesh,
    liveRemesh,
    dyntopo,
    dyntopoMode,
    dyntopoDetail,
    methods,
    grabPoisson,
    filter,
    symmetry,
    units,
    pickTool,
    remeshRoi,
  };
}

/**
 * The mask group's state and actions on `targetMesh`; the paint radius
 * follows `radius`, the target mesh's, like the sculpt radius.
 */
export function useMaskTools(app: AppState, targetMesh: MeshHandle | null, radius: number) {
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

  /** Clears a mask its mesh outlived: loud, with the notice, never silent. */
  const staleClear = () => {
    app.setSculptMask(null);
    app.setSculptMaskNotice("The mask went stale after a topology edit and was cleared — repaint it.");
    inspect.sculpt.maskStale++;
  };

  /** Protects everything outside the mask instead, in place over the mirrored views. */
  const onInvert = () => {
    if (mask === null || targetMesh === null || mask.mesh !== targetMesh) {
      return;
    }
    try {
      const v = store.scene.views(targetMesh);
      const slots = v.positions().length / 3;
      checkMask(mask, targetMesh, v.indices(), slots);
      app.setSculptMask({ mesh: targetMesh, vertices: invertMaskIds(mask.vertices, v.indices(), slots), slots });
    } catch {
      // Removed meanwhile, or stale: a stale mask never silently reinterprets.
      staleClear();
    }
  };

  /** Turns mask painting on or off on the target. */
  const togglePaint = () => {
    if (targetMesh === null) {
      return;
    }
    const on = !painting;
    if (on) {
      // One pointer owner: mask painting disarms the sculpt brush.
      app.setSculpt(null);
    }
    app.setMaskPaint(on);
  };

  return { mask, painting, maskRadius, notice, shown, units, onInvert, togglePaint };
}
