// The wrap section's behaviour (#263): its settings, the grid the engine
// plans for them (replanned once the settings, the selection or the selected
// meshes rest), and the run as one new mesh, with a one-line report. The
// engine calls are `engine/workbench-ops.ts`'s `planWrap` and `runWrap`: on
// the Worker when the scene is there, so the page stays live.
import type { WrapPlan } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { planWrap, runWrap } from "../../../engine/workbench-ops";
import { hullSources } from "../../mesh-edit/ux/use-convex-hull";
import {
  DEFAULT_WRAP_SETTINGS,
  defaultWrapGap,
  describeWrap,
  detailVoxelSize,
  worldExtent,
  wrapMeshName,
  wrapOptions,
  type WrapSettings,
} from "./wrap";

/** How long the settings and meshes must rest before the grid is planned again, in ms. */
export const WRAP_PLAN_DELAY_MS = 150;

export interface WrapReport {
  text: string;
  error: boolean;
}

export function useWrap(args: { app: AppState; store: SceneStore; items: readonly Item[]; activeItem: Item | null }) {
  const { app, store, items, activeItem } = args;
  const multi = useApp(app, (s) => s.multi);
  // The multi-selection while it starts at the active mesh, else the active mesh.
  const sources = hullSources(items, multi, activeItem);
  const key = sources.map((i) => i.id).join(",");
  const [settings, setSettings] = useState<WrapSettings>(DEFAULT_WRAP_SETTINGS);
  const [extent, setExtent] = useState(0);
  const [plan, setPlan] = useState<{ plan: WrapPlan | null; error: string | null }>({ plan: null, error: null });
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<WrapReport | null>(null);
  const [revision, setRevision] = useState(0);
  const planned = useRef(0);

  // The meshes a wrap would read, for the change listener.
  const meshes = useRef<ReadonlySet<unknown>>(new Set());
  meshes.current = new Set(sources.map((i) => i.mesh));
  useEffect(() => {
    // Once the meshes rest: a drag or a stroke replans at its end.
    let timer: ReturnType<typeof setTimeout> | undefined;
    const off = store.scene.onChanges((changes) => {
      if (changes.some((c) => meshes.current.has(c.mesh))) {
        clearTimeout(timer);
        timer = setTimeout(() => setRevision((r) => r + 1), WRAP_PLAN_DELAY_MS);
      }
    });
    return () => {
      off();
      clearTimeout(timer);
    };
  }, [store]);

  const sourcesRef = useRef(sources);
  sourcesRef.current = sources;
  useEffect(() => {
    const n = ++planned.current;
    const targets = sourcesRef.current;
    if (targets.length === 0) {
      setExtent(0);
      setPlan({ plan: null, error: null });
      return;
    }
    const timer = setTimeout(() => {
      const size = worldExtent(store.scene, targets);
      setExtent(size);
      planWrap({ store, sources: targets, options: wrapOptions(settings, defaultWrapGap(size)) }).then(
        (p) => {
          if (n === planned.current) setPlan({ plan: p, error: null });
        },
        (e: unknown) => {
          if (n === planned.current) setPlan({ plan: null, error: e instanceof Error ? e.message : String(e) });
        },
      );
    }, WRAP_PLAN_DELAY_MS);
    return () => clearTimeout(timer);
  }, [store, key, settings, revision]);

  const update = (patch: Partial<WrapSettings>) => setSettings((s) => ({ ...s, ...patch }));
  const autoGap = defaultWrapGap(extent);

  const run = async () => {
    const targets = sourcesRef.current;
    if (busy || targets.length === 0) {
      return;
    }
    setBusy(true);
    setReport(null);
    try {
      const gap = defaultWrapGap(worldExtent(store.scene, targets));
      const r = await runWrap({
        store,
        app,
        name: wrapMeshName(targets),
        sources: targets,
        options: wrapOptions(settings, gap),
      });
      setReport({ text: describeWrap({ ...r, name: r.item.name }), error: false });
    } catch (e) {
      setReport({ text: `Wrap failed: ${e instanceof Error ? e.message : String(e)}`, error: true });
    } finally {
      setBusy(false);
    }
  };

  return {
    sources,
    settings,
    update,
    /** The gap in mm: typed, or the selection's default. */
    gap: settings.gap ?? autoGap,
    gapIsAuto: settings.gap === null,
    /** The voxel size in mm: typed, or what the detail gives. */
    voxelSize: settings.voxelSize ?? detailVoxelSize(extent, settings.detail),
    plan: plan.plan,
    planError: plan.error,
    /** Nothing to wrap, a plan that failed, or a grid over the limit. */
    blocked: sources.length === 0 || plan.error !== null || (plan.plan !== null && !plan.plan.fits),
    busy,
    report,
    run,
  };
}
