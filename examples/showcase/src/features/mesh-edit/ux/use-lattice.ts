// The lattice section's behaviour (#265): its settings, the grid the engine
// plans for them (replanned once the settings, the mesh or its selection
// rest), the preview on a coarse grid while it is on (rebuilt the same way,
// one at a time), and the fill as one undo step, with a one-line report. The
// engine calls are `engine/workbench-ops.ts`'s `planLattice`,
// `previewLattice` and `fillLattice`: on the Worker when the scene is there,
// so the page stays live. Nothing is planned or previewed while another
// tab's panels show (this one stays mounted): a plan reads the whole mesh.
import type { LatticePlan } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { fillLattice, planLattice, previewLattice } from "../../../engine/workbench-ops";
import type { LengthUnit } from "../../../shared/ux/units";
import { worldExtent } from "../../mesh-repair/ux/wrap";
import {
  DEFAULT_LATTICE_SETTINGS,
  describeLattice,
  describeLatticePreview,
  latticeLengths,
  latticeOptions,
  latticeProblem,
  type LatticeSettings,
} from "./lattice";
import { latticePreviewOf } from "./lattice-preview-state";

/** How long the settings and the mesh must rest before the grid is planned again, in ms. */
export const LATTICE_PLAN_DELAY_MS = 150;

export interface LatticeLine {
  text: string;
  error: boolean;
}

/** An engine error as a sentence, without the engine's own preamble. */
function message(e: unknown): string {
  const text = (e instanceof Error ? e.message : String(e)).replace(/^repair operation failed: (invalid input: )?/, "");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function useLattice(args: {
  app: AppState;
  store: SceneStore;
  /** The selected mesh, else the first one. */
  activeItem: Item | null;
  units: LengthUnit;
}) {
  const { app, store, activeItem, units } = args;
  const shown = latticePreviewOf(app);
  const open = useApp(app, (s) => s.activeTab === "edit");
  const [settings, setSettings] = useState<LatticeSettings>(DEFAULT_LATTICE_SETTINGS);
  const [extent, setExtent] = useState(0);
  const [plan, setPlan] = useState<{ plan: LatticePlan | null; error: LatticeLine | null }>({ plan: null, error: null });
  const [previewing, setPreviewing] = useState(false);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewLine, setPreviewLine] = useState<LatticeLine | null>(null);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<LatticeLine | null>(null);
  const [revision, setRevision] = useState(0);

  const itemRef = useRef(activeItem);
  itemRef.current = activeItem;
  useEffect(() => {
    // Once the mesh rests: a drag or a stroke replans at its end.
    let timer: ReturnType<typeof setTimeout> | undefined;
    const off = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === itemRef.current?.mesh)) {
        clearTimeout(timer);
        timer = setTimeout(() => setRevision((r) => r + 1), LATTICE_PLAN_DELAY_MS);
      }
    });
    return () => {
      off();
      clearTimeout(timer);
    };
  }, [store]);

  /** The options on `item` as it is now, or why there are none. */
  const optionsFor = (item: Item) => {
    const size = worldExtent(store.scene, [item]);
    const lengths = latticeLengths(settings, size);
    return { size, lengths, problem: latticeProblem(lengths), options: latticeOptions(settings, lengths) };
  };

  // The plan follows the settings and the mesh.
  const planned = useRef(0);
  const mesh = activeItem?.mesh ?? null;
  useEffect(() => {
    const n = ++planned.current;
    const item = itemRef.current;
    if (!open) {
      return;
    }
    if (item === null) {
      setExtent(0);
      setPlan({ plan: null, error: null });
      return;
    }
    const timer = setTimeout(() => {
      const { size, problem, options } = optionsFor(item);
      setExtent(size);
      if (problem !== null) {
        setPlan({ plan: null, error: size > 0 ? { text: problem, error: true } : null });
        return;
      }
      planLattice({ store, mesh: item.mesh, options }).then(
        (p) => {
          if (n === planned.current) setPlan({ plan: p, error: null });
        },
        (e: unknown) => {
          // A mesh no lattice fits in (an open scan, say) is not the form's fault.
          if (n === planned.current) setPlan({ plan: null, error: { text: message(e), error: false } });
        },
      );
    }, LATTICE_PLAN_DELAY_MS);
    return () => clearTimeout(timer);
  }, [store, open, mesh, settings, revision]);

  // The preview follows them too while it is on. One runs at a time; what
  // changed meanwhile starts the next, with the settings it finds then.
  const running = useRef(false);
  const again = useRef(false);
  const wanted = useRef(false);
  wanted.current = previewing && activeItem !== null;
  const build = async () => {
    if (running.current) {
      again.current = true;
      return;
    }
    running.current = true;
    setPreviewBusy(true);
    try {
      do {
        again.current = false;
        const item = itemRef.current;
        if (!wanted.current || item === null) {
          break;
        }
        const { lengths, problem, options } = buildOptions.current(item);
        try {
          if (problem !== null) {
            throw new Error(problem);
          }
          const walls = await previewLattice({ store, mesh: item.mesh, options });
          if (wanted.current && !again.current && itemRef.current?.mesh === item.mesh) {
            shown.show({ mesh: item.mesh, walls });
            const triangles = walls.indices.length / 3;
            setPreviewLine({ text: describeLatticePreview(walls.plan, triangles, lengths.wall, units), error: false });
          }
        } catch (e) {
          if (wanted.current && !again.current) {
            shown.show(null);
            setPreviewLine({ text: `Preview failed: ${message(e)}`, error: true });
          }
        }
      } while (again.current);
    } finally {
      running.current = false;
      setPreviewBusy(false);
    }
  };
  // The latest render's settings, for a run that starts after the next one.
  const buildOptions = useRef(optionsFor);
  buildOptions.current = optionsFor;
  const buildRef = useRef(build);
  buildRef.current = build;
  useEffect(() => {
    if (!open) {
      setPreviewing(false);
    }
  }, [open]);
  useEffect(() => {
    if (!previewing || mesh === null) {
      shown.show(null);
      setPreviewLine(null);
      return;
    }
    // Another mesh's walls go at once; this one's stay up meanwhile.
    if (shown.getSnapshot()?.mesh !== mesh) {
      shown.show(null);
    }
    const timer = setTimeout(() => void buildRef.current(), LATTICE_PLAN_DELAY_MS);
    return () => clearTimeout(timer);
  }, [shown, previewing, mesh, settings, revision]);
  // The panel goes: so does the preview.
  useEffect(
    () => () => {
      wanted.current = false;
      shown.show(null);
    },
    [shown],
  );

  const update = (patch: Partial<LatticeSettings>) => setSettings((s) => ({ ...s, ...patch }));

  const run = async () => {
    const item = activeItem;
    if (busy || item === null) {
      return;
    }
    const { problem, options } = optionsFor(item);
    if (problem !== null) {
      setReport({ text: problem, error: true });
      return;
    }
    setBusy(true);
    setReport(null);
    // The walls it previews are about to be real.
    setPreviewing(false);
    try {
      const r = await fillLattice({ store, mesh: item.mesh, options });
      setReport({ text: describeLattice(item.name, settings.kind, r, units), error: false });
    } catch (e) {
      setReport({ text: `Lattice failed: ${message(e)}`, error: true });
    } finally {
      setBusy(false);
    }
  };

  const lengths = latticeLengths(settings, extent);
  return {
    settings,
    update,
    /** The lengths in mm, typed or by default. */
    lengths,
    /** The voxel size in mm: typed, or the planned one; `0` before a plan. */
    voxel: settings.voxel ?? plan.plan?.voxelSize ?? 0,
    plan: plan.plan,
    planError: plan.error,
    previewing,
    setPreviewing,
    previewBusy,
    previewLine,
    /** Nothing to fill, a plan that failed, or a grid over the limit. */
    blocked: activeItem === null || plan.error !== null || (plan.plan !== null && !plan.plan.fits),
    busy,
    report,
    run,
  };
}
