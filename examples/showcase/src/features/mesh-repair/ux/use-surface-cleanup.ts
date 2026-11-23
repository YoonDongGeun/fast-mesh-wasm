// The surface cleanup section's behaviour (#264): the denoise and Delaunay
// flip fields, the face selection they can be limited to, and the two runs
// through `engine/workbench-ops.ts`, each one undo step, with the
// before/after readouts of the last one. The panel's runner shows the report
// lines and keeps to one op at a time. A later change of the mesh that is
// not the run's own (an undo, another edit) clears the readouts.
import type { DelaunayFlipReport, DenoiseReport, MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useMemo, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { runDelaunayFlips, runDenoise } from "../../../engine/workbench-ops";
import type { LengthUnit } from "../../../shared/ux/units";
import {
  cleanupRegion,
  DEFAULT_DENOISE_ITERATIONS,
  DEFAULT_DENOISE_STRENGTH,
  DEFAULT_DIHEDRAL_LIMIT_DEG,
  DEFAULT_FEATURE_ANGLE_DEG,
  describeDelaunayFlips,
  describeDenoise,
  denoiseIterationsInput,
  denoiseStrengthInput,
  dihedralLimitInput,
  featureAngleInput,
  largestScale,
  type SurfaceReadout,
  surfaceReadouts,
} from "./surface-cleanup";
import type { LiveRepairAnalysis } from "./use-repair-analysis";
import type { RepairRunner } from "./use-repair-runner";

export function useSurfaceCleanup(args: {
  app: AppState;
  store: SceneStore;
  item: Item | null;
  units: LengthUnit;
  runner: RepairRunner;
  live: LiveRepairAnalysis;
}) {
  const { app, store, item, units, runner, live } = args;
  const mesh = item?.mesh ?? null;
  const [iterations, setIterations] = useState(DEFAULT_DENOISE_ITERATIONS);
  const [strength, setStrength] = useState(DEFAULT_DENOISE_STRENGTH);
  const [featureAngle, setFeatureAngle] = useState(DEFAULT_FEATURE_ANGLE_DEG);
  const [dihedralLimit, setDihedralLimit] = useState(DEFAULT_DIHEDRAL_LIMIT_DEG);
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [readouts, setReadouts] = useState<{ mesh: MeshHandle; label: string; rows: SurfaceReadout[] } | null>(null);
  /** The mesh of the last run while its readouts stand, and whether its own change is still to come. */
  const last = useRef<{ mesh: MeshHandle; own: boolean } | null>(null);

  const selected = useApp(app, (s) => (item === null ? undefined : s.wave6.select.selectedByMesh[item.id]));
  const selectedCount = useMemo(() => {
    if (mesh === null || selected === undefined || selected.length === 0) {
      return 0;
    }
    try {
      return cleanupRegion(store.scene.views(mesh).indices(), selected, true)?.length ?? 0;
    } catch {
      return 0; // Removed meanwhile: the selection follows.
    }
  }, [store, mesh, selected]);

  const { noteTriangleChange } = runner;
  useEffect(() => {
    if (mesh === null) {
      return;
    }
    return store.scene.onChanges((changes) => {
      for (const c of changes) {
        const run = last.current;
        if (run === null || c.mesh !== mesh || run.mesh !== mesh || c.removed) {
          continue;
        }
        const rewired = c.indices === "all" || c.indices.length > 0;
        const moved = c.positions === "all" || c.positions.length > 0;
        if (!rewired && !moved) {
          continue;
        }
        // The panel watches triangle changes only; denoise moves vertices.
        if (!rewired) {
          noteTriangleChange();
        }
        if (run.own) {
          run.own = false;
        } else {
          last.current = null;
          setReadouts(null);
        }
      }
    });
  }, [store, mesh, noteTriangleChange]);

  /** Runs `op` on the mesh (or its selected faces) and shows its line and readouts. */
  const run = <R extends Parameters<typeof surfaceReadouts>[0]>(
    label: string,
    changed: (r: R) => boolean,
    op: (target: MeshHandle, region: Uint32Array | undefined) => Promise<R>,
    describe: (r: R, target: MeshHandle) => string,
  ) =>
    runner.runGuarded(
      label,
      async (target) => {
        const region = cleanupRegion(store.scene.views(target).indices(), selected, selectedOnly);
        const watched = { mesh: target, own: true };
        last.current = watched;
        setReadouts(null);
        try {
          const r = await op(target, region);
          if (!changed(r)) {
            watched.own = false;
          }
          last.current = watched;
          setReadouts({ mesh: target, label, rows: surfaceReadouts(r) });
          runner.showReport(target, describe(r, target), false);
        } catch (e) {
          last.current = null;
          throw e;
        }
        void live.analyze(target);
      },
      (message) => `${label} failed: ${message}`,
    );

  return {
    iterations,
    setIterationsText: (text: string) => setIterations(denoiseIterationsInput(text)),
    strength,
    setStrengthText: (text: string) => setStrength(denoiseStrengthInput(text)),
    featureAngle,
    setFeatureAngleText: (text: string) => setFeatureAngle(featureAngleInput(text)),
    dihedralLimit,
    setDihedralLimitText: (text: string) => setDihedralLimit(dihedralLimitInput(text)),
    selectedOnly,
    setSelectedOnly,
    selectedCount,
    /** Limited to the selected faces, and there are none. */
    blocked: selectedOnly && selectedCount === 0,
    readouts: readouts !== null && readouts.mesh === mesh ? readouts : null,
    denoise: () =>
      run<DenoiseReport>(
        "Denoise",
        (r) => r.moved > 0,
        (target, region) =>
          runDenoise({ store, mesh: target, options: { iterations, strength, featureAngle, region } }),
        (r, target) => describeDenoise(r, units, largestScale(store.scene.placement(target))),
      ),
    delaunayFlips: () =>
      run<DelaunayFlipReport>(
        "Delaunay flips",
        (r) => r.flips > 0,
        (target, region) => runDelaunayFlips({ store, mesh: target, options: { dihedralLimit, region } }),
        describeDelaunayFlips,
      ),
  };
}

export type SurfaceCleanup = ReturnType<typeof useSurfaceCleanup>;
