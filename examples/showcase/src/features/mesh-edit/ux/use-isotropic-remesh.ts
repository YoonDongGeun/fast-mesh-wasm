// The isotropic remesh section's behaviour (#254): the active mesh's edge
// statistics, kept current as it changes; the target edge length (its mean
// edge length until one is typed for it), passes, boundary and crease
// fields; and the run through `engine/workbench-ops.ts` with the triangle
// counts before and after.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState } from "react";
import type { Item } from "../../../app/state/app-state";
import { runIsotropicRemesh } from "../../../engine/workbench-ops";
import {
  creaseAngleInput,
  DEFAULT_CREASE_ANGLE_DEG,
  DEFAULT_REMESH_ITERATIONS,
  defaultTargetEdge,
  describeIsotropicRemesh,
  type EdgeStats,
  edgeStats,
  estimateRemeshTriangles,
  isotropicRemeshOptions,
  remeshBlocked,
  remeshIterationsInput,
} from "./isotropic-remesh";
import type { ModelingReport } from "./use-modeling-ops";

const NO_STATS: EdgeStats = { triangles: 0, meanEdge: 0, area: 0 };

/** How long the mesh must rest after a change before its statistics are recounted, in ms. */
export const STATS_DELAY_MS = 150;

export function useIsotropicRemesh(args: { store: SceneStore; activeItem: Item | null }) {
  const { store, activeItem } = args;
  const mesh = activeItem?.mesh ?? null;
  const [stats, setStats] = useState<EdgeStats>(NO_STATS);
  // A typed target holds for the mesh it was typed for; another mesh starts at its own mean.
  const [typed, setTyped] = useState<{ mesh: MeshHandle; mm: number } | null>(null);
  const [iterations, setIterations] = useState(DEFAULT_REMESH_ITERATIONS);
  const [lockBoundaries, setLockBoundaries] = useState(true);
  const [preserveCreases, setPreserveCreases] = useState(true);
  const [creaseAngle, setCreaseAngle] = useState(DEFAULT_CREASE_ANGLE_DEG);
  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<ModelingReport | null>(null);

  useEffect(() => {
    if (mesh === null) {
      setStats(NO_STATS);
      return;
    }
    const read = () => {
      try {
        const v = store.scene.views(mesh);
        setStats(edgeStats(v.positions(), v.indices()));
      } catch {
        setStats(NO_STATS);
      }
    };
    read();
    // Recounted once the mesh rests: each change restarts the wait, so a
    // stroke or a drag on a large mesh costs one pass at its end.
    let timer: ReturnType<typeof setTimeout> | null = null;
    const off = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === mesh)) {
        if (timer !== null) {
          clearTimeout(timer);
        }
        timer = setTimeout(() => {
          timer = null;
          read();
        }, STATS_DELAY_MS);
      }
    });
    return () => {
      off();
      if (timer !== null) {
        clearTimeout(timer);
      }
    };
  }, [store, mesh]);

  const target = typed !== null && typed.mesh === mesh ? typed.mm : defaultTargetEdge(stats.meanEdge);
  const estimate = estimateRemeshTriangles(stats.area, target);
  const blocked = mesh === null || stats.triangles === 0 || remeshBlocked(target);

  const run = async () => {
    if (activeItem === null || blocked || busy !== null) {
      return;
    }
    const label = "Isotropic remesh";
    setBusy(label);
    setReport(null);
    try {
      const r = await runIsotropicRemesh({
        store,
        mesh: activeItem.mesh,
        options: isotropicRemeshOptions({
          targetEdge: target,
          iterations,
          lockBoundaries,
          preserveCreases,
          creaseAngleDeg: creaseAngle,
        }),
      });
      setReport({ label, text: describeIsotropicRemesh(r) });
    } catch (e) {
      setReport({ label, text: e instanceof Error ? e.message : String(e), error: true });
    } finally {
      setBusy(null);
    }
  };

  return {
    stats,
    target,
    /** A typed target edge length, in mm (for the active mesh). */
    setTarget: (mm: number) => {
      if (mesh !== null) {
        setTyped({ mesh, mm });
      }
    },
    /** Back to the active mesh's mean edge length. */
    resetTargetToMean: () => setTyped(null),
    targetIsMean: typed === null || typed.mesh !== mesh,
    estimate,
    blocked,
    iterations,
    setIterationsText: (text: string) => setIterations(remeshIterationsInput(text)),
    lockBoundaries,
    setLockBoundaries,
    preserveCreases,
    setPreserveCreases,
    creaseAngle,
    setCreaseAngleText: (text: string) => setCreaseAngle(creaseAngleInput(text)),
    busy,
    report,
    run,
  };
}
