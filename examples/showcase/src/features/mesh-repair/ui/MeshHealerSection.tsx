/**
 * Extended Mesh Healer controls (#235):
 * - Extended 3D visual defect overlays (self-intersections, inverted, slivers)
 * - Click-to-focus defect hotspots
 * - Selective per-category repair checkboxes + [data-repair-op="selectiveHeal"]
 * - Before/after delta readout ([data-healer-delta])
 */
import type { Analysis, MeshHandle, RepairOp, SelfIntersectionMode } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useMemo, useState } from "react";
import type { AppState, HealerOverlayKind, RepairOverlayKind } from "../../../app/state/app-state";
import { useApp } from "../../../app/state/app-state";
import {
  countHealerOverlays,
  type DefectHotspot,
  extractDefectHotspots,
  HEALER_OVERLAY_COLORS,
  HEALER_OVERLAY_KINDS,
  type HealerOverlayCounts,
} from "./healer-overlays";
import { REPAIR_OVERLAY_COLORS } from "./repair-overlays";
import { plural, type AnalysisSummary } from "../ux/repair-summary";

export type SelectiveHealCategory =
  | "weld"
  | "removeDegenerate"
  | "orient"
  | "makeManifold"
  | "removeSmallComponents"
  | "fillHoles"
  | "cleanupSlivers"
  | "resolveSelfIntersections";

export const SELECTIVE_HEAL_CATEGORIES: readonly { key: SelectiveHealCategory; label: string }[] = [
  { key: "weld", label: "Weld coincident vertices" },
  { key: "removeDegenerate", label: "Degenerate & duplicate tris" },
  { key: "orient", label: "Orient normals & shells" },
  { key: "makeManifold", label: "Make manifold (edges/pinch)" },
  { key: "removeSmallComponents", label: "Prune small islands" },
  { key: "fillHoles", label: "Fill holes" },
  { key: "cleanupSlivers", label: "Clean sliver faces" },
  { key: "resolveSelfIntersections", label: "Resolve self-intersections" },
];

export interface SelectiveHealSettings {
  weldTolerance: number;
  minTriangles: number;
  maxEdges: number;
  fair: boolean;
  anySize: number;
  selfIntersectionMode: SelfIntersectionMode;
  siVoxelSize: string;
}

export function buildSelectiveHealOps(
  enabled: Record<SelectiveHealCategory, boolean>,
  settings: SelectiveHealSettings,
): RepairOp[] {
  const ops: RepairOp[] = [];
  if (enabled.weld) {
    ops.push({ op: "weld", tolerance: settings.weldTolerance });
  }
  if (enabled.removeDegenerate) {
    ops.push({ op: "removeDegenerate" });
  }
  if (enabled.orient) {
    ops.push({ op: "orient" });
  }
  if (enabled.makeManifold) {
    ops.push({ op: "makeManifold" });
  }
  if (enabled.removeSmallComponents) {
    ops.push({ op: "removeSmallComponents", minTriangles: settings.minTriangles });
  }
  if (enabled.fillHoles) {
    ops.push({
      op: "fillHoles",
      maxEdges: settings.maxEdges > 0 ? settings.maxEdges : settings.anySize,
      fair: settings.fair,
    });
  }
  if (enabled.cleanupSlivers) {
    ops.push({ op: "cleanupSlivers", maxAspectRatio: 20 });
  }
  if (enabled.resolveSelfIntersections) {
    const mode = settings.selfIntersectionMode === "none" ? "deleteAndFill" : settings.selfIntersectionMode;
    ops.push({
      op: "resolveSelfIntersections",
      mode,
      fair: settings.fair,
      ...(settings.siVoxelSize !== "" && Number(settings.siVoxelSize) > 0
        ? { voxelSize: Number(settings.siVoxelSize) }
        : {}),
    });
  }
  return ops;
}

export function formatHealerDelta(
  before: AnalysisSummary,
  after: AnalysisSummary,
  beforeExtra?: Partial<HealerOverlayCounts> | null,
  afterExtra?: Partial<HealerOverlayCounts> | null,
): string {
  const parts: string[] = [
    `Holes: ${before.holes.length} → ${after.holes.length}`,
    `Non-manifold: ${before.nonManifoldEdges} → ${after.nonManifoldEdges}`,
    `Orientation: ${before.orientationConflicts} → ${after.orientationConflicts}`,
    `Degenerate: ${before.degenerate + before.duplicate} → ${after.degenerate + after.duplicate}`,
    `Components: ${before.components.length} → ${after.components.length}`,
  ];
  if (beforeExtra?.slivers !== undefined && afterExtra?.slivers !== undefined) {
    parts.push(`Slivers: ${beforeExtra.slivers} → ${afterExtra.slivers}`);
  }
  if (beforeExtra?.selfIntersections !== undefined && afterExtra?.selfIntersections !== undefined) {
    parts.push(`Self-intersect tris: ${beforeExtra.selfIntersections} → ${afterExtra.selfIntersections}`);
  }
  return parts.join(" · ");
}

const HEALER_LABELS: Record<HealerOverlayKind, string> = {
  selfIntersections: "Self-intersections",
  inverted: "Inverted / orientation",
  slivers: "Sliver faces",
};

function swatchHex(kind: RepairOverlayKind | HealerOverlayKind): string {
  const hex =
    kind in HEALER_OVERLAY_COLORS
      ? HEALER_OVERLAY_COLORS[kind as HealerOverlayKind]
      : REPAIR_OVERLAY_COLORS[kind as RepairOverlayKind];
  return `#${hex.toString(16).padStart(6, "0")}`;
}

export function MeshHealerSection(props: {
  app: AppState;
  scene: { views: (mesh: MeshHandle) => { positions(): Float32Array; indices(): Uint32Array } } | WorkerScene;
  mesh: MeshHandle;
  rawAnalysis: Analysis | null;
  summary: AnalysisSummary;
  disabled: boolean;
  deltaLine: string | null;
  onRunSelective: (enabled: Record<SelectiveHealCategory, boolean>) => Promise<void>;
}) {
  const { app, scene, mesh, rawAnalysis, disabled, deltaLine, onRunSelective } = props;
  const healerKinds = useApp(app, (s) => s.healerOverlays);

  const [selectedCats, setSelectedCats] = useState<Record<SelectiveHealCategory, boolean>>({
    weld: true,
    removeDegenerate: true,
    orient: true,
    makeManifold: true,
    removeSmallComponents: true,
    fillHoles: true,
    cleanupSlivers: true,
    resolveSelfIntersections: false,
  });

  const { hotspots, extraCounts } = useMemo(() => {
    if (rawAnalysis === null) {
      return {
        hotspots: [] as DefectHotspot[],
        extraCounts: { selfIntersections: 0, inverted: 0, slivers: 0 } satisfies HealerOverlayCounts,
      };
    }
    try {
      const views = scene.views(mesh);
      const pos = views.positions();
      const idx = views.indices();
      return {
        hotspots: extractDefectHotspots(rawAnalysis, pos, idx),
        extraCounts: countHealerOverlays(rawAnalysis, pos, idx),
      };
    } catch {
      return {
        hotspots: [] as DefectHotspot[],
        extraCounts: { selfIntersections: 0, inverted: 0, slivers: 0 } satisfies HealerOverlayCounts,
      };
    }
  }, [rawAnalysis, scene, mesh]);

  const anySelected = Object.values(selectedCats).some(Boolean);

  const focusSpot = (spot: DefectHotspot) => {
    if (spot.kind in HEALER_OVERLAY_COLORS) {
      app.setHealerOverlay(spot.kind as HealerOverlayKind, true);
    } else {
      app.setRepairOverlay(spot.kind as RepairOverlayKind, true);
    }
    app.focusDefect(mesh, spot.center, spot.radius);
  };

  return (
    <div className="healer-section" data-mesh-healer>
      <fieldset className="overlays healer-overlays">
        <legend className="muted small">Healer 3D overlays</legend>
        {HEALER_OVERLAY_KINDS.map((kind) => (
          <label
            key={kind}
            className="check"
            title={`${HEALER_LABELS[kind]}: ${extraCounts[kind] === 0 ? "none" : plural(extraCounts[kind], "issue")}`}
          >
            <input
              type="checkbox"
              data-healer-overlay={kind}
              checked={healerKinds[kind]}
              onChange={(e) => app.setHealerOverlay(kind, e.currentTarget.checked)}
            />
            <span
              aria-hidden="true"
              style={{
                display: "inline-block",
                width: "0.7em",
                height: "0.7em",
                borderRadius: "2px",
                background: swatchHex(kind),
              }}
            />{" "}
            {HEALER_LABELS[kind]} ({extraCounts[kind]})
          </label>
        ))}
      </fieldset>

      {hotspots.length > 0 ? (
        <div className="healer-hotspots-wrap">
          <p className="muted small">Click a defect to focus camera:</p>
          <ul className="healer-hotspots">
            {hotspots.map((spot) => (
              <li key={spot.id}>
                <button
                  type="button"
                  className="healer-hotspot-btn"
                  data-healer-focus={spot.id}
                  onClick={() => focusSpot(spot)}
                  title={`Focus camera on ${spot.label}`}
                >
                  <span
                    className="swatch"
                    aria-hidden="true"
                    style={{ background: swatchHex(spot.kind) }}
                  />
                  <span>{spot.label}</span>
                  <span className="muted">Focus</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <fieldset className="overlays healer-selective">
        <legend className="muted small">Selective heal categories</legend>
        {SELECTIVE_HEAL_CATEGORIES.map(({ key, label }) => (
          <label key={key} className="check">
            <input
              type="checkbox"
              data-healer-category={key}
              checked={selectedCats[key]}
              onChange={(e) => {
                const on = e.currentTarget.checked;
                setSelectedCats((prev) => ({ ...prev, [key]: on }));
              }}
            />
            {label}
          </label>
        ))}
        <button
          type="button"
          data-repair-op="selectiveHeal"
          disabled={disabled || !anySelected}
          onClick={() => void onRunSelective(selectedCats)}
          title="Run only checked healing stages in one undo step"
        >
          Heal selected
        </button>
      </fieldset>

      {deltaLine !== null ? (
        <p className="healer-delta small" data-healer-delta role="status">
          <strong>Delta:</strong> {deltaLine}
        </p>
      ) : null}
    </div>
  );
}
