/**
 * Extended Mesh Healer controls (#235):
 * - Extended 3D visual defect overlays (self-intersections, inverted, slivers)
 * - Click-to-focus defect hotspots
 * - Selective per-category repair checkboxes + [data-repair-op="selectiveHeal"]
 * - Before/after delta readout ([data-healer-delta])
 * What they do lives in `ux/mesh-healer.ts`.
 */
import type { Analysis, MeshHandle } from "fast-mesh-wasm";
import { useMemo, useState } from "react";
import type { AppState, HealerOverlayKind, RepairOverlayKind } from "../../../app/state/app-state";
import { useApp } from "../../../app/state/app-state";
import { HEALER_OVERLAY_KINDS } from "../ux/healer-defects";
import {
  DEFAULT_SELECTIVE_HEAL,
  focusDefectHotspot,
  type HealerMeshViews,
  readHealerDefects,
  SELECTIVE_HEAL_CATEGORIES,
  type SelectiveHealCategory,
} from "../ux/mesh-healer";
import { plural, type AnalysisSummary } from "../ux/repair-summary";
import { HEALER_OVERLAY_COLORS } from "./healer-overlays";
import { REPAIR_OVERLAY_COLORS } from "./repair-overlays";

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
  scene: HealerMeshViews;
  mesh: MeshHandle;
  rawAnalysis: Analysis | null;
  summary: AnalysisSummary;
  disabled: boolean;
  deltaLine: string | null;
  onRunSelective: (enabled: Record<SelectiveHealCategory, boolean>) => Promise<void>;
}) {
  const { app, scene, mesh, rawAnalysis, disabled, deltaLine, onRunSelective } = props;
  const healerKinds = useApp(app, (s) => s.healerOverlays);

  const [selectedCats, setSelectedCats] = useState<Record<SelectiveHealCategory, boolean>>(DEFAULT_SELECTIVE_HEAL);

  const { hotspots, extraCounts } = useMemo(
    () => readHealerDefects(rawAnalysis, scene, mesh),
    [rawAnalysis, scene, mesh],
  );

  const anySelected = Object.values(selectedCats).some(Boolean);

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
                  onClick={() => focusDefectHotspot(app, mesh, spot)}
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
