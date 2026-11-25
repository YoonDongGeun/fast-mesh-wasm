// The repair panel for the selected mesh: its analysis (holes and their
// sizes, orientation, components, non-manifold and degenerate elements),
// refreshed whenever its triangles change (a repair, its undo), the 3D
// overlays drawing those issues on the mesh, and the library's repair ops,
// each one undo step through `store.repair` (denoise and Delaunay flips
// through the engine bridge). The analysis refresh, the op
// parameters, the loop picks and the op sequences live in `ux/`.
import type { AutoRepairPreset, FillContinuity, Scene, SelfIntersectionMode } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { REPAIR_OVERLAY_KINDS, type AppState, type RepairOverlayKind, useApp } from "../../../app/state/app-state";
import { repairActions } from "../ux/repair-actions";
import {
  type AnalysisSummary,
  AUTO_REPAIR_PRESET_LABELS,
  autoRepairNote,
  overlayIssueCounts,
  plural,
  SELF_INTERSECTION_MODE_LABELS,
} from "../ux/repair-summary";
import { useRepairAnalysis } from "../ux/use-repair-analysis";
import { holeTargetsOf, useRepairLoopPicks } from "../ux/use-repair-loop-picks";
import { useRepairOptions, wholeAtLeast, wholeNumber } from "../ux/use-repair-options";
import { useRepairRunner } from "../ux/use-repair-runner";
import { useSurfaceCleanup } from "../ux/use-surface-cleanup";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { MeshHealerSection } from "./MeshHealerSection";
import { REPAIR_OVERLAY_COLORS } from "./repair-overlays";
import { SurfaceCleanupSection } from "./SurfaceCleanupSection";
import { WrapSection } from "./WrapSection";

/** The holes listed by size. */
const HOLES_SHOWN = 6;

export function RepairPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const item = items.find((i) => i.mesh === selection) ?? null;
  const mesh = item?.mesh ?? null;
  const units = useApp(app, (s) => s.units);
  const repairAnalysis = useApp(app, (snap) => snap.repairAnalysis);

  const options = useRepairOptions();
  const runner = useRepairRunner(mesh);
  const live = useRepairAnalysis({
    app,
    scene,
    store,
    mesh,
    weldTolerance: options.weldTolerance,
    onTriangleChange: runner.noteTriangleChange,
  });
  const picks = useRepairLoopPicks({ app, scene, mesh, items, analysis: live.analysis });
  const cleanup = useSurfaceCleanup({ app, store, item, units, runner, live });

  const s = live.analysis?.mesh === mesh ? live.analysis.summary : null;
  const rawLoops = repairAnalysis?.mesh === mesh ? repairAnalysis.analysis.loops : [];
  const holeTargets = holeTargetsOf(s, rawLoops);
  const actions = repairActions({
    app,
    scene,
    store,
    mesh,
    items,
    units,
    options,
    runner,
    live,
    picks,
    summary: s,
    holeTargets,
  });

  if (item === null) {
    return (
      <section className="panel" aria-label="Repair" data-tour="repair">
        <h2>Repair</h2>
        <p className="muted">Select a mesh to analyse and repair it.</p>
      </section>
    );
  }

  const disabled = runner.busy !== null;
  return (
    <section className="panel" aria-label="Repair" data-repair={item.name} data-tour="repair">
      <header className="panel-head">
        <h2>Repair</h2>
        <button
          type="button"
          data-repair-op="analyze"
          onClick={() => void live.analyze(item.mesh)}
          disabled={live.analyzing}
        >
          Analyze
        </button>
      </header>
      <p className="muted small mesh-title" title={item.name}>
        {item.name}
      </p>
      {s === null ? (
        <p className="muted small">{live.analyzing ? "Analysing…" : "No analysis."}</p>
      ) : (
        <>
          <AnalysisList summary={s} selfIntersects={live.selfIntersects} stale={live.analyzing} unit={units} />
          <label className="check" title="Include triangle self-intersection detection in live analysis">
            <input
              type="checkbox"
              data-check-self-intersections
              checked={live.checkSelfIntersections}
              onChange={(e) => live.toggleSelfIntersections(e.currentTarget.checked)}
            />
            Check self-intersections
          </label>
          <OverlayToggles app={app} summary={s} />
          {mesh !== null ? (
            <MeshHealerSection
              app={app}
              scene={scene}
              mesh={mesh}
              rawAnalysis={repairAnalysis?.mesh === mesh ? repairAnalysis.analysis : null}
              summary={s}
              disabled={disabled}
              deltaLine={runner.healerDelta?.mesh === mesh ? runner.healerDelta.text : null}
              onRunSelective={actions.selectiveHeal}
            />
          ) : null}
        </>
      )}

      <div className="ops">
        <div className="op">
          <button
            type="button"
            data-repair-op="weld"
            disabled={disabled}
            onClick={() => void actions.weld()}
          >
            Weld
          </button>
          <label className="field" title="Vertices at most this far apart are merged">
            within ({units})
            <LengthInput
              min={0}
              mm={options.weldTolerance}
              unit={units}
              onMm={(mm) => options.setWeldTolerance(Math.max(0, mm))}
            />
          </label>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="removeDegenerate"
            disabled={disabled}
            onClick={() => void actions.removeDegenerate()}
          >
            Remove degenerate
          </button>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="orient"
            disabled={disabled}
            onClick={() => void actions.orient()}
          >
            Orient
          </button>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="removeSmallComponents"
            disabled={disabled}
            onClick={() => void actions.removeSmallComponents()}
          >
            Remove small components
          </button>
          <label className="field" title="Components with fewer triangles are removed">
            below
            <input
              type="number"
              min={1}
              step={1}
              value={options.minTriangles}
              onChange={(e) => options.setMinTriangles(wholeAtLeast(e.currentTarget.value, 1))}
            />
            tris
          </label>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="fillHoles"
            disabled={disabled}
            onClick={() => void actions.fillHoles()}
          >
            Fill holes
          </button>
          <label
            className="field"
            title="Only holes with at most this many edges. Empty: any size, which also fills an open mesh's outer boundary"
          >
            max
            <input
              type="number"
              min={0}
              step={1}
              placeholder="any"
              value={options.maxEdges === 0 ? "" : options.maxEdges}
              onChange={(e) => options.setMaxEdges(wholeAtLeast(e.currentTarget.value, 0))}
            />
            edges
          </label>
          <label className="check" title="Refine and fair the patch so it follows the surrounding curvature">
            <input type="checkbox" checked={options.fair} onChange={(e) => options.setFair(e.currentTarget.checked)} />
            Fair (follow curvature)
          </label>
        </div>
        <div className="op" data-continuity-fill="true">
          <button
            type="button"
            data-repair-op="fillContinuity"
            disabled={disabled}
            title="Fill holes using selectable G0/G1/G2 boundary continuity (#210)"
            onClick={() => void actions.fillContinuity()}
          >
            Fill continuity
          </button>
          <label className="field" title="Boundary seam continuity across filled patch">
            continuity
            <select
              data-continuity-mode="true"
              value={options.continuity}
              onChange={(e) => options.setContinuity(e.currentTarget.value as FillContinuity)}
            >
              <option value="flat">Flat (G0)</option>
              <option value="tangent">Tangent (G1)</option>
              <option value="curvature">Curvature (G2)</option>
            </select>
          </label>
          <label className="field" title="Fill all eligible holes or one specific boundary loop">
            target
            <select
              data-continuity-target="true"
              value={picks.continuityLoop}
              onChange={(e) => picks.setContinuityLoop(e.currentTarget.value)}
            >
              <option value="all">All holes</option>
              {holeTargets.map((h) => (
                <option key={h.displayIndex} value={String(h.loopIndex)}>
                  #{h.displayIndex + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="check" title="Subdivide and smooth interior patch triangles">
            <input
              type="checkbox"
              data-continuity-refine="true"
              checked={options.refinePatch}
              onChange={(e) => options.setRefinePatch(e.currentTarget.checked)}
            />
            Refine Patch
          </label>
          <label className="check" title="Detect and merge enclosed nested island loops inside holes">
            <input
              type="checkbox"
              data-continuity-islands="true"
              checked={options.detectIslands}
              onChange={(e) => options.setDetectIslands(e.currentTarget.checked)}
            />
            Detect Islands
          </label>
        </div>
        <p className="muted small">Fill holes: an empty max fills any size, including an open mesh's outer boundary.</p>
        <div className="op">
          <button
            type="button"
            data-repair-op="makeManifold"
            disabled={disabled}
            onClick={() => void actions.makeManifold()}
          >
            Make manifold
          </button>
          <button
            type="button"
            data-repair-op="cleanupSlivers"
            disabled={disabled}
            onClick={() => void actions.cleanupSlivers()}
          >
            Cleanup slivers
          </button>
        </div>
        <div className="op">
          <button
            type="button"
            data-repair-op="resolveSelfIntersections"
            disabled={disabled}
            onClick={() => void actions.resolveSelfIntersections()}
          >
            Resolve self-intersections
          </button>
          <label className="field" title="Self-intersection resolution mode">
            mode
            <select
              data-si-mode
              value={options.healSiMode}
              onChange={(e) => options.setHealSiMode(e.currentTarget.value as SelfIntersectionMode)}
            >
              {(Object.keys(SELF_INTERSECTION_MODE_LABELS) as SelfIntersectionMode[]).map((mode) => (
                <option key={mode} value={mode}>
                  {SELF_INTERSECTION_MODE_LABELS[mode]}
                </option>
              ))}
            </select>
          </label>
          {options.healSiMode === "voxelHeal" ? (
            <label className="field" title="Voxel spacing (empty = automatic)">
              voxel ({units})
              <input
                type="number"
                min={0}
                step="any"
                placeholder="auto"
                data-si-voxel
                value={options.siVoxelSize}
                onChange={(e) => options.setSiVoxelSize(e.currentTarget.value)}
              />
            </label>
          ) : null}
        </div>
      </div>

      <SurfaceCleanupSection cleanup={cleanup} disabled={disabled} />

      <div className="ops">
        <div className="op">
          <label className="field" title="Domain repair preset selecting pipeline stages and tolerances">
            preset
            <select
              data-repair-preset
              value={options.preset}
              onChange={(e) => options.setPreset(e.currentTarget.value as AutoRepairPreset)}
            >
              {(Object.keys(AUTO_REPAIR_PRESET_LABELS) as AutoRepairPreset[]).map((p) => (
                <option key={p} value={p}>
                  {AUTO_REPAIR_PRESET_LABELS[p]}
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Override self-intersection healing stage for Auto repair">
            intersections
            <select
              data-repair-si-mode
              value={options.siOverride}
              onChange={(e) => options.setSiOverride(e.currentTarget.value as SelfIntersectionMode | "auto")}
            >
              <option value="auto">Preset default</option>
              {(Object.keys(SELF_INTERSECTION_MODE_LABELS) as SelfIntersectionMode[]).map((mode) => (
                <option key={mode} value={mode}>
                  {SELF_INTERSECTION_MODE_LABELS[mode]}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            data-repair-op="autoRepair"
            disabled={disabled}
            onClick={() => void actions.autoRepair()}
          >
            Auto repair
          </button>
          <p className="muted small">
            {autoRepairNote(options.preset, options.siOverride)}
          </p>
        </div>
      </div>

      <div className="ops">
        <div className="op">
          <span className="op-label">Bridge</span>
          <label className="field" title="First loop (holes of this mesh, largest first)">
            loop A
            <select
              data-bridge-loop="a"
              value={picks.bridgeA}
              onChange={(e) => picks.setBridgeA(e.currentTarget.value)}
            >
              <option value="">—</option>
              {(s?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Second loop">
            loop B
            <select
              data-bridge-loop="b"
              value={picks.bridgeB}
              onChange={(e) => picks.setBridgeB(e.currentTarget.value)}
            >
              <option value="">—</option>
              {(s?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            data-repair-op="bridgeLoops"
            disabled={disabled || picks.bridgeA === "" || picks.bridgeB === "" || picks.bridgeA === picks.bridgeB}
            title={s === null || s.holes.length < 2 ? "Needs two holes" : "Stitch the two loops with a tube"}
            onClick={() => actions.bridgeLoops()}
          >
            Bridge
          </button>
          <label className="field" title="Maximum vertices per loop">
            max
            <input
              type="number"
              min={0}
              step={1}
              value={options.loopMax}
              onChange={(e) => options.setLoopMax(wholeAtLeast(e.currentTarget.value, 0))}
            />
          </label>
        </div>
        <div className="op" data-partial-bridge="true">
          <span className="op-label">Arc bridge</span>
          <label className="field" title="First boundary loop">
            loop A
            <select
              data-partial-bridge-loop="a"
              value={picks.partialA}
              onChange={(e) => picks.setPartialA(e.currentTarget.value)}
            >
              <option value="">—</option>
              {holeTargets.map((h) => (
                <option key={h.displayIndex} value={h.displayIndex}>
                  #{h.displayIndex + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Start..end vertex indices along loop A">
            arc A
            <input
              type="number"
              data-partial-arc="a-start"
              min={0}
              step={1}
              value={options.arcAStart}
              onChange={(e) => options.setArcAStart(wholeAtLeast(e.currentTarget.value, 0))}
            />
            <span>-</span>
            <input
              type="number"
              data-partial-arc="a-end"
              min={0}
              step={1}
              value={options.arcAEnd}
              onChange={(e) => options.setArcAEnd(wholeAtLeast(e.currentTarget.value, 0))}
            />
          </label>
          <label className="field" title="Second boundary loop (may match loop A to split a hole)">
            loop B
            <select
              data-partial-bridge-loop="b"
              value={picks.partialB}
              onChange={(e) => picks.setPartialB(e.currentTarget.value)}
            >
              <option value="">—</option>
              {holeTargets.map((h) => (
                <option key={h.displayIndex} value={h.displayIndex}>
                  #{h.displayIndex + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="Start..end vertex indices along loop B">
            arc B
            <input
              type="number"
              data-partial-arc="b-start"
              min={0}
              step={1}
              value={options.arcBStart}
              onChange={(e) => options.setArcBStart(wholeAtLeast(e.currentTarget.value, 0))}
            />
            <span>-</span>
            <input
              type="number"
              data-partial-arc="b-end"
              min={0}
              step={1}
              value={options.arcBEnd}
              onChange={(e) => options.setArcBEnd(wholeAtLeast(e.currentTarget.value, 0))}
            />
          </label>
          <label className="field" title="Cyclic offset along arc B">
            twist
            <input
              type="number"
              data-partial-twist="true"
              step={1}
              value={options.partialTwist}
              onChange={(e) => options.setPartialTwist(wholeNumber(e.currentTarget.value))}
            />
          </label>
          <label className="field" title="Interior bridge subdivisions">
            subdiv
            <input
              type="number"
              data-partial-subdiv="true"
              min={0}
              step={1}
              value={options.partialSubdiv}
              onChange={(e) => options.setPartialSubdiv(wholeAtLeast(e.currentTarget.value, 0))}
            />
          </label>
          <button
            type="button"
            data-repair-op="bridgePartialArcs"
            disabled={disabled || picks.partialA === "" || picks.partialB === ""}
            title="Bridge two partial boundary arcs with undo (#210)"
            onClick={() => void actions.bridgePartialArcs()}
          >
            Bridge arc
          </button>
        </div>
        <div className="op">
          <span className="op-label">Join</span>
          <label className="field" title="The mesh copied into this one (and removed)">
            source
            <select
              data-join-source="true"
              value={picks.joinSource}
              onChange={(e) => picks.setJoinSource(e.currentTarget.value === "" ? "" : Number(e.currentTarget.value))}
            >
              <option value="">—</option>
              {items
                .filter((i) => i.mesh !== mesh)
                .map((i) => (
                  <option key={i.id} value={i.id}>
                    {i.name}
                  </option>
                ))}
            </select>
          </label>
          <label className="field" title="This mesh's loop">
            loop A
            <select
              data-join-loop="a"
              value={picks.joinLoopA}
              onChange={(e) => picks.setJoinLoopA(e.currentTarget.value)}
            >
              <option value="">—</option>
              {(s?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <label className="field" title="The source's loop">
            loop B
            <select
              data-join-loop="b"
              value={picks.joinLoopB}
              onChange={(e) => picks.setJoinLoopB(e.currentTarget.value)}
            >
              <option value="">—</option>
              {(picks.sourceSummary?.holes ?? []).map((h, n) => (
                <option key={n} value={n}>
                  #{n + 1} · {h.edges} edges
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            data-action="join"
            disabled={disabled || picks.joinSource === "" || picks.joinLoopA === "" || picks.joinLoopB === ""}
            title="Copy the source into this mesh, bridge the loops, remove the source"
            onClick={() => actions.join()}
          >
            Join
          </button>
        </div>
        {picks.queue.some((q) => q !== null) ? (
          <p className="muted small" data-join-queue>
            Shift-selected join into {item.name}, in order:{" "}
            {picks.queue
              .filter((q) => q !== null)
              .map((q) => q.name)
              .join(", then ")}
            .
          </p>
        ) : null}
        <p className="muted small">Bridge stitches two holes of this mesh; Join merges another mesh into it (the source is removed).</p>
      </div>

      <div className="ops">
        <WrapSection app={app} store={store} items={items} activeItem={item} units={units} disabled={disabled} />
      </div>

      {runner.busy !== null ? (
        <p className="busy small" data-repair-busy={runner.busy}>
          {runner.busy}: computing… <span className="progress" aria-hidden="true" />
        </p>
      ) : runner.report !== null && runner.report.mesh === mesh ? (
        <p className={runner.report.failed ? "report error small" : "report small"} data-repair-report>
          {runner.report.line} <span className="muted">(undo with Ctrl+Z)</span>
        </p>
      ) : null}
    </section>
  );
}

function AnalysisList(props: {
  summary: AnalysisSummary;
  selfIntersects: boolean | null;
  stale: boolean;
  unit: LengthUnit;
}) {
  const { summary: s, selfIntersects, stale, unit } = props;
  const bad = (n: number) => (n > 0 ? "warn" : undefined);
  return (
    <dl
      className={stale ? "analysis stale" : "analysis"}
      data-holes={s.holes.length}
      data-components={s.components.length}
    >
      <dt>Holes</dt>
      <dd className={bad(s.holes.length)}>
        {s.holes.length === 0 ? "none" : s.holes.length}
        {s.holes.length > 0 ? (
          <ul className="holes">
            {s.holes.slice(0, HOLES_SHOWN).map((h, k) => (
              <li key={k}>
                {plural(h.edges, "edge")} · perimeter {formatLength(h.perimeter, unit)}
              </li>
            ))}
            {s.holes.length > HOLES_SHOWN ? <li className="muted">and {s.holes.length - HOLES_SHOWN} more</li> : null}
          </ul>
        ) : null}
        {s.openChains > 0 ? <span className="muted"> ({plural(s.openChains, "open boundary chain")})</span> : null}
      </dd>
      <dt>Orientation</dt>
      <dd className={bad(s.orientationConflicts)}>
        {s.orientationConflicts === 0 ? "consistent" : `${plural(s.orientationConflicts, "conflicting edge")}`}
      </dd>
      <dt>Components</dt>
      <dd className={bad(s.components.length - 1)}>
        {s.components.length}
        <span className="muted">
          {" "}
          (
          {s.components
            .slice(0, 3)
            .map((c) => `${c.triangles.toLocaleString("en")}${c.closed ? "" : " open"}`)
            .join(", ")}
          {s.components.length > 3 ? ", …" : ""} tris)
        </span>
      </dd>
      <dt>Non-manifold</dt>
      <dd className={bad(s.nonManifoldEdges + s.nonManifoldVertices + s.pinchVertices)}>
        {plural(s.nonManifoldEdges, "edge")}, {plural(s.nonManifoldVertices, "vertex", "vertices")}
        {s.pinchVertices > 0 ? `, ${plural(s.pinchVertices, "pinch", "pinches")}` : ""}
      </dd>
      <dt>Degenerate</dt>
      <dd className={bad(s.degenerate + s.duplicate)}>
        {plural(s.degenerate, "triangle")}
        {s.duplicate > 0 ? `, ${plural(s.duplicate, "duplicate")}` : ""}
      </dd>
      <dt>Self-intersections</dt>
      <dd className={selfIntersects ? "warn" : undefined}>
        {selfIntersects === null ? "not checked" : selfIntersects ? "detected" : "none"}
      </dd>
    </dl>
  );
}

/** The overlay toggles: which analysed issues draw on the mesh in 3D. */
const OVERLAY_LABELS: Record<RepairOverlayKind, string> = {
  holes: "Holes",
  nonManifold: "Non-manifold",
  degenerate: "Degenerate",
  components: "Small components",
};

function OverlayToggles(props: { app: AppState; summary: AnalysisSummary }) {
  const { app, summary: s } = props;
  const kinds = useApp(app, (snap) => snap.repairOverlays);
  const counts = overlayIssueCounts(s);
  return (
    <fieldset className="overlays" data-overlays>
      <legend className="muted small">Show in 3D</legend>
      {REPAIR_OVERLAY_KINDS.map((kind) => (
        <label
          key={kind}
          className="check"
          title={`${OVERLAY_LABELS[kind]}: ${counts[kind] === 0 ? "none" : plural(counts[kind], "issue")} on the mesh`}
        >
          <input
            type="checkbox"
            data-overlay={kind}
            checked={kinds[kind]}
            onChange={(e) => app.setRepairOverlay(kind, e.currentTarget.checked)}
          />
          <span
            aria-hidden="true"
            style={{
              display: "inline-block",
              width: "0.7em",
              height: "0.7em",
              borderRadius: "2px",
              background: `#${REPAIR_OVERLAY_COLORS[kind].toString(16).padStart(6, "0")}`,
            }}
          />{" "}
          {OVERLAY_LABELS[kind]} ({counts[kind]})
        </label>
      ))}
    </fieldset>
  );
}
