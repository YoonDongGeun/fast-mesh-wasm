// The collision panel: the contact heatmap of meshes coloured by their
// signed distance to each other (`contactHeatmap`). The heatmap controls are
// shared by two modes: a pair, each mesh coloured against the other, with
// readouts of the deepest penetration, the gap, whether they intersect and
// the triangles in contact; and a contact group (`store.setContactGroup`),
// several meshes each coloured by its nearest partner, with readouts per pair.
import type {
  BooleanMethod,
  CollisionDistanceBackend,
  CollisionMode,
  CollisionRole,
  CollisionRule,
  CollisionStepReport,
  CutTrigger,
  MaterialMemory,
  MeshHandle,
  PlacementInput,
  Scene,
  SoftDeformSolver,
} from "fast-mesh-wasm";
import { useMeshDisplay, useScene } from "fast-mesh-wasm/react";
import { contactHeatmap, type DistanceColoring } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo, useRef, useState } from "react";
import { type AppState, type CollisionPair, type HeatmapParams, type Item, useApp } from "../scene/app-state";
import { createDefaultWave6State, type Wave6CollisionState } from "../scene/wave6-state";
import {
  buildCollisionRule,
  clampTolerance,
  collisionRuleSignature,
  DEFAULT_COLLISION_STIFFNESS,
  DEFAULT_CONTACT_MARGIN,
  deepestPenetration,
  formatDistance,
  niceScale,
  summarizeCollisionStepReport,
  toleranceLimit,
} from "./collision-math";
import { GroupMode } from "./GroupMode";
import { inspect, type Readouts } from "./inspect";
import { useFieldInteraction } from "./interaction";
import { formatLength, formatLengthValue, fromDisplayLength, type LengthUnit, rangeToDisplay, toDisplayLength } from "./units";
import { meshesKey, useLiveReadouts, useTopologyWarnings } from "./live";

export type Heatmap = Omit<DistanceColoring, "distanceTo">;

export function CollisionPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const items = useApp(app, (s) => s.items);
  const pair = useApp(app, (s) => s.collision);
  const mode = useApp(app, (s) => s.collisionMode);
  const group = useApp(app, (s) => s.group);
  const shown = mode === "pair" ? [pair.a, pair.b] : group.members;
  const units = useApp(app, (s) => s.units);
  useFieldInteraction(app, scene);

  // The sliders' scale follows the size of the meshes coloured: a file in
  // millimetres and the unit-sized demo meshes both get useful ranges.
  // In mm, whatever the display unit: changing that moves no value, so no
  // colouring. Only the Slider converts, both ways.
  const scale = niceScale(Math.max(0, ...items.filter((i) => shown.includes(i.mesh)).map((i) => i.radius)));
  const step = scale / 1000;
  // `null` until a slider moves: the defaults then follow the scale.
  // In the app state (persisted: last-used), not local: the status bar's
  // reset restores them with the rest of the display settings.
  const params = useApp(app, (s) => s.heatmapParams);
  const fast = useApp(app, (s) => s.heatmapFast);
  const softness = useApp(app, (s) => s.edgeSoftness);
  const p = params ?? { depth: scale / 2, gap: scale / 2, tolerance: scale / 50 };
  // Clamped to contactHeatmap's limits (spec §6.5, gap ≥ 0 from §7.1).
  const depth = Math.min(Math.max(p.depth, step), scale);
  const gap = Math.min(Math.max(p.gap, 0), scale);
  const tolerance = clampTolerance(p.tolerance, depth, gap, step);
  const tolMax = Math.max(0, toleranceLimit(depth, gap) - step);
  const set = (patch: Partial<HeatmapParams>) => app.setHeatmapParams({ depth, gap, tolerance, ...patch });

  const heat = useMemo(
    () => contactHeatmap({ depth, gap, tolerance, exactDepth: !fast }),
    [depth, gap, tolerance, fast],
  );

  return (
    <section className="panel" aria-label="Collision" data-tour="collision">
      <h2>Collision</h2>
      <div className="modes" role="group" aria-label="Collision mode">
        <button type="button" data-mode="pair" aria-pressed={mode === "pair"} onClick={() => app.setCollisionMode("pair")}>
          Pair
        </button>
        <button
          type="button"
          data-mode="group"
          aria-pressed={mode === "group"}
          onClick={() => app.setCollisionMode("group")}
        >
          Group
        </button>
      </div>
      <p className="muted small">
        {mode === "pair"
          ? "Each mesh coloured by its distance to the other"
          : "Each mesh coloured by its distance to the nearest of its partners"}
        : penetration in reds, contact white, the gap green to blue.
      </p>
      <Slider
        label={`Penetration depth (${units})`}
        unit={units}
        name="depth"
        min={step}
        max={scale}
        step={step}
        value={depth}
        onChange={(v) => set({ depth: v })}
        title="The penetration shown darkest; deeper clamps to it"
      />
      <Slider
        label={`Colour from (${units})`}
        unit={units}
        name="gap"
        min={0}
        max={scale}
        step={step}
        value={gap}
        onChange={(v) => set({ gap: v })}
        title="The gap shown blue; farther shows the mesh colour. 0 colours penetration only"
      />
      <Slider
        label={`Contact tolerance (${units})`}
        unit={units}
        name="tolerance"
        min={0}
        max={tolMax}
        step={step}
        value={tolerance}
        onChange={(v) => set({ tolerance: v })}
        title="Within ± this distance counts as contact (white), and for the contact count"
      />
      <Slider
        label={`Boundary softness (${units})`}
        unit={units}
        name="softness"
        min={0}
        max={scale / 10}
        step={step}
        value={Math.min(softness, scale / 10)}
        onChange={(v) => app.setEdgeSoftness(v)}
        title="The width of the blend where colours meet: the coloured area's edge and the contact band's. 0 is a hard edge"
      />
      <Legend depth={depth} gap={gap} tolerance={tolerance} unit={units} />
      <label
        className="check"
        title="Bound the fields at twice max(depth, colour from): much faster on large meshes, but deeper penetration shows the mesh colour"
      >
        <input
          type="checkbox"
          data-control="fast"
          checked={fast}
          onChange={(e) => app.setHeatmapFast(e.currentTarget.checked)}
        />
        Fast mode (large meshes)
      </label>
      {mode === "pair" ? (
        <PairMode app={app} scene={scene} items={items} pair={pair} heat={heat} tolerance={tolerance} unit={units} />
      ) : (
        <GroupMode app={app} scene={scene} items={items} group={group} heat={heat} />
      )}
      <InterObjectRulesSection app={app} scene={scene} items={items} pair={pair} unit={units} />
    </section>
  );
}

const MODE_OPTIONS: ReadonlyArray<{ value: CollisionMode; label: string }> = [
  { value: "passThrough", label: "passThrough (Overlap)" },
  { value: "rigid", label: "rigid (Block / Slide)" },
  { value: "cut", label: "cut (Boolean)" },
  { value: "softDeform", label: "softDeform (Soft Deform)" },
];

const ROLE_OPTIONS: ReadonlyArray<{ value: CollisionRole; label: string }> = [
  { value: "symmetric", label: "Symmetric" },
  { value: "movingIsCutter", label: "Moving Is Cutter" },
  { value: "movingIsTarget", label: "Moving Is Target" },
];

const DISTANCE_BACKEND_OPTIONS: ReadonlyArray<{ value: CollisionDistanceBackend; label: string }> = [
  { value: "bvhSdf", label: "BVH SDF" },
  { value: "fhEdt", label: "FH-EDT" },
];

const CUT_BACKEND_OPTIONS: ReadonlyArray<{ value: BooleanMethod; label: string }> = [
  { value: "manifold", label: "Manifold" },
  { value: "bvhSdf", label: "BVH SDF" },
  { value: "fhEdt", label: "FH-EDT" },
];

const CUT_TRIGGER_OPTIONS: ReadonlyArray<{ value: CutTrigger; label: string }> = [
  { value: "onRelease", label: "On Release" },
  { value: "liveStep", label: "Live Step" },
];

const SOFT_SOLVER_OPTIONS: ReadonlyArray<{ value: SoftDeformSolver; label: string }> = [
  { value: "sdfLaplacian", label: "SDF Laplacian" },
  { value: "kelvinlet", label: "Kelvinlet" },
  { value: "xpbd", label: "XPBD" },
];

const MEMORY_OPTIONS: ReadonlyArray<{ value: MaterialMemory; label: string }> = [
  { value: "elastic", label: "Elastic (Rollback)" },
  { value: "plastic", label: "Plastic (Retain)" },
];

const INITIAL_DEFAULT_RULE_SIG = collisionRuleSignature(
  buildCollisionRule(createDefaultWave6State().collision),
);

interface PairOverrideEntry {
  a: MeshHandle;
  b: MeshHandle;
  mode: CollisionMode;
}

function canonicalPairKey(a: MeshHandle, b: MeshHandle): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function RuleSelect<T extends string>(props: {
  label: string;
  control: string;
  value: T;
  options: ReadonlyArray<{ value: T; label: string }>;
  onChange: (value: T) => void;
}) {
  const dataAttr = { [`data-collision-${props.control}`]: "" } as Record<string, string>;
  return (
    <label className="field">
      <span>{props.label}</span>
      <select {...dataAttr} value={props.value} onChange={(e) => props.onChange(e.currentTarget.value as T)}>
        {props.options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function RuleRange(props: {
  label: string;
  control: string;
  min: number;
  max: number;
  step: number;
  value: number;
  displayValue: string;
  onChange: (value: number) => void;
}) {
  const dataAttr = { [`data-collision-${props.control}`]: "" } as Record<string, string>;
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{props.label}</span>
        <output>{props.displayValue}</output>
      </span>
      <input
        type="range"
        {...dataAttr}
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.currentTarget.value))}
      />
    </label>
  );
}

/** Wave 6 inter-object collision response rules, pair overrides, and step probe. */
function InterObjectRulesSection(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  pair: CollisionPair;
  unit: LengthUnit;
}) {
  const { app, scene, items, pair, unit } = props;
  const store = useScene();
  const collision = useApp(app, (s) => s.wave6.collision);
  const [stiffness, setStiffness] = useState(DEFAULT_COLLISION_STIFFNESS);
  const [contactMargin, setContactMargin] = useState(DEFAULT_CONTACT_MARGIN);
  const [pairOverrides, setPairOverrides] = useState<Record<string, PairOverrideEntry>>({});
  const [lastReport, setLastReport] = useState<CollisionStepReport<MeshHandle> | null>(null);
  const [nudging, setNudging] = useState(false);
  const [nudgeError, setNudgeError] = useState<string | null>(null);

  const lastSceneRef = useRef<object | null>(store.scene);
  const lastSyncedSigRef = useRef<string>(INITIAL_DEFAULT_RULE_SIG);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  if (lastSceneRef.current !== store.scene) {
    lastSceneRef.current = store.scene;
    lastSyncedSigRef.current = INITIAL_DEFAULT_RULE_SIG;
  }

  const liveMeshes = useMemo(() => new Set(items.map((item) => item.mesh)), [items]);
  useEffect(() => {
    setPairOverrides((prev) => {
      let changed = false;
      const next: Record<string, PairOverrideEntry> = {};
      for (const [key, entry] of Object.entries(prev)) {
        if (liveMeshes.has(entry.a) && liveMeshes.has(entry.b)) {
          next[key] = entry;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [liveMeshes]);

  function syncSceneRules(
    nextState: Wave6CollisionState,
    nextStiffness: number,
    nextMargin: number,
    overrides: Record<string, PairOverrideEntry>,
  ) {
    const rule: CollisionRule = buildCollisionRule(nextState, {
      stiffness: nextStiffness,
      contactMargin: nextMargin,
    });
    const sig = collisionRuleSignature(rule);
    if (sig === lastSyncedSigRef.current) {
      return;
    }
    lastSyncedSigRef.current = sig;
    store.scene.setDefaultCollisionRule?.(rule);
    for (const entry of Object.values(overrides)) {
      store.scene.setCollisionRule?.(
        entry.a,
        entry.b,
        buildCollisionRule(nextState, {
          mode: entry.mode,
          stiffness: nextStiffness,
          contactMargin: nextMargin,
        }),
      );
    }
  }

  useEffect(() => {
    syncSceneRules(collision, stiffness, contactMargin, pairOverrides);
  }, [collision, stiffness, contactMargin, pairOverrides, store]);

  function applyCollisionPatch(patch: Partial<Wave6CollisionState>) {
    const nextState: Wave6CollisionState = { ...collision, ...patch };
    app.updateWave6((prev) => ({
      ...prev,
      collision: { ...prev.collision, ...patch },
    }));
    syncSceneRules(nextState, stiffness, contactMargin, pairOverrides);
  }

  function applyStiffness(raw: number) {
    const next = Number.isFinite(raw) ? Math.max(0, raw) : DEFAULT_COLLISION_STIFFNESS;
    setStiffness(next);
    syncSceneRules(collision, next, contactMargin, pairOverrides);
  }

  function applyContactMargin(displayValue: number) {
    const mm = fromDisplayLength(displayValue, unit);
    const next = Number.isFinite(mm) ? Math.max(0, mm) : DEFAULT_CONTACT_MARGIN;
    setContactMargin(next);
    syncSceneRules(collision, stiffness, next, pairOverrides);
  }

  const activePair =
    pair.a !== null && pair.b !== null && pair.a !== pair.b ? ([pair.a, pair.b] as const) : null;
  const activePairKey = activePair ? canonicalPairKey(activePair[0], activePair[1]) : null;
  const activePairMode = activePairKey ? pairOverrides[activePairKey]?.mode : undefined;

  function setPairOverride(a: MeshHandle, b: MeshHandle, mode: CollisionMode | "default") {
    const key = canonicalPairKey(a, b);
    if (mode === "default") {
      clearPairOverride(a, b);
      return;
    }
    const rule = buildCollisionRule(collision, { mode, stiffness, contactMargin });
    store.scene.setCollisionRule?.(a, b, rule);
    setPairOverrides((prev) => ({ ...prev, [key]: { a, b, mode } }));
  }

  function clearPairOverride(a: MeshHandle, b: MeshHandle) {
    const key = canonicalPairKey(a, b);
    void store.scene.clearCollisionRule?.(a, b);
    setPairOverrides((prev) => {
      if (!(key in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }

  async function nudgeBTowardA() {
    if (!activePair || typeof store.scene.moveWithCollision !== "function") {
      return;
    }
    const [a, b] = activePair;
    setNudging(true);
    setNudgeError(null);
    try {
      const [boundsA, boundsB, placeB] = await Promise.all([
        Promise.resolve(scene.bounds(a)),
        Promise.resolve(scene.bounds(b)),
        Promise.resolve(scene.placement(b)),
      ]);
      if (!boundsA || !boundsB) {
        if (mountedRef.current) {
          setNudgeError("Empty mesh cannot be nudged.");
        }
        return;
      }
      const dx = boundsA.center[0] - boundsB.center[0];
      const dy = boundsA.center[1] - boundsB.center[1];
      const dz = boundsA.center[2] - boundsB.center[2];
      const dist = Math.hypot(dx, dy, dz);
      const step = dist > 1e-6 ? Math.min(dist, Math.max(0.5, dist * 0.5)) : 0;
      const sx = dist > 1e-6 ? (dx / dist) * step : 0.25;
      const sy = dist > 1e-6 ? (dy / dist) * step : 0;
      const sz = dist > 1e-6 ? (dz / dist) * step : 0;
      const nextPlacement: PlacementInput = {
        rotation: placeB.rotation,
        translation: [placeB.translation[0] + sx, placeB.translation[1] + sy, placeB.translation[2] + sz],
        scale: placeB.scale,
      };
      const report = await Promise.resolve(store.scene.moveWithCollision(b, nextPlacement, true));
      if (!mountedRef.current) {
        return;
      }
      const summary = summarizeCollisionStepReport(report);
      setLastReport(report);
      app.updateWave6((prev) => ({
        ...prev,
        collision: { ...prev.collision, lastStepSummary: summary },
      }));
    } catch (err) {
      if (mountedRef.current) {
        setNudgeError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mountedRef.current) {
        setNudging(false);
      }
    }
  }

  const marginRange = rangeToDisplay({ min: 0, max: 10, step: 0.05, value: contactMargin }, unit);

  return (
    <div className="ops" data-collision-rules>
      <div className="panel-head">
        <h3>Inter-Object Response Rules</h3>
      </div>

      <RuleSelect
        label="Default Response"
        control="default-mode"
        value={collision.mode}
        options={MODE_OPTIONS}
        onChange={(mode) => applyCollisionPatch({ mode })}
      />

      {activePair ? (
        <div className="ops">
          <RuleSelect
            label="Pair A↔B Rule"
            control="pair-mode"
            value={activePairMode ?? "default"}
            options={[
              { value: "default", label: `Use Default (${collision.mode})` },
              ...MODE_OPTIONS,
            ]}
            onChange={(mode) => setPairOverride(activePair[0], activePair[1], mode)}
          />
          <button
            type="button"
            data-collision-pair-clear
            disabled={activePairMode === undefined}
            onClick={() => clearPairOverride(activePair[0], activePair[1])}
          >
            Clear Override
          </button>
        </div>
      ) : (
        <p className="muted small">Pick A and B above to configure a pair override rule.</p>
      )}

      <RuleSelect
        label="Role"
        control="role"
        value={collision.role}
        options={ROLE_OPTIONS}
        onChange={(role) => applyCollisionPatch({ role })}
      />
      <RuleSelect
        label="Distance Backend"
        control="distance-backend"
        value={collision.distanceBackend}
        options={DISTANCE_BACKEND_OPTIONS}
        onChange={(distanceBackend) => applyCollisionPatch({ distanceBackend })}
      />
      <RuleSelect
        label="Cut Method"
        control="cut-backend"
        value={collision.cutBackend}
        options={CUT_BACKEND_OPTIONS}
        onChange={(cutBackend) => applyCollisionPatch({ cutBackend })}
      />
      <RuleSelect
        label="Cut Trigger"
        control="cut-trigger"
        value={collision.cutTrigger}
        options={CUT_TRIGGER_OPTIONS}
        onChange={(cutTrigger) => applyCollisionPatch({ cutTrigger })}
      />
      <RuleSelect
        label="Soft Solver"
        control="soft-solver"
        value={collision.softSolver}
        options={SOFT_SOLVER_OPTIONS}
        onChange={(softSolver) => applyCollisionPatch({ softSolver })}
      />
      <RuleSelect
        label="Material Memory"
        control="memory"
        value={collision.materialMemory}
        options={MEMORY_OPTIONS}
        onChange={(materialMemory) => applyCollisionPatch({ materialMemory })}
      />

      <RuleRange
        label="Compliance"
        control="compliance"
        min={0}
        max={1}
        step={0.01}
        value={collision.compliance}
        displayValue={formatDistance(collision.compliance)}
        onChange={(val) => applyCollisionPatch({ compliance: Number.isFinite(val) ? Math.max(0, val) : 0 })}
      />
      <RuleRange
        label="Soft Stiffness"
        control="stiffness"
        min={0}
        max={2}
        step={0.05}
        value={stiffness}
        displayValue={formatDistance(stiffness)}
        onChange={applyStiffness}
      />
      <RuleRange
        label="Elastic Recovery"
        control="recovery"
        min={0}
        max={1}
        step={0.05}
        value={collision.recoveryRate}
        displayValue={formatDistance(collision.recoveryRate)}
        onChange={(val) => applyCollisionPatch({ recoveryRate: Number.isFinite(val) ? Math.min(1, Math.max(0, val)) : 1 })}
      />
      <RuleRange
        label="Contact Margin"
        control="margin"
        min={marginRange.min}
        max={marginRange.max}
        step={marginRange.step}
        value={toDisplayLength(contactMargin, unit)}
        displayValue={formatLength(contactMargin, unit)}
        onChange={applyContactMargin}
      />

      <label className="check">
        <input
          type="checkbox"
          data-collision-rigid-slide
          checked={collision.rigidSlide}
          onChange={(e) => applyCollisionPatch({ rigidSlide: e.currentTarget.checked })}
        />
        Rigid slide along contact
      </label>

      <label className="check">
        <input
          type="checkbox"
          data-collision-self-guard
          checked={collision.preventSelfIntersection}
          onChange={(e) => applyCollisionPatch({ preventSelfIntersection: e.currentTarget.checked })}
        />
        Prevent self-intersection
      </label>

      {activePair && (
        <div className="ops">
          <button
            type="button"
            data-collision-nudge
            disabled={nudging || typeof store.scene.moveWithCollision !== "function"}
            onClick={() => void nudgeBTowardA()}
          >
            {nudging ? "Nudging..." : "Nudge B toward A"}
          </button>
        </div>
      )}

      {(lastReport || collision.lastStepSummary) && (
        <p className="report small" data-collision-report>
          {collision.lastStepSummary ?? (lastReport ? summarizeCollisionStepReport(lastReport) : "")}
        </p>
      )}
      {nudgeError && <p className="warn small">{nudgeError}</p>}
    </div>
  );
}

/** Two picked meshes, each coloured against the other, and their readouts. */
function PairMode(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  items: readonly Item[];
  pair: CollisionPair;
  heat: Heatmap;
  tolerance: number;
  unit: LengthUnit;
}) {
  const { app, scene, items, pair, heat, tolerance, unit } = props;
  const store = useScene();
  const { a, b } = pair;
  const ready = a !== null && b !== null && a !== b;

  // Colour both, each against the other. Picking another mesh (or removing
  // one: AppState.removeMesh resets its picker) stops the old colouring.
  useEffect(() => {
    if (a === null || b === null || a === b) {
      return;
    }
    store.setDisplay(a, { coloring: { distanceTo: b, ...heat } });
    store.setDisplay(b, { coloring: { distanceTo: a, ...heat } });
    return () => {
      for (const m of [a, b]) {
        if (store.getSnapshot().meshes.has(m)) {
          store.setDisplay(m, { coloring: null });
        }
      }
    };
  }, [store, a, b, heat]);

  const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? "the mesh";
  // B's warning first: the side A's colours are measured to.
  const watched = ready ? [b, a] : [];
  const key = meshesKey(items, watched);
  const bound = heat.maxDistance;
  const { readouts, poke } = useLiveReadouts<Readouts>(
    store,
    ready ? watched : null,
    key,
    async () => {
      if (!ready) {
        throw new Error("no pair");
      }
      const [va, vb, distance, hit, contacts] = await Promise.all([
        // After a commit (this runs from change records): the colouring last
        // sent. A colouring replaced meanwhile rejects; read it as none.
        store.fetchColoringValues(a).catch(() => null),
        store.fetchColoringValues(b).catch(() => null),
        Promise.resolve(scene.meshDistance(a, b)),
        Promise.resolve(scene.intersects(a, b)),
        Promise.resolve(scene.findContacts({ margin: tolerance, pairs: [[a, b]] })),
      ]);
      const c = contacts[0];
      return {
        penetration: deepestPenetration([va, vb], bound, hit),
        bound,
        gap: distance === null ? null : distance.distance,
        intersects: hit,
        contacts:
          c === undefined
            ? { a: 0, b: 0 }
            : c.a === a
              ? { a: c.trianglesA.length, b: c.trianglesB.length }
              : { a: c.trianglesB.length, b: c.trianglesA.length },
      };
    },
    inspect.collision,
  );
  // A new tolerance changes the contact count, a new bound how the depth
  // reads (the colours follow through change records).
  useEffect(() => poke(), [tolerance, bound]);
  const warnings = useTopologyWarnings(scene, watched, key, nameOf);
  useEffect(() => {
    inspect.collision.warnings = warnings;
  });

  return (
    <div className="pair">
      <Picker label="A" side="a" app={app} items={items} value={a} />
      <Picker label="B" side="b" app={app} items={items} value={b} />
      {!ready ? (
        <p className="muted small">
          {a !== null && a === b ? "Pick two different meshes." : "Pick two meshes to colour."}
        </p>
      ) : (
        <>
          <PairStatus a={a} b={b} />
          <ReadoutList readouts={readouts} bound={heat.maxDistance} unit={unit} />
          {warnings.map((w) => (
            <p key={w} className="warn small" data-warning>
              {w}
            </p>
          ))}
        </>
      )}
    </div>
  );
}

function Picker(props: {
  label: string;
  side: keyof CollisionPair;
  app: AppState;
  items: readonly Item[];
  value: MeshHandle | null;
}) {
  const { label, side, app, items, value } = props;
  const id = items.find((i) => i.mesh === value)?.id;
  return (
    <label className="picker">
      <span>{label}</span>
      <select
        data-pick={side}
        value={id === undefined ? "" : String(id)}
        onChange={(e) => {
          const v = e.currentTarget.value;
          app.pick(side, items.find((i) => String(i.id) === v)?.mesh ?? null);
        }}
      >
        <option value="">—</option>
        {items.map((i) => (
          <option key={i.id} value={String(i.id)}>
            {i.name}
          </option>
        ))}
      </select>
    </label>
  );
}

/** "Colouring…" while either mesh's colouring is being computed. */
function PairStatus(props: { a: MeshHandle; b: MeshHandle }) {
  const [sa] = useMeshDisplay(props.a);
  const [sb] = useMeshDisplay(props.b);
  if (sa.status === "computing" || sb.status === "computing") {
    return (
      <p className="busy small" data-collision="computing">
        Colouring… <span className="progress" aria-hidden="true" />
      </p>
    );
  }
  return null;
}

function ReadoutList(props: { readouts: Readouts | null; bound: number; unit: LengthUnit }) {
  const { readouts: r, bound, unit } = props;
  const dash = "—";
  const pen = r?.penetration ?? null;
  // A bounded field (fast mode) measures within its bound only; a readout
  // keeps the bound it was measured with.
  const within = r === null ? bound : r.bound;
  return (
    <>
      <dl className="readouts" data-readouts={r === null ? "none" : "ready"}>
        <dt>Max penetration</dt>
        <dd>
          {pen === null
            ? dash
            : pen.atLeast
              ? `≥ ${formatLength(Math.max(pen.depth, within), unit)}`
              : formatLength(pen.depth, unit)}
          {pen !== null && Number.isFinite(within) ? (
            <span className="muted"> (measured within {formatLength(within, unit)})</span>
          ) : null}
        </dd>
        <dt>Min gap</dt>
        <dd>{r === null || r.gap === null ? dash : formatLength(r.gap, unit)}</dd>
        <dt>Intersecting</dt>
        <dd>{r === null ? dash : r.intersects ? "yes" : "no"}</dd>
        <dt title="Triangles of each mesh within the contact tolerance of the other">In contact</dt>
        <dd>
          {r === null ? dash : `${r.contacts.a.toLocaleString("en")} · ${r.contacts.b.toLocaleString("en")} triangles`}
        </dd>
      </dl>
      {Number.isFinite(bound) ? (
        <p className="muted small">
          Fast mode: distances are measured within {formatLength(bound, unit)}; deeper penetration shows the mesh colour.
        </p>
      ) : null}
    </>
  );
}

/** The heatmap's colours from −depth to the gap, as `contactHeatmap` maps them by default. */
function Legend(props: { depth: number; gap: number; tolerance: number; unit: LengthUnit }) {
  const { depth, gap, tolerance, unit } = props;
  const top = gap > 0 ? gap : tolerance;
  const at = (v: number) => `${(((v + depth) / (depth + top)) * 100).toFixed(2)}%`;
  const stops =
    gap > 0
      ? `#7f0000 0%, #ff6b6b ${at(-tolerance)}, #ffffff ${at(-tolerance)}, #ffffff ${at(tolerance)}, #4caf50 ${at(tolerance)}, #1e3a8a 100%`
      : `#7f0000 0%, #ff6b6b ${at(-tolerance)}, #ffffff ${at(-tolerance)}, #ffffff 100%`;
  return (
    <div className="legend" aria-hidden="true">
      <div className="legend-bar" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <div className="legend-labels">
        <span>−{formatLength(depth, unit)}</span>
        {gap > 0 ? (
          <span className="legend-zero" style={{ left: at(0) }}>
            0
          </span>
        ) : null}
        <span>{gap > 0 ? formatLength(gap, unit) : `${formatLength(tolerance, unit)}, then the mesh colour`}</span>
      </div>
    </div>
  );
}

/** A length slider: holds mm, and its input runs in the display unit, converting both ways. */
function Slider(props: {
  label: string;
  unit: LengthUnit;
  name: string;
  min: number;
  max: number;
  step: number;
  value: number;
  title: string;
  onChange: (v: number) => void;
}) {
  const { label, unit, name, min, max, step, value, title, onChange } = props;
  const shown = rangeToDisplay({ min, max, step, value }, unit);
  return (
    <label className="slider" title={title}>
      <span className="slider-head">
        <span>{label}</span>
        <output>{formatLengthValue(value, unit)}</output>
      </span>
      <input
        type="range"
        data-control={name}
        min={shown.min}
        max={shown.max}
        step={shown.step}
        value={shown.value}
        disabled={max <= min}
        onChange={(e) => onChange(fromDisplayLength(Number(e.currentTarget.value), unit))}
      />
    </label>
  );
}
