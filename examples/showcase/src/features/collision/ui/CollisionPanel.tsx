// The collision panel: the contact heatmap of meshes coloured by their
// signed distance to each other (`contactHeatmap`). The heatmap controls are
// shared by two modes: a pair, each mesh coloured against the other, with
// readouts of the deepest penetration, the gap, whether they intersect and
// the triangles in contact; and a contact group (`store.setContactGroup`),
// several meshes each coloured by its nearest partner, with readouts per pair.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useMeshDisplay } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { type AppState, type CollisionPair, type Item, useApp } from "../../../app/state/app-state";
import { formatDistance, summarizeCollisionStepReport } from "../ux/collision-math";
import { GroupMode } from "./GroupMode";
import type { Readouts } from "../../../app/state/readouts";
import { useFieldInteraction } from "../ux/interaction";
import { formatLength, formatLengthValue, fromDisplayLength, type LengthUnit, rangeToDisplay, toDisplayLength } from "../../../shared/ux/units";
import {
  CUT_BACKEND_OPTIONS,
  CUT_TRIGGER_OPTIONS,
  DISTANCE_BACKEND_OPTIONS,
  MEMORY_OPTIONS,
  MODE_OPTIONS,
  ROLE_OPTIONS,
  SOFT_SOLVER_OPTIONS,
  useCollisionRules,
} from "../ux/use-collision-rules";
import { type Heatmap, useHeatmapSettings } from "../ux/use-heatmap-settings";
import { usePairContact } from "../ux/use-pair-contact";

export function CollisionPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const items = useApp(app, (s) => s.items);
  const pair = useApp(app, (s) => s.collision);
  const mode = useApp(app, (s) => s.collisionMode);
  const group = useApp(app, (s) => s.group);
  const shown = mode === "pair" ? [pair.a, pair.b] : group.members;
  const units = useApp(app, (s) => s.units);
  useFieldInteraction(app, scene);
  const { scale, step, depth, gap, tolerance, toleranceMax: tolMax, fast, softness, heat, set } = useHeatmapSettings(
    app,
    items,
    shown,
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
  const {
    collision,
    stiffness,
    contactMargin,
    activePair,
    activePairMode,
    applyCollisionPatch,
    applyCompliance,
    applyRecovery,
    applyStiffness,
    applyContactMargin,
    setPairOverride,
    clearPairOverride,
    canNudge,
    nudging,
    nudgeError,
    lastReport,
    nudgeBTowardA,
  } = useCollisionRules(app, scene, items, pair, unit);

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
        onChange={applyCompliance}
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
        onChange={applyRecovery}
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
            disabled={nudging || !canNudge}
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
  const { a, b } = pair;
  const { active, readouts, warnings } = usePairContact(scene, items, pair, heat, tolerance);

  return (
    <div className="pair">
      <Picker label="A" side="a" app={app} items={items} value={a} />
      <Picker label="B" side="b" app={app} items={items} value={b} />
      {active === null ? (
        <p className="muted small">
          {a !== null && a === b ? "Pick two different meshes." : "Pick two meshes to colour."}
        </p>
      ) : (
        <>
          <PairStatus a={active[0]} b={active[1]} />
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
