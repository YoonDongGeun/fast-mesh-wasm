// The Wave 6 inter-object collision response rules: the option tables of
// their selects, the default rule and the pair overrides kept in sync with
// the scene (`setDefaultCollisionRule`, `setCollisionRule`), and the step
// probe that nudges B toward A through `moveWithCollision`.
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
import { useScene } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo, useRef, useState } from "react";
import { type AppState, type CollisionPair, type Item, useApp } from "../../../app/state/app-state";
import { createDefaultWave6State, type Wave6CollisionState } from "../../../app/state/wave6-state";
import { worldCenter } from "../../../shared/ux/mesh-center";
import { fromDisplayLength, type LengthUnit } from "../../../shared/ux/units";
import {
  buildCollisionRule,
  collisionRuleSignature,
  DEFAULT_COLLISION_STIFFNESS,
  DEFAULT_CONTACT_MARGIN,
  nudgeOffset,
  summarizeCollisionStepReport,
} from "./collision-math";

export const MODE_OPTIONS: ReadonlyArray<{ value: CollisionMode; label: string }> = [
  { value: "passThrough", label: "passThrough (Overlap)" },
  { value: "rigid", label: "rigid (Block / Slide)" },
  { value: "cut", label: "cut (Boolean)" },
  { value: "softDeform", label: "softDeform (Soft Deform)" },
];

export const ROLE_OPTIONS: ReadonlyArray<{ value: CollisionRole; label: string }> = [
  { value: "symmetric", label: "Symmetric" },
  { value: "movingIsCutter", label: "Moving Is Cutter" },
  { value: "movingIsTarget", label: "Moving Is Target" },
];

export const DISTANCE_BACKEND_OPTIONS: ReadonlyArray<{ value: CollisionDistanceBackend; label: string }> = [
  { value: "bvhSdf", label: "BVH SDF" },
  { value: "fhEdt", label: "FH-EDT" },
];

export const CUT_BACKEND_OPTIONS: ReadonlyArray<{ value: BooleanMethod; label: string }> = [
  { value: "manifold", label: "Manifold" },
  { value: "bvhSdf", label: "BVH SDF" },
  { value: "fhEdt", label: "FH-EDT" },
];

export const CUT_TRIGGER_OPTIONS: ReadonlyArray<{ value: CutTrigger; label: string }> = [
  { value: "onRelease", label: "On Release" },
  { value: "liveStep", label: "Live Step" },
];

export const SOFT_SOLVER_OPTIONS: ReadonlyArray<{ value: SoftDeformSolver; label: string }> = [
  { value: "sdfLaplacian", label: "SDF Laplacian" },
  { value: "kelvinlet", label: "Kelvinlet" },
  { value: "xpbd", label: "XPBD" },
];

export const MEMORY_OPTIONS: ReadonlyArray<{ value: MaterialMemory; label: string }> = [
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

/**
 * The response rules for the meshes in `items`, with the picked `pair`'s
 * override (`activePair` once two different meshes are picked). The
 * `apply*` setters clamp their input and sync the scene at once; the
 * contact margin is entered in the display `unit`. `nudgeBTowardA` moves B
 * one step toward A with collision, its report landing in `lastReport` and
 * the app state's `lastStepSummary`.
 */
export function useCollisionRules(
  app: AppState,
  scene: Scene | WorkerScene,
  items: readonly Item[],
  pair: CollisionPair,
  unit: LengthUnit,
): {
  collision: Wave6CollisionState;
  stiffness: number;
  contactMargin: number;
  activePair: readonly [MeshHandle, MeshHandle] | null;
  activePairMode: CollisionMode | undefined;
  applyCollisionPatch: (patch: Partial<Wave6CollisionState>) => void;
  applyCompliance: (value: number) => void;
  applyRecovery: (value: number) => void;
  applyStiffness: (raw: number) => void;
  applyContactMargin: (displayValue: number) => void;
  setPairOverride: (a: MeshHandle, b: MeshHandle, mode: CollisionMode | "default") => void;
  clearPairOverride: (a: MeshHandle, b: MeshHandle) => void;
  canNudge: boolean;
  nudging: boolean;
  nudgeError: string | null;
  lastReport: CollisionStepReport<MeshHandle> | null;
  nudgeBTowardA: () => Promise<void>;
} {
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
  const lastLiveRef = useRef(liveMeshes);
  useEffect(() => {
    // An override waits for a mesh that undo or redo can bring back.
    const kept = (mesh: MeshHandle) => liveMeshes.has(mesh) || store.isRestorable?.(mesh) === true;
    setPairOverrides((prev) => {
      let changed = false;
      const next: Record<string, PairOverrideEntry> = {};
      for (const [key, entry] of Object.entries(prev)) {
        if (kept(entry.a) && kept(entry.b)) {
          next[key] = entry;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    // One that is back has the rule it had: it may have changed since.
    const before = lastLiveRef.current;
    lastLiveRef.current = liveMeshes;
    for (const { a, b, mode } of Object.values(pairOverrides)) {
      if (liveMeshes.has(a) && liveMeshes.has(b) && !(before.has(a) && before.has(b))) {
        store.scene.setCollisionRule?.(a, b, buildCollisionRule(collision, { mode, stiffness, contactMargin }));
      }
    }
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
      if (!liveMeshes.has(entry.a) || !liveMeshes.has(entry.b)) {
        continue;
      }
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
      const [sx, sy, sz] = nudgeOffset(worldCenter(scene, a), worldCenter(scene, b));
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

  return {
    collision,
    stiffness,
    contactMargin,
    activePair,
    activePairMode,
    applyCollisionPatch,
    applyCompliance: (val) => applyCollisionPatch({ compliance: Number.isFinite(val) ? Math.max(0, val) : 0 }),
    applyRecovery: (val) =>
      applyCollisionPatch({ recoveryRate: Number.isFinite(val) ? Math.min(1, Math.max(0, val)) : 1 }),
    applyStiffness,
    applyContactMargin,
    setPairOverride,
    clearPairOverride,
    canNudge: typeof store.scene.moveWithCollision === "function",
    nudging,
    nudgeError,
    lastReport,
    nudgeBTowardA,
  };
}
