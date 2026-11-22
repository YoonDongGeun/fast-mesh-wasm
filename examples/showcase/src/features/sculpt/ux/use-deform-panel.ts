// The deform panel's behaviour (#193): its method and viewport tool tables,
// its input parsing, the per-solver parameters it keeps locally, and the
// actions that build and edit the FFD lattice, pick ARAP anchors and handles,
// and run FFD, ARAP, Kelvinlets and XPBD on the active mesh under the
// panel's protection (sculpt mask and pin box).
import type { FfdOutsideMode, FreeformProtection, MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState } from "react";
import type { AppState } from "../../../app/state/app-state";
import type { Wave6DeformState } from "../../../app/state/wave6-state";
import { buildProtectionOption } from "./deform-protection";
import {
  anchorPins,
  arapOptions,
  bottomBandAnchors,
  boundsCenter,
  buildKelvinletLoad,
  formatError,
  formatFreeformReport,
  handleTargetFromDelta,
  kelvinletOptions,
  requireDeformDomain,
} from "./deform-solve";
import { moveFfdControlPoint, resetLatticeOffsets, type Vec3Tuple } from "./ffd-lattice";
import { patchDeform, useDeformState } from "./use-deform-state";

type DeformAlgorithm = Wave6DeformState["algorithm"];
type DeformTool = Wave6DeformState["activeTool"];

/** The method picker's entries. */
export const ALGORITHMS: ReadonlyArray<{ id: DeformAlgorithm; label: string }> = [
  { id: "ffd", label: "FFD" },
  { id: "arap", label: "ARAP" },
  { id: "kelvinlet", label: "Kelvinlets" },
  { id: "xpbd", label: "XPBD" },
];

/** The viewport tool picker's entries; `null` turns the tool off. */
export const VIEWPORT_TOOLS: ReadonlyArray<{ id: DeformTool; label: string }> = [
  { id: null, label: "Off" },
  { id: "ffdCage", label: "FFD Cage" },
  { id: "arapPin", label: "ARAP Pin" },
  { id: "arapHandle", label: "ARAP Handle" },
  { id: "kelvinletDrag", label: "Kelvinlet Drag" },
  { id: "xpbdPin", label: "XPBD Pin" },
];

/** A number input's integer value clamped to `[min, max]`, or `fallback` when not a number. */
export function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const n = Math.trunc(Number(raw));
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

/** A number input's value, or `fallback` when not finite. */
export function parseFinite(raw: string, fallback: number): number {
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

/** The deform panel's state, synced with `wave6.deform`, and its actions. */
export function useDeformPanel(app: AppState, store: SceneStore) {
  const { deform, items, sculptMask, activeMesh } = useDeformState(app, store);
  const activeItem = items.find((i) => i.mesh === activeMesh) ?? null;

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Protection: Pin Box filter (0 = frozen, 1 = full move)
  const [pinBoxEnabled, setPinBoxEnabled] = useState(false);
  const [pinBoxMin, setPinBoxMin] = useState<Vec3Tuple>([-10, -10, -10]);
  const [pinBoxMax, setPinBoxMax] = useState<Vec3Tuple>([10, -0.25, 10]);

  // FFD local parameters
  const [ffdPadding, setFfdPadding] = useState(0);
  const [ffdOutside, setFfdOutside] = useState<FfdOutsideMode>("clamp");
  const [cpIndexIjk, setCpIndexIjk] = useState<[number, number, number]>([1, 2, 1]);
  const [cpDelta, setCpDelta] = useState<Vec3Tuple>([0, 0.25, 0]);

  // Clamp FFD control point indices when lattice dims shrink.
  useEffect(() => {
    const dims = deform.ffdLattice?.dims ?? deform.ffdDims;
    setCpIndexIjk(([i, j, k]) => [
      Math.max(0, Math.min(dims[0] - 1, i)),
      Math.max(0, Math.min(dims[1] - 1, j)),
      Math.max(0, Math.min(dims[2] - 1, k)),
    ]);
  }, [deform.ffdDims, deform.ffdLattice]);

  // ARAP local parameters
  const [arapIterations, setArapIterations] = useState(15);
  const [anchorInput, setAnchorInput] = useState(0);
  const [handleVertex, setHandleVertex] = useState(1);
  const [handleVec, setHandleVec] = useState<Vec3Tuple>([0, 0.25, 0]);

  // Kelvinlet local parameters
  const [kelvinCenter, setKelvinCenter] = useState<Vec3Tuple>([0, 0, 0]);
  const [kelvinForce, setKelvinForce] = useState<Vec3Tuple>([0, 0.3, 0]);
  const [kelvinMagnitude, setKelvinMagnitude] = useState(0.25);

  // XPBD local parameters
  const [xpbdIterations, setXpbdIterations] = useState(4);
  const [xpbdDamping, setXpbdDamping] = useState(0.02);
  const [xpbdGravity, setXpbdGravity] = useState<Vec3Tuple>([0, -0.25, 0]);

  const patch = (next: Partial<Wave6DeformState>) => patchDeform(app, next);

  const buildCurrentProtection = (mesh: MeshHandle): FreeformProtection | undefined => {
    let positions: Float32Array | undefined;
    if (pinBoxEnabled) {
      try {
        positions = store.scene.views(mesh).positions();
      } catch {
        positions = undefined;
      }
    }
    return buildProtectionOption({
      mask: sculptMask,
      activeMesh: mesh,
      protectMask: deform.xpbdProtectMask,
      pinBox:
        pinBoxEnabled && positions
          ? { min: pinBoxMin, max: pinBoxMax, positions }
          : null,
    });
  };

  const runOp = async (label: string, fn: (mesh: MeshHandle) => Promise<void>) => {
    if (activeMesh === null) {
      setError("Load or select a mesh before running deformation.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await fn(activeMesh);
    } catch (err) {
      setError(`${label}: ${formatError(err)}`);
    } finally {
      setBusy(false);
    }
  };

  // ---- Method and viewport tool ----
  const handleToggleTool = (id: DeformTool) =>
    patch({
      activeTool: id === null ? null : deform.activeTool === id ? null : id,
    });

  // ---- FFD handlers ----
  const handleBuildLattice = () =>
    void runOp("Build FFD Lattice", async (mesh) => {
      const domain = requireDeformDomain(store);
      const lattice = await domain.buildFfdLattice(mesh, deform.ffdDims, deform.ffdBasis, ffdPadding);
      patch({
        ffdLattice: lattice,
        lastReport: `Built ${lattice.dims.join("x")} ${lattice.basis} FFD lattice.`,
      });
    });

  const handleMoveCp = () => {
    if (!deform.ffdLattice) {
      setError("Build an FFD lattice first.");
      return;
    }
    setError(null);
    try {
      const next = moveFfdControlPoint(
        deform.ffdLattice,
        cpIndexIjk[0],
        cpIndexIjk[1],
        cpIndexIjk[2],
        cpDelta,
        "delta",
      );
      patch({
        ffdLattice: next,
        lastReport: `Moved FFD control point (${cpIndexIjk.join(", ")}) by [${cpDelta.join(", ")}].`,
      });
    } catch (err) {
      setError(formatError(err));
    }
  };

  const handleApplyFfd = () =>
    void runOp("Apply FFD", async (mesh) => {
      if (!deform.ffdLattice) {
        throw new Error("Build an FFD lattice before applying FFD.");
      }
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      const report = await domain.applyFfd(mesh, {
        lattice: deform.ffdLattice,
        outsideMode: ffdOutside,
        protect,
      });
      store.commitFrame();
      patch({ lastReport: formatFreeformReport("FFD", report) });
    });

  const handleResetLattice = () => {
    if (!deform.ffdLattice) return;
    setError(null);
    patch({
      ffdLattice: resetLatticeOffsets(deform.ffdLattice),
      lastReport: "Reset FFD control point offsets to zero.",
    });
  };

  // ---- ARAP handlers ----
  const handleAddAnchor = () =>
    patch({
      arapAnchors: deform.arapAnchors.includes(anchorInput)
        ? deform.arapAnchors
        : [...deform.arapAnchors, anchorInput],
    });

  const handlePinBottomBand = () => {
    if (activeMesh === null) return;
    try {
      const positions = store.scene.views(activeMesh).positions();
      const band = bottomBandAnchors(positions);
      if (band === null) return;
      const { pinned, topVertex } = band;
      if (pinned.includes(handleVertex) && !pinned.includes(topVertex)) {
        setHandleVertex(topVertex);
      }
      patch({
        arapAnchors: pinned,
        lastReport: `Pinned ${pinned.length} bottom-band vertices as anchors.`,
      });
    } catch (err) {
      setError(formatError(err));
    }
  };

  const handleAddArapHandleFromDelta = () => {
    if (activeMesh === null) {
      setError("Load or select a mesh first.");
      return;
    }
    setError(null);
    try {
      const positions = store.scene.views(activeMesh).positions();
      const target = handleTargetFromDelta(positions, handleVertex, handleVec);
      const nextHandles = [
        ...deform.arapHandles.filter((h) => h.vertex !== handleVertex),
        { vertex: handleVertex, target },
      ];
      const nextAnchors = deform.arapAnchors.filter((v) => v !== handleVertex);
      patch({ arapAnchors: nextAnchors, arapHandles: nextHandles });
    } catch (err) {
      setError(formatError(err));
    }
  };

  const handleApplyArap = () =>
    void runOp("Apply ARAP", async (mesh) => {
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      const report = await domain.applyArap(mesh, arapOptions(deform, arapIterations, protect));
      store.commitFrame();
      patch({ lastReport: formatFreeformReport("ARAP", report) });
    });

  // ---- Kelvinlet handlers ----
  const handleCenterOnMesh = () => {
    if (activeMesh === null) return;
    try {
      const center = boundsCenter(store.scene.views(activeMesh).positions());
      if (center !== null) {
        setKelvinCenter(center);
      }
    } catch (err) {
      setError(formatError(err));
    }
  };

  const handleApplyKelvinlet = () =>
    void runOp("Apply Kelvinlet", async (mesh) => {
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      const load = buildKelvinletLoad(deform.kelvinletKind, kelvinForce, kelvinMagnitude);
      const report = await domain.applyKelvinlet(mesh, kelvinletOptions(deform, kelvinCenter, load, protect));
      store.commitFrame();
      patch({ lastReport: formatFreeformReport("Kelvinlet", report) });
    });

  // ---- XPBD handler ----
  const handleApplyXpbd = () =>
    void runOp("Apply XPBD", async (mesh) => {
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      const pins =
        deform.arapAnchors.length > 0 ? anchorPins(store.scene.views(mesh).positions(), deform.arapAnchors) : [];
      const report = await domain.applyXpbd(mesh, {
        substeps: deform.xpbdSubsteps,
        iterations: xpbdIterations,
        stretchCompliance: deform.xpbdStretchCompliance,
        bendCompliance: deform.xpbdBendCompliance,
        damping: xpbdDamping,
        gravity: xpbdGravity,
        pins,
        handles: deform.arapHandles.map((h) => ({ vertex: h.vertex, target: h.target })),
        selfCollision: deform.xpbdSelfCollision,
        collisionThickness: deform.xpbdThickness,
        protect,
      });
      store.commitFrame();
      patch({ lastReport: formatFreeformReport("XPBD", report) });
    });

  const maskCount = sculptMask && sculptMask.mesh === activeMesh ? sculptMask.vertices.length : 0;

  return {
    deform,
    activeMesh,
    activeItem,
    busy,
    error,
    maskCount,
    patchDeform: patch,
    pinBoxEnabled,
    setPinBoxEnabled,
    pinBoxMin,
    setPinBoxMin,
    pinBoxMax,
    setPinBoxMax,
    ffdPadding,
    setFfdPadding,
    ffdOutside,
    setFfdOutside,
    cpIndexIjk,
    setCpIndexIjk,
    cpDelta,
    setCpDelta,
    arapIterations,
    setArapIterations,
    anchorInput,
    setAnchorInput,
    handleVertex,
    setHandleVertex,
    handleVec,
    setHandleVec,
    kelvinCenter,
    setKelvinCenter,
    kelvinForce,
    setKelvinForce,
    kelvinMagnitude,
    setKelvinMagnitude,
    xpbdIterations,
    setXpbdIterations,
    xpbdDamping,
    setXpbdDamping,
    xpbdGravity,
    setXpbdGravity,
    handleToggleTool,
    handleBuildLattice,
    handleMoveCp,
    handleApplyFfd,
    handleResetLattice,
    handleAddAnchor,
    handlePinBottomBand,
    handleAddArapHandleFromDelta,
    handleApplyArap,
    handleCenterOnMesh,
    handleApplyKelvinlet,
    handleApplyXpbd,
  };
}
