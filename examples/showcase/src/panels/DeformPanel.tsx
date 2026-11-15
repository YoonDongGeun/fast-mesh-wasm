// Free-Form Deformation & Self-Collision control panel for #193 (FFD, ARAP, Kelvinlets, XPBD).
import type {
  ArapWeights,
  FfdBasis,
  FfdOutsideMode,
  FreeformProtection,
  KelvinletLoad,
  KelvinletScale,
  MeshHandle,
  XpbdPinInput,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState, useSyncExternalStore } from "react";
import { type AppState, useApp } from "../scene/app-state";
import type { Wave6DeformState } from "../scene/wave6-state";
import {
  buildProtectionOption,
  formatFreeformReport,
  moveFfdControlPoint,
  requireDeformDomain,
  resetLatticeOffsets,
  type Vec3Tuple,
} from "../viewport/DeformHandlesLayer";

export interface DeformPanelProps {
  app: AppState;
  store: SceneStore;
}

type DeformAlgorithm = Wave6DeformState["algorithm"];
type DeformTool = Wave6DeformState["activeTool"];

const ALGORITHMS: ReadonlyArray<{ id: DeformAlgorithm; label: string }> = [
  { id: "ffd", label: "FFD" },
  { id: "arap", label: "ARAP" },
  { id: "kelvinlet", label: "Kelvinlets" },
  { id: "xpbd", label: "XPBD" },
];

const VIEWPORT_TOOLS: ReadonlyArray<{ id: DeformTool; label: string }> = [
  { id: null, label: "Off" },
  { id: "ffdCage", label: "FFD Cage" },
  { id: "arapPin", label: "ARAP Pin" },
  { id: "arapHandle", label: "ARAP Handle" },
  { id: "kelvinletDrag", label: "Kelvinlet Drag" },
  { id: "xpbdPin", label: "XPBD Pin" },
];

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const n = Math.trunc(Number(raw));
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function parseFinite(raw: string, fallback: number): number {
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function formatError(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** React control panel synced with `wave6.deform` for FFD, ARAP, Kelvinlets, and XPBD. */
export function DeformPanel({ app, store }: DeformPanelProps) {
  const deform = useApp(app, (s) => s.wave6.deform);
  const items = useApp(app, (s) => s.items);
  const sculptMask = useApp(app, (s) => s.sculptMask);

  const selection = useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().selection,
    () => store.getSnapshot().selection,
  );
  const activeMesh: MeshHandle | null = selection ?? items[0]?.mesh ?? null;
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

  const patchDeform = (patch: Partial<Wave6DeformState>) => {
    app.updateWave6((prev) => ({
      ...prev,
      deform: { ...prev.deform, ...patch },
    }));
  };

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

  // ---- FFD handlers ----
  const handleBuildLattice = () =>
    void runOp("Build FFD Lattice", async (mesh) => {
      const domain = requireDeformDomain(store);
      const lattice = await domain.buildFfdLattice(mesh, deform.ffdDims, deform.ffdBasis, ffdPadding);
      patchDeform({
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
      patchDeform({
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
      patchDeform({ lastReport: formatFreeformReport("FFD", report) });
    });

  const handleResetLattice = () => {
    if (!deform.ffdLattice) return;
    setError(null);
    patchDeform({
      ffdLattice: resetLatticeOffsets(deform.ffdLattice),
      lastReport: "Reset FFD control point offsets to zero.",
    });
  };

  // ---- ARAP handlers ----
  const handlePinBottomBand = () => {
    if (activeMesh === null) return;
    try {
      const positions = store.scene.views(activeMesh).positions();
      let minY = Infinity;
      let maxY = -Infinity;
      let topVertex = 0;
      for (let v = 0; v * 3 + 1 < positions.length; v++) {
        const y = positions[v * 3 + 1];
        if (y < minY) minY = y;
        if (y >= maxY) {
          maxY = y;
          topVertex = v;
        }
      }
      if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return;
      const cutoff = minY + (maxY - minY) * 0.2;
      const pinned: number[] = [];
      for (let v = 0; v * 3 + 1 < positions.length; v++) {
        if (positions[v * 3 + 1] <= cutoff) {
          pinned.push(v);
        }
      }
      if (pinned.includes(handleVertex) && !pinned.includes(topVertex)) {
        setHandleVertex(topVertex);
      }
      patchDeform({
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
      const b = handleVertex * 3;
      if (b < 0 || b + 2 >= positions.length) {
        throw new Error(`Vertex ${handleVertex} is out of range.`);
      }
      const target: Vec3Tuple = [
        positions[b] + handleVec[0],
        positions[b + 1] + handleVec[1],
        positions[b + 2] + handleVec[2],
      ];
      const nextHandles = [
        ...deform.arapHandles.filter((h) => h.vertex !== handleVertex),
        { vertex: handleVertex, target },
      ];
      const nextAnchors = deform.arapAnchors.filter((v) => v !== handleVertex);
      patchDeform({ arapAnchors: nextAnchors, arapHandles: nextHandles });
    } catch (err) {
      setError(formatError(err));
    }
  };

  const handleApplyArap = () =>
    void runOp("Apply ARAP", async (mesh) => {
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      const report = await domain.applyArap(mesh, {
        anchors: deform.arapAnchors.map((vertex) => ({ vertex })),
        handles: deform.arapHandles,
        weights: deform.arapWeights,
        iterations: arapIterations,
        protect,
      });
      store.commitFrame();
      patchDeform({ lastReport: formatFreeformReport("ARAP", report) });
    });

  // ---- Kelvinlet handlers ----
  const handleCenterOnMesh = () => {
    if (activeMesh === null) return;
    try {
      const positions = store.scene.views(activeMesh).positions();
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      let minZ = Infinity;
      let maxZ = -Infinity;
      for (let i = 0; i + 2 < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
      }
      if (Number.isFinite(minX)) {
        setKelvinCenter([
          Number(((minX + maxX) * 0.5).toFixed(4)),
          Number(((minY + maxY) * 0.5).toFixed(4)),
          Number(((minZ + maxZ) * 0.5).toFixed(4)),
        ]);
      }
    } catch (err) {
      setError(formatError(err));
    }
  };

  const handleApplyKelvinlet = () =>
    void runOp("Apply Kelvinlet", async (mesh) => {
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      let load: KelvinletLoad;
      switch (deform.kelvinletKind) {
        case "grab":
          load = { kind: "grab", force: kelvinForce };
          break;
        case "twist":
          load = { kind: "twist", torque: kelvinForce };
          break;
        case "scale":
          load = { kind: "scale", magnitude: kelvinMagnitude };
          break;
        case "pinch":
          load = {
            kind: "pinch",
            forceDipole: [
              [kelvinForce[0], 0, 0],
              [0, kelvinForce[1], 0],
              [0, 0, kelvinForce[2]],
            ],
          };
          break;
      }
      const report = await domain.applyKelvinlet(mesh, {
        center: kelvinCenter,
        load,
        epsilon: deform.kelvinletEpsilon,
        nu: deform.kelvinletNu,
        scale: deform.kelvinletScale,
        symmetry: {
          x: deform.kelvinletSymX,
          y: deform.kelvinletSymY,
          z: deform.kelvinletSymZ,
        },
        protect,
      });
      store.commitFrame();
      patchDeform({ lastReport: formatFreeformReport("Kelvinlet", report) });
    });

  // ---- XPBD handler ----
  const handleApplyXpbd = () =>
    void runOp("Apply XPBD", async (mesh) => {
      const domain = requireDeformDomain(store);
      const protect = buildCurrentProtection(mesh);
      const pins: XpbdPinInput[] = [];
      if (deform.arapAnchors.length > 0) {
        const positions = store.scene.views(mesh).positions();
        for (const v of deform.arapAnchors) {
          const b = v * 3;
          if (b >= 0 && b + 2 < positions.length) {
            pins.push({ vertex: v, target: [positions[b], positions[b + 1], positions[b + 2]] });
          }
        }
      }
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
      patchDeform({ lastReport: formatFreeformReport("XPBD", report) });
    });

  const maskCount = sculptMask && sculptMask.mesh === activeMesh ? sculptMask.vertices.length : 0;

  return (
    <div className="panel">
      <div className="op">
        <div className="op-head">
          <h4>Free-Form Deformation</h4>
          <span className="muted">
            {activeItem ? activeItem.name : activeMesh !== null ? "Selected mesh" : "No mesh loaded"}
          </span>
        </div>

        <div className="modes" role="group" aria-label="Deform method">
          {ALGORITHMS.map((alg) => (
            <button
              key={alg.id}
              type="button"
              className={deform.algorithm === alg.id ? "active" : ""}
              aria-pressed={deform.algorithm === alg.id}
              onClick={() => patchDeform({ algorithm: alg.id })}
            >
              {alg.label}
            </button>
          ))}
        </div>

        <div className="field">
          <label>Viewport tool</label>
          <div className="modes" role="group" aria-label="Deform viewport tool">
            {VIEWPORT_TOOLS.map((tool) => {
              const active = deform.activeTool === tool.id;
              return (
                <button
                  key={String(tool.id)}
                  type="button"
                  className={active ? "active" : ""}
                  aria-pressed={active}
                  onClick={() =>
                    patchDeform({
                      activeTool: tool.id === null ? null : active ? null : tool.id,
                    })
                  }
                >
                  {tool.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Algorithm-specific controls */}
      {deform.algorithm === "ffd" && (
        <div className="op">
          <h4>FFD Lattice</h4>
          <div className="op-row">
            <label className="field">
              <span>Nx</span>
              <input
                aria-label="FFD grid X"
                type="number"
                min={2}
                max={16}
                value={deform.ffdDims[0]}
                onChange={(e) =>
                  patchDeform({
                    ffdDims: [clampInt(e.target.value, 2, 16, deform.ffdDims[0]), deform.ffdDims[1], deform.ffdDims[2]],
                  })
                }
              />
            </label>
            <label className="field">
              <span>Ny</span>
              <input
                aria-label="FFD grid Y"
                type="number"
                min={2}
                max={16}
                value={deform.ffdDims[1]}
                onChange={(e) =>
                  patchDeform({
                    ffdDims: [deform.ffdDims[0], clampInt(e.target.value, 2, 16, deform.ffdDims[1]), deform.ffdDims[2]],
                  })
                }
              />
            </label>
            <label className="field">
              <span>Nz</span>
              <input
                aria-label="FFD grid Z"
                type="number"
                min={2}
                max={16}
                value={deform.ffdDims[2]}
                onChange={(e) =>
                  patchDeform({
                    ffdDims: [deform.ffdDims[0], deform.ffdDims[1], clampInt(e.target.value, 2, 16, deform.ffdDims[2])],
                  })
                }
              />
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Basis</span>
              <select
                aria-label="FFD basis"
                value={deform.ffdBasis}
                onChange={(e) => patchDeform({ ffdBasis: e.target.value as FfdBasis })}
              >
                <option value="bernstein">Bernstein</option>
                <option value="cubicBSpline">Cubic B-Spline</option>
              </select>
            </label>
            <label className="field">
              <span>Padding</span>
              <input
                aria-label="FFD padding"
                type="number"
                min={0}
                step={0.05}
                value={ffdPadding}
                onChange={(e) => setFfdPadding(Math.max(0, parseFinite(e.target.value, ffdPadding)))}
              />
            </label>
            <label className="field">
              <span>Outside</span>
              <select
                aria-label="FFD outside mode"
                value={ffdOutside}
                onChange={(e) => setFfdOutside(e.target.value as FfdOutsideMode)}
              >
                <option value="clamp">Clamp</option>
                <option value="ignore">Ignore</option>
              </select>
            </label>
          </div>

          <div className="op-row">
            <button type="button" className="primary" disabled={busy || activeMesh === null} onClick={handleBuildLattice}>
              Build Lattice
            </button>
            <button type="button" disabled={busy || !deform.ffdLattice} onClick={handleResetLattice}>
              Reset Lattice
            </button>
          </div>

          {deform.ffdLattice && (
            <>
              <div className="muted">
                Lattice: {deform.ffdLattice.dims.join(" x ")} ({deform.ffdLattice.basis})
              </div>
              <div className="op-row">
                <label className="field">
                  <span>CP i</span>
                  <input
                    aria-label="CP i"
                    type="number"
                    min={0}
                    max={deform.ffdLattice.dims[0] - 1}
                    value={cpIndexIjk[0]}
                    onChange={(e) =>
                      setCpIndexIjk([
                        clampInt(e.target.value, 0, deform.ffdLattice!.dims[0] - 1, cpIndexIjk[0]),
                        cpIndexIjk[1],
                        cpIndexIjk[2],
                      ])
                    }
                  />
                </label>
                <label className="field">
                  <span>CP j</span>
                  <input
                    aria-label="CP j"
                    type="number"
                    min={0}
                    max={deform.ffdLattice.dims[1] - 1}
                    value={cpIndexIjk[1]}
                    onChange={(e) =>
                      setCpIndexIjk([
                        cpIndexIjk[0],
                        clampInt(e.target.value, 0, deform.ffdLattice!.dims[1] - 1, cpIndexIjk[1]),
                        cpIndexIjk[2],
                      ])
                    }
                  />
                </label>
                <label className="field">
                  <span>CP k</span>
                  <input
                    aria-label="CP k"
                    type="number"
                    min={0}
                    max={deform.ffdLattice.dims[2] - 1}
                    value={cpIndexIjk[2]}
                    onChange={(e) =>
                      setCpIndexIjk([
                        cpIndexIjk[0],
                        cpIndexIjk[1],
                        clampInt(e.target.value, 0, deform.ffdLattice!.dims[2] - 1, cpIndexIjk[2]),
                      ])
                    }
                  />
                </label>
              </div>
              <div className="op-row">
                <label className="field">
                  <span>dX</span>
                  <input
                    aria-label="CP delta X"
                    type="number"
                    step={0.05}
                    value={cpDelta[0]}
                    onChange={(e) => setCpDelta([parseFinite(e.target.value, cpDelta[0]), cpDelta[1], cpDelta[2]])}
                  />
                </label>
                <label className="field">
                  <span>dY</span>
                  <input
                    aria-label="CP delta Y"
                    type="number"
                    step={0.05}
                    value={cpDelta[1]}
                    onChange={(e) => setCpDelta([cpDelta[0], parseFinite(e.target.value, cpDelta[1]), cpDelta[2]])}
                  />
                </label>
                <label className="field">
                  <span>dZ</span>
                  <input
                    aria-label="CP delta Z"
                    type="number"
                    step={0.05}
                    value={cpDelta[2]}
                    onChange={(e) => setCpDelta([cpDelta[0], cpDelta[1], parseFinite(e.target.value, cpDelta[2])])}
                  />
                </label>
              </div>
              <div className="op-row">
                <button type="button" disabled={busy} onClick={handleMoveCp}>
                  Move Control Point
                </button>
                <button type="button" className="primary" disabled={busy || activeMesh === null} onClick={handleApplyFfd}>
                  Apply FFD
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {deform.algorithm === "arap" && (
        <div className="op">
          <h4>ARAP Surface Deformation</h4>
          <div className="op-row">
            <label className="field">
              <span>Weights</span>
              <select
                aria-label="ARAP weights"
                value={deform.arapWeights}
                onChange={(e) => patchDeform({ arapWeights: e.target.value as ArapWeights })}
              >
                <option value="cotangent">Cotangent</option>
                <option value="uniform">Uniform</option>
              </select>
            </label>
            <label className="field">
              <span>Iterations</span>
              <input
                aria-label="ARAP iterations"
                type="number"
                min={1}
                max={200}
                value={arapIterations}
                onChange={(e) => setArapIterations(clampInt(e.target.value, 1, 200, arapIterations))}
              />
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Anchor vertex ({deform.arapAnchors.length})</span>
              <input
                aria-label="ARAP anchor vertex"
                type="number"
                min={0}
                value={anchorInput}
                onChange={(e) => setAnchorInput(clampInt(e.target.value, 0, 1_000_000, anchorInput))}
              />
            </label>
            <button
              type="button"
              onClick={() =>
                patchDeform({
                  arapAnchors: deform.arapAnchors.includes(anchorInput)
                    ? deform.arapAnchors
                    : [...deform.arapAnchors, anchorInput],
                })
              }
            >
              Add Anchor
            </button>
            <button type="button" disabled={activeMesh === null} onClick={handlePinBottomBand}>
              Pin Base
            </button>
            <button type="button" onClick={() => patchDeform({ arapAnchors: [] })}>
              Clear Anchors
            </button>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Handle vertex ({deform.arapHandles.length})</span>
              <input
                aria-label="ARAP handle vertex"
                type="number"
                min={0}
                value={handleVertex}
                onChange={(e) => setHandleVertex(clampInt(e.target.value, 0, 1_000_000, handleVertex))}
              />
            </label>
            <button type="button" disabled={activeMesh === null} onClick={handleAddArapHandleFromDelta}>
              Add Handle (+Delta)
            </button>
            <button type="button" onClick={() => patchDeform({ arapHandles: [] })}>
              Clear Handles
            </button>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Delta X</span>
              <input
                aria-label="ARAP delta X"
                type="number"
                step={0.05}
                value={handleVec[0]}
                onChange={(e) => setHandleVec([parseFinite(e.target.value, handleVec[0]), handleVec[1], handleVec[2]])}
              />
            </label>
            <label className="field">
              <span>Delta Y</span>
              <input
                aria-label="ARAP delta Y"
                type="number"
                step={0.05}
                value={handleVec[1]}
                onChange={(e) => setHandleVec([handleVec[0], parseFinite(e.target.value, handleVec[1]), handleVec[2]])}
              />
            </label>
            <label className="field">
              <span>Delta Z</span>
              <input
                aria-label="ARAP delta Z"
                type="number"
                step={0.05}
                value={handleVec[2]}
                onChange={(e) => setHandleVec([handleVec[0], handleVec[1], parseFinite(e.target.value, handleVec[2])])}
              />
            </label>
          </div>

          <div className="op-row">
            <button type="button" className="primary" disabled={busy || activeMesh === null} onClick={handleApplyArap}>
              Apply ARAP
            </button>
          </div>
        </div>
      )}

      {deform.algorithm === "kelvinlet" && (
        <div className="op">
          <h4>Kelvinlets</h4>
          <div className="op-row">
            <label className="field">
              <span>Mode</span>
              <select
                aria-label="Kelvinlet mode"
                value={deform.kelvinletKind}
                onChange={(e) =>
                  patchDeform({ kelvinletKind: e.target.value as Wave6DeformState["kelvinletKind"] })
                }
              >
                <option value="grab">Grab</option>
                <option value="pinch">Pinch</option>
                <option value="twist">Twist</option>
                <option value="scale">Scale</option>
              </select>
            </label>
            <label className="field">
              <span>Scale</span>
              <select
                aria-label="Kelvinlet scale"
                value={deform.kelvinletScale}
                onChange={(e) => patchDeform({ kelvinletScale: e.target.value as KelvinletScale })}
              >
                <option value="single">Single</option>
                <option value="bi">Bi-scale</option>
                <option value="tri">Tri-scale</option>
              </select>
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Radius (epsilon)</span>
              <input
                aria-label="Kelvinlet radius epsilon"
                type="number"
                min={0.01}
                step={0.05}
                value={deform.kelvinletEpsilon}
                onChange={(e) =>
                  patchDeform({
                    kelvinletEpsilon: Math.max(0.01, parseFinite(e.target.value, deform.kelvinletEpsilon)),
                  })
                }
              />
            </label>
            <label className="field">
              <span>Poisson nu</span>
              <input
                aria-label="Kelvinlet Poisson nu"
                type="number"
                min={0}
                max={0.499}
                step={0.01}
                value={deform.kelvinletNu}
                onChange={(e) =>
                  patchDeform({
                    kelvinletNu: Math.max(0, Math.min(0.499, parseFinite(e.target.value, deform.kelvinletNu))),
                  })
                }
              />
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Center X</span>
              <input
                aria-label="Kelvinlet center X"
                type="number"
                step={0.05}
                value={kelvinCenter[0]}
                onChange={(e) =>
                  setKelvinCenter([parseFinite(e.target.value, kelvinCenter[0]), kelvinCenter[1], kelvinCenter[2]])
                }
              />
            </label>
            <label className="field">
              <span>Center Y</span>
              <input
                aria-label="Kelvinlet center Y"
                type="number"
                step={0.05}
                value={kelvinCenter[1]}
                onChange={(e) =>
                  setKelvinCenter([kelvinCenter[0], parseFinite(e.target.value, kelvinCenter[1]), kelvinCenter[2]])
                }
              />
            </label>
            <label className="field">
              <span>Center Z</span>
              <input
                aria-label="Kelvinlet center Z"
                type="number"
                step={0.05}
                value={kelvinCenter[2]}
                onChange={(e) =>
                  setKelvinCenter([kelvinCenter[0], kelvinCenter[1], parseFinite(e.target.value, kelvinCenter[2])])
                }
              />
            </label>
            <button type="button" disabled={activeMesh === null} onClick={handleCenterOnMesh}>
              Center
            </button>
          </div>

          {deform.kelvinletKind === "scale" ? (
            <div className="op-row">
              <label className="field">
                <span>Magnitude</span>
                <input
                  aria-label="Kelvinlet scale magnitude"
                  type="number"
                  step={0.05}
                  value={kelvinMagnitude}
                  onChange={(e) => setKelvinMagnitude(parseFinite(e.target.value, kelvinMagnitude))}
                />
              </label>
            </div>
          ) : (
            <div className="op-row">
              <label className="field">
                <span>Force X</span>
                <input
                  aria-label="Kelvinlet force X"
                  type="number"
                  step={0.05}
                  value={kelvinForce[0]}
                  onChange={(e) =>
                    setKelvinForce([parseFinite(e.target.value, kelvinForce[0]), kelvinForce[1], kelvinForce[2]])
                  }
                />
              </label>
              <label className="field">
                <span>Force Y</span>
                <input
                  aria-label="Kelvinlet force Y"
                  type="number"
                  step={0.05}
                  value={kelvinForce[1]}
                  onChange={(e) =>
                    setKelvinForce([kelvinForce[0], parseFinite(e.target.value, kelvinForce[1]), kelvinForce[2]])
                  }
                />
              </label>
              <label className="field">
                <span>Force Z</span>
                <input
                  aria-label="Kelvinlet force Z"
                  type="number"
                  step={0.05}
                  value={kelvinForce[2]}
                  onChange={(e) =>
                    setKelvinForce([kelvinForce[0], kelvinForce[1], parseFinite(e.target.value, kelvinForce[2])])
                  }
                />
              </label>
            </div>
          )}

          <div className="op-row">
            <label className="check">
              <input
                type="checkbox"
                checked={deform.kelvinletSymX}
                onChange={(e) => patchDeform({ kelvinletSymX: e.target.checked })}
              />
              <span>Sym X</span>
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={deform.kelvinletSymY}
                onChange={(e) => patchDeform({ kelvinletSymY: e.target.checked })}
              />
              <span>Sym Y</span>
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={deform.kelvinletSymZ}
                onChange={(e) => patchDeform({ kelvinletSymZ: e.target.checked })}
              />
              <span>Sym Z</span>
            </label>
          </div>

          <div className="op-row">
            <button type="button" className="primary" disabled={busy || activeMesh === null} onClick={handleApplyKelvinlet}>
              Apply Kelvinlet
            </button>
          </div>
        </div>
      )}

      {deform.algorithm === "xpbd" && (
        <div className="op">
          <h4>XPBD Cloth / Soft Body</h4>
          <div className="op-row">
            <label className="field">
              <span>Substeps</span>
              <input
                aria-label="XPBD substeps"
                type="number"
                min={1}
                max={32}
                value={deform.xpbdSubsteps}
                onChange={(e) => patchDeform({ xpbdSubsteps: clampInt(e.target.value, 1, 32, deform.xpbdSubsteps) })}
              />
            </label>
            <label className="field">
              <span>Iterations</span>
              <input
                aria-label="XPBD iterations"
                type="number"
                min={1}
                max={64}
                value={xpbdIterations}
                onChange={(e) => setXpbdIterations(clampInt(e.target.value, 1, 64, xpbdIterations))}
              />
            </label>
            <label className="field">
              <span>Damping</span>
              <input
                aria-label="XPBD damping"
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={xpbdDamping}
                onChange={(e) => setXpbdDamping(Math.max(0, Math.min(1, parseFinite(e.target.value, xpbdDamping))))}
              />
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Stretch compliance</span>
              <input
                aria-label="XPBD stretch compliance"
                type="number"
                min={0}
                step={0.005}
                value={deform.xpbdStretchCompliance}
                onChange={(e) =>
                  patchDeform({
                    xpbdStretchCompliance: Math.max(0, parseFinite(e.target.value, deform.xpbdStretchCompliance)),
                  })
                }
              />
            </label>
            <label className="field">
              <span>Bend compliance</span>
              <input
                aria-label="XPBD bend compliance"
                type="number"
                min={0}
                step={0.005}
                value={deform.xpbdBendCompliance}
                onChange={(e) =>
                  patchDeform({
                    xpbdBendCompliance: Math.max(0, parseFinite(e.target.value, deform.xpbdBendCompliance)),
                  })
                }
              />
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              <span>Gravity X</span>
              <input
                aria-label="XPBD gravity X"
                type="number"
                step={0.1}
                value={xpbdGravity[0]}
                onChange={(e) => setXpbdGravity([parseFinite(e.target.value, xpbdGravity[0]), xpbdGravity[1], xpbdGravity[2]])}
              />
            </label>
            <label className="field">
              <span>Gravity Y</span>
              <input
                aria-label="XPBD gravity Y"
                type="number"
                step={0.1}
                value={xpbdGravity[1]}
                onChange={(e) => setXpbdGravity([xpbdGravity[0], parseFinite(e.target.value, xpbdGravity[1]), xpbdGravity[2]])}
              />
            </label>
            <label className="field">
              <span>Gravity Z</span>
              <input
                aria-label="XPBD gravity Z"
                type="number"
                step={0.1}
                value={xpbdGravity[2]}
                onChange={(e) => setXpbdGravity([xpbdGravity[0], xpbdGravity[1], parseFinite(e.target.value, xpbdGravity[2])])}
              />
            </label>
          </div>

          <div className="op-row">
            <label className="check">
              <input
                aria-label="Self-collision"
                type="checkbox"
                checked={deform.xpbdSelfCollision}
                onChange={(e) => patchDeform({ xpbdSelfCollision: e.target.checked })}
              />
              <span>Self-collision</span>
            </label>
            <label className="field">
              <span>Thickness</span>
              <input
                aria-label="Self-collision thickness"
                type="number"
                min={0.0001}
                step={0.005}
                disabled={!deform.xpbdSelfCollision}
                value={deform.xpbdThickness}
                onChange={(e) =>
                  patchDeform({
                    xpbdThickness: Math.max(0.0001, parseFinite(e.target.value, deform.xpbdThickness)),
                  })
                }
              />
            </label>
          </div>

          <div className="muted">Pins: {deform.arapAnchors.length} anchor vertex(es)</div>

          <div className="op-row">
            <button type="button" className="primary" disabled={busy || activeMesh === null} onClick={handleApplyXpbd}>
              Apply XPBD
            </button>
          </div>
        </div>
      )}

      {/* Protection section (0 = frozen, 1 = full move) */}
      <div className="op">
        <h4>Protection &amp; Pinning</h4>
        <label className="check">
          <input
            aria-label="Protect sculpt mask"
            type="checkbox"
            checked={deform.xpbdProtectMask}
            onChange={(e) => patchDeform({ xpbdProtectMask: e.target.checked })}
          />
          <span>
            Protect sculpt mask ({maskCount > 0 ? `${maskCount} frozen, weight = 0` : "0 = frozen, 1 = full move"})
          </span>
        </label>

        <label className="check">
          <input
            aria-label="Enable pin box protection"
            type="checkbox"
            checked={pinBoxEnabled}
            onChange={(e) => setPinBoxEnabled(e.target.checked)}
          />
          <span>Freeze pin-box vertices (weight = 0)</span>
        </label>

        {pinBoxEnabled && (
          <>
            <div className="op-row">
              <label className="field">
                <span>Min X</span>
                <input
                  aria-label="Pin box min X"
                  type="number"
                  step={0.1}
                  value={pinBoxMin[0]}
                  onChange={(e) => setPinBoxMin([parseFinite(e.target.value, pinBoxMin[0]), pinBoxMin[1], pinBoxMin[2]])}
                />
              </label>
              <label className="field">
                <span>Min Y</span>
                <input
                  aria-label="Pin box min Y"
                  type="number"
                  step={0.1}
                  value={pinBoxMin[1]}
                  onChange={(e) => setPinBoxMin([pinBoxMin[0], parseFinite(e.target.value, pinBoxMin[1]), pinBoxMin[2]])}
                />
              </label>
              <label className="field">
                <span>Min Z</span>
                <input
                  aria-label="Pin box min Z"
                  type="number"
                  step={0.1}
                  value={pinBoxMin[2]}
                  onChange={(e) => setPinBoxMin([pinBoxMin[0], pinBoxMin[1], parseFinite(e.target.value, pinBoxMin[2])])}
                />
              </label>
            </div>
            <div className="op-row">
              <label className="field">
                <span>Max X</span>
                <input
                  aria-label="Pin box max X"
                  type="number"
                  step={0.1}
                  value={pinBoxMax[0]}
                  onChange={(e) => setPinBoxMax([parseFinite(e.target.value, pinBoxMax[0]), pinBoxMax[1], pinBoxMax[2]])}
                />
              </label>
              <label className="field">
                <span>Max Y</span>
                <input
                  aria-label="Pin box max Y"
                  type="number"
                  step={0.1}
                  value={pinBoxMax[1]}
                  onChange={(e) => setPinBoxMax([pinBoxMax[0], parseFinite(e.target.value, pinBoxMax[1]), pinBoxMax[2]])}
                />
              </label>
              <label className="field">
                <span>Max Z</span>
                <input
                  aria-label="Pin box max Z"
                  type="number"
                  step={0.1}
                  value={pinBoxMax[2]}
                  onChange={(e) => setPinBoxMax([pinBoxMax[0], pinBoxMax[1], parseFinite(e.target.value, pinBoxMax[2])])}
                />
              </label>
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="report error" role="alert">
          {error}
        </div>
      )}
      {deform.lastReport && (
        <div className="report" data-testid="deform-report">
          {deform.lastReport}
        </div>
      )}
    </div>
  );
}
