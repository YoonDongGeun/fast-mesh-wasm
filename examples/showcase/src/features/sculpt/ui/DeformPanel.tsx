// Free-Form Deformation & Self-Collision control panel for #193 (FFD, ARAP, Kelvinlets, XPBD).
import type { ArapWeights, FfdBasis, FfdOutsideMode, KelvinletScale } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState } from "../../../app/state/app-state";
import type { Wave6DeformState } from "../../../app/state/wave6-state";
import { ALGORITHMS, clampInt, parseFinite, useDeformPanel, VIEWPORT_TOOLS } from "../ux/use-deform-panel";

export interface DeformPanelProps {
  app: AppState;
  store: SceneStore;
}

/** React control panel synced with `wave6.deform` for FFD, ARAP, Kelvinlets, and XPBD. */
export function DeformPanel({ app, store }: DeformPanelProps) {
  const {
    deform,
    activeMesh,
    activeItem,
    busy,
    error,
    maskCount,
    patchDeform,
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
  } = useDeformPanel(app, store);

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
                  onClick={() => handleToggleTool(tool.id)}
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
            <button type="button" onClick={handleAddAnchor}>
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
