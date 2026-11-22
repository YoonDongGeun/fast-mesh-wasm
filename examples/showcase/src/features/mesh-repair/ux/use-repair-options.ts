// The repair panel's op parameters, shared by the per-op buttons, Auto
// repair and Selective heal so they never disagree: weld tolerance,
// component cutoff, the Fill holes size limit and fairing, continuity fill,
// the Auto repair preset and self-intersection modes, the arc bridge ranges
// and the bridge loop size; and how their number fields parse.
import type { AutoRepairPreset, FillContinuity, SelfIntersectionMode } from "fast-mesh-wasm";
import { useState } from "react";
import { AUTO_REPAIR_ANY_SIZE } from "./auto-repair";

/** "Any size" for Fill holes (an empty field): the op needs a bound (or a list of loops). */
export const ANY_SIZE = AUTO_REPAIR_ANY_SIZE;
/** The Fill holes size limit to start with: small and medium holes, not an open mesh's outer boundary. */
const DEFAULT_MAX_EDGES = 64;

export function useRepairOptions() {
  const [weldTolerance, setWeldTolerance] = useState(1e-5);
  const [minTriangles, setMinTriangles] = useState(500);
  const [maxEdges, setMaxEdges] = useState(DEFAULT_MAX_EDGES);
  const [fair, setFair] = useState(true);
  const [continuity, setContinuity] = useState<FillContinuity>("curvature");
  const [refinePatch, setRefinePatch] = useState(true);
  const [detectIslands, setDetectIslands] = useState(true);
  const [preset, setPreset] = useState<AutoRepairPreset>("general");
  const [siOverride, setSiOverride] = useState<SelfIntersectionMode | "auto">("auto");
  const [healSiMode, setHealSiMode] = useState<SelfIntersectionMode>("deleteAndFill");
  const [siVoxelSize, setSiVoxelSize] = useState("");
  const [arcAStart, setArcAStart] = useState(0);
  const [arcAEnd, setArcAEnd] = useState(3);
  const [arcBStart, setArcBStart] = useState(0);
  const [arcBEnd, setArcBEnd] = useState(3);
  const [partialTwist, setPartialTwist] = useState(0);
  const [partialSubdiv, setPartialSubdiv] = useState(1);
  const [loopMax, setLoopMax] = useState(128);
  return {
    weldTolerance,
    setWeldTolerance,
    minTriangles,
    setMinTriangles,
    maxEdges,
    setMaxEdges,
    fair,
    setFair,
    continuity,
    setContinuity,
    refinePatch,
    setRefinePatch,
    detectIslands,
    setDetectIslands,
    preset,
    setPreset,
    siOverride,
    setSiOverride,
    healSiMode,
    setHealSiMode,
    siVoxelSize,
    setSiVoxelSize,
    arcAStart,
    setArcAStart,
    arcAEnd,
    setArcAEnd,
    arcBStart,
    setArcBStart,
    arcBEnd,
    setArcBEnd,
    partialTwist,
    setPartialTwist,
    partialSubdiv,
    setPartialSubdiv,
    loopMax,
    setLoopMax,
  };
}

export type RepairOptions = ReturnType<typeof useRepairOptions>;

/** A whole-number field at least `min`; an empty or invalid entry reads as `min`. */
export function wholeAtLeast(text: string, min: number): number {
  return Math.max(min, Math.round(Number(text)) || min);
}

/** A whole-number field of either sign; an empty or invalid entry reads as 0. */
export function wholeNumber(text: string): number {
  return Math.round(Number(text)) || 0;
}

/** The optional voxel size field as a request field: a positive size, or none (automatic). */
export function voxelSizeField(text: string): { voxelSize?: number } {
  return text !== "" && Number(text) > 0 ? { voxelSize: Number(text) } : {};
}
