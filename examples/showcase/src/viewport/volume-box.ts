// Placing the main scene's volume box: which volume it shows, where the
// box sits in mesh-local coordinates, and where the cutaway clips it (the
// same planes that cut the mesh, in box fractions).
import type { SliceAxis } from "../panels/volume-slice";

export interface VolumeDims {
  columns: number;
  rows: number;
  slices: number;
}

export interface VolumeFrame extends VolumeDims {
  origin: readonly [number, number, number];
  spacing: readonly [number, number, number];
}

/**
 * The volume the scene shows, if any: the selected mesh's, while the
 * linked MPR state points at the same series. The overlay, the slice
 * quads and the volume box all read this.
 */
export function matchVolume<
  TItem extends { mesh: unknown; volumeId: string | null },
  TMpr extends { volumeId: string },
>(
  items: readonly TItem[],
  selection: TItem["mesh"] | null,
  mpr: TMpr | null,
): { item: TItem; mpr: TMpr; volumeId: string } | null {
  if (selection === null || mpr === null) {
    return null;
  }
  const item = items.find((i) => i.mesh === selection) ?? null;
  if (item === null || item.volumeId === null || mpr.volumeId !== item.volumeId) {
    return null;
  }
  return { item, mpr, volumeId: item.volumeId };
}

/** The box's centre and size in mesh-local coordinates, from the volume frame and the mesh origin. */
export function volumeBoxLayout(
  frame: VolumeFrame,
  center: readonly [number, number, number],
): { center: [number, number, number]; size: [number, number, number] } {
  const ex = frame.columns * frame.spacing[0];
  const ey = frame.rows * frame.spacing[1];
  const ez = frame.slices * frame.spacing[2];
  return {
    center: [
      frame.origin[0] + ex / 2 - center[0],
      frame.origin[1] + ey / 2 - center[1],
      frame.origin[2] + ez / 2 - center[2],
    ],
    size: [ex, ey, ez],
  };
}

/**
 * The cutaway in box fractions (x: sagittal, y: coronal, z: axial): the
 * march keeps the voxels at and below each slice, like the mesh clip.
 */
export function cutFractions(
  slices: Record<SliceAxis, number>,
  dims: VolumeDims,
): [number, number, number] {
  return [
    (slices.sagittal + 0.5) / dims.columns,
    (slices.coronal + 0.5) / dims.rows,
    (slices.axial + 0.5) / dims.slices,
  ];
}
