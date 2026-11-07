// The sculpt mask's overlay: a translucent tint over the masked vertices.
//
// A child mesh of the bound mesh, drawn on top of it: its own triangle list
// (only triangles touching the mask) and its own per-vertex alpha over the
// binding's position attribute. The mesh's material, colour attributes and
// fragment shading are never touched, so a heatmap on the same mesh stays
// exactly as it was outside the tint, and comes back whole when it goes.
import * as THREE from "three";

/**
 * The tint. Magenta sits apart from the viewer's amber selection glow, the
 * mesh greys and a heatmap's red-white-blue, and it stays a clear colour
 * shift on both near-white and near-black surfaces.
 */
export const MASK_TINT = 0xff1fb4;
/** The tint's opacity at a masked vertex. */
export const MASK_TINT_OPACITY = 0.6;

/** What the overlay draws. Arrays are reused between builds (see `maskOverlayBuild`). */
export interface MaskOverlayBuffers {
  /** RGBA8 per vertex slot: white, with the tint's alpha on a masked slot and 0 elsewhere. */
  colors: Uint8Array;
  /** The drawn triangles' corners; only the first `count` are meaningful. */
  indices: Uint32Array;
  /** How many corners of `indices` are drawn. */
  count: number;
  /** The `colors` byte range this build changed, `[from, to)`; empty when `from >= to`. */
  changed: [number, number];
}

/** The masked-slot flags, one scratch array: a build runs on the frame, one at a time. */
let scratchFlags = new Uint8Array(0);

/**
 * Builds the overlay's buffers over the mesh's current index buffer: every
 * live triangle with a masked corner, so the tint fades over the triangles
 * leaving the mask. Masked ids out of range (a stale mask between its check
 * and its clear) draw nothing. Reuses `prev`'s arrays while they fit: one
 * build per changed mask, on the frame, at O(triangles).
 */
export function maskOverlayBuild(
  masked: ArrayLike<number>,
  indices: ArrayLike<number>,
  slots: number,
  prev: MaskOverlayBuffers | null,
): MaskOverlayBuffers {
  const reuseColors = prev !== null && prev.colors.length === 4 * slots;
  const colors = reuseColors ? prev.colors : new Uint8Array(4 * slots).fill(255);
  if (scratchFlags.length < slots) {
    scratchFlags = new Uint8Array(slots);
  } else {
    scratchFlags.fill(0, 0, slots);
  }
  const flags = scratchFlags;
  for (let k = 0; k < masked.length; k++) {
    const v = masked[k];
    if (v >= 0 && v < slots) {
      flags[v] = 1;
    }
  }
  // Alpha bytes only: the rest stays white. The range keeps uploads small.
  let from = reuseColors ? colors.length : 0;
  let to = reuseColors ? 0 : colors.length;
  const alpha = Math.round(255 * MASK_TINT_OPACITY);
  for (let v = 0; v < slots; v++) {
    const want = flags[v] === 1 ? alpha : 0;
    const at = 4 * v + 3;
    if (colors[at] !== want) {
      colors[at] = want;
      if (at < from) {
        from = at;
      }
      if (at + 1 > to) {
        to = at + 1;
      }
    }
  }
  let out = prev?.indices ?? new Uint32Array(0);
  let count = 0;
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (flags[a] === 1 || flags[b] === 1 || flags[c] === 1) {
      if (count + 3 > out.length) {
        const grown = new Uint32Array(Math.min(Math.max(1024, out.length * 2), indices.length));
        grown.set(out.subarray(0, count));
        out = grown;
      }
      out[count] = a;
      out[count + 1] = b;
      out[count + 2] = c;
      count += 3;
    }
  }
  return { colors, indices: out, count, changed: [from, to] };
}

/** The overlay on one bound mesh: created when shown, disposed when it goes. */
export class MaskOverlay {
  readonly #host: THREE.Mesh;
  readonly #geometry = new THREE.BufferGeometry();
  readonly #material = new THREE.MeshBasicMaterial({
    color: MASK_TINT,
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    // Over the solid's own depth (the same triangles, so it would z-fight).
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
    // The tint reads the same whatever the viewer's tone mapping.
    toneMapped: false,
  });
  readonly #mesh: THREE.Mesh;
  #buffers: MaskOverlayBuffers | null = null;
  #mask: ArrayLike<number> | null = null;
  #indexSeen: THREE.BufferAttribute | null = null;
  #indexVersion = -1;

  constructor(host: THREE.Mesh) {
    this.#host = host;
    this.#mesh = new THREE.Mesh(this.#geometry, this.#material);
    // The binding's object never updates its matrix: the child inherits it.
    this.#mesh.matrixAutoUpdate = false;
    // The binding's geometry has no cheap bounds to share.
    this.#mesh.frustumCulled = false;
    // Picks go through the engine's BVH; never three's per-triangle raycast.
    this.#mesh.raycast = () => {};
    host.add(this.#mesh);
  }

  /** Triangles drawn right now. */
  get triangles(): number {
    return (this.#buffers?.count ?? 0) / 3;
  }

  /**
   * Follows the mask and the host's buffers; call once per frame. Rebuilds
   * only when the mask or the host's triangles changed, so painting pays per
   * frame at most, and a stroke's moves pay nothing (positions are shared).
   * Returns whether it rebuilt.
   */
  sync(masked: ArrayLike<number>): boolean {
    const host = this.#host.geometry;
    const position = host.getAttribute("position") as THREE.BufferAttribute | undefined;
    const index = host.getIndex();
    // The composed clipping (slices, section, box) cuts the tint with the solid.
    const planes = (this.#host.material as THREE.Material).clippingPlanes;
    if (this.#material.clippingPlanes !== planes) {
      this.#material.clippingPlanes = planes;
      this.#material.needsUpdate = true;
    }
    if (position === undefined || index === null) {
      this.#mesh.visible = false;
      return false;
    }
    this.#mesh.visible = true;
    if (this.#geometry.getAttribute("position") !== position) {
      this.#geometry.setAttribute("position", position);
    }
    const slots = position.count;
    const stale =
      this.#mask !== masked ||
      this.#indexSeen !== index ||
      this.#indexVersion !== index.version ||
      this.#buffers?.colors.length !== 4 * slots;
    if (!stale) {
      return false;
    }
    this.#mask = masked;
    this.#indexSeen = index;
    this.#indexVersion = index.version;
    const prev = this.#buffers;
    const next = maskOverlayBuild(masked, index.array as ArrayLike<number>, slots, prev);
    this.#buffers = next;
    this.#upload(prev, next);
    return true;
  }

  #upload(prev: MaskOverlayBuffers | null, next: MaskOverlayBuffers): void {
    if (prev?.colors !== next.colors) {
      this.#geometry.setAttribute("color", new THREE.BufferAttribute(next.colors, 4, true));
    } else if (next.changed[0] < next.changed[1]) {
      const color = this.#geometry.getAttribute("color") as THREE.BufferAttribute;
      color.addUpdateRange(next.changed[0], next.changed[1] - next.changed[0]);
      color.needsUpdate = true;
    }
    const current = this.#geometry.getIndex();
    if (current === null || current.array !== next.indices) {
      this.#geometry.setIndex(new THREE.BufferAttribute(next.indices, 1));
    } else if (next.count > 0) {
      current.addUpdateRange(0, next.count);
      current.needsUpdate = true;
    }
    this.#geometry.setDrawRange(0, next.count);
  }

  /** Removes the overlay. The position attribute is the binding's: it must survive the disposal. */
  dispose(): void {
    this.#host.remove(this.#mesh);
    this.#geometry.deleteAttribute("position");
    this.#geometry.dispose();
    this.#material.dispose();
  }
}
