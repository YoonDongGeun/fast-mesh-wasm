// The mask overlay's buffers, and its object over a bound mesh's geometry
// (no GL: the attributes and the draw range are what the renderer reads).
// Real pixels are in `scripts/showcase-mask-check.mjs`.
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { MASK_TINT_OPACITY, MaskOverlay, maskOverlayBuild } from "../src/viewport/mask-overlay";

const ALPHA = Math.round(255 * MASK_TINT_OPACITY);

/** A strip of `n` quads: `2 * (n + 1)` vertices, `2 * n` triangles. */
function strip(n: number): Uint32Array {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    out.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  return Uint32Array.from(out);
}

describe("maskOverlayBuild", () => {
  it("draws_only_triangles_touching_the_mask", () => {
    const b = maskOverlayBuild([0], strip(3), 8, null);
    expect(Array.from(b.indices.subarray(0, b.count))).toEqual([0, 1, 2]);
    expect(b.colors[3]).toBe(ALPHA);
    expect(b.colors[7]).toBe(0);
  });

  it("skips_dead_triangles", () => {
    const idx = Uint32Array.of(0, 0, 0, 0, 1, 2);
    const b = maskOverlayBuild([0], idx, 3, null);
    expect(Array.from(b.indices.subarray(0, b.count))).toEqual([0, 1, 2]);
  });

  it("ignores_ids_out_of_range", () => {
    const b = maskOverlayBuild([99], strip(1), 4, null);
    expect(b.count).toBe(0);
  });

  it("reuses_its_arrays_and_reports_the_changed_range", () => {
    const idx = strip(3);
    const first = maskOverlayBuild([0, 1, 2], idx, 8, null);
    const second = maskOverlayBuild([0, 1, 2, 5], idx, 8, first);
    expect(second.colors).toBe(first.colors);
    expect(second.changed).toEqual([4 * 5 + 3, 4 * 5 + 4]);
    const same = maskOverlayBuild([0, 1, 2, 5], idx, 8, second);
    expect(same.changed[0]).toBeGreaterThanOrEqual(same.changed[1]);
  });

  it("clears_the_alpha_of_vertices_leaving_the_mask", () => {
    const idx = strip(3);
    const first = maskOverlayBuild([0, 1], idx, 8, null);
    const second = maskOverlayBuild([6], idx, 8, first);
    expect(second.colors[3]).toBe(0);
    expect(second.colors[7]).toBe(0);
    expect(second.colors[4 * 6 + 3]).toBe(ALPHA);
  });
});

describe("MaskOverlay", () => {
  function host(n: number): THREE.Mesh {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * (2 * n + 2)), 3));
    g.setIndex(new THREE.BufferAttribute(strip(n), 1));
    return new THREE.Mesh(g, new THREE.MeshStandardMaterial());
  }

  const overlayMesh = (m: THREE.Mesh) => m.children[0] as THREE.Mesh;

  it("adds_a_child_and_leaves_the_host_untouched", () => {
    const m = host(2);
    const material = m.material;
    const color = m.geometry.getAttribute("color");
    const overlay = new MaskOverlay(m);
    overlay.sync([0]);
    expect(m.children).toHaveLength(1);
    expect(m.material).toBe(material);
    expect(m.geometry.getAttribute("color")).toBe(color);
    expect(overlayMesh(m).geometry.getAttribute("position")).toBe(m.geometry.getAttribute("position"));
  });

  it("rebuilds_only_for_a_new_mask_or_new_triangles", () => {
    const m = host(2);
    const overlay = new MaskOverlay(m);
    const mask = Uint32Array.of(0);
    expect(overlay.sync(mask)).toBe(true);
    expect(overlay.sync(mask)).toBe(false);
    // A stroke's moves: positions change, nothing rebuilds.
    m.geometry.getAttribute("position").needsUpdate = true;
    expect(overlay.sync(mask)).toBe(false);
    // A remesh or an undo: the binding marks the index.
    (m.geometry.getIndex() as THREE.BufferAttribute).needsUpdate = true;
    expect(overlay.sync(mask)).toBe(true);
    expect(overlay.sync(Uint32Array.of(0, 1))).toBe(true);
  });

  it("follows_recreated_attributes", () => {
    const m = host(2);
    const overlay = new MaskOverlay(m);
    overlay.sync([0]);
    const grown = new THREE.BufferAttribute(new Float32Array(3 * 8), 3);
    m.geometry.setAttribute("position", grown);
    overlay.sync([0]);
    expect(overlayMesh(m).geometry.getAttribute("position")).toBe(grown);
  });

  it("takes_a_fresh_geometry_when_the_host_drops_its_buffers", () => {
    // The renderer deletes the host's GL buffers; a kept binding state would keep drawing the old one.
    const m = host(2);
    const overlay = new MaskOverlay(m);
    overlay.sync([0]);
    const before = overlayMesh(m).geometry;
    m.geometry.dispose();
    overlay.sync([0]);
    const after = overlayMesh(m).geometry;
    expect(after).not.toBe(before);
    expect(after.getAttribute("position")).toBe(m.geometry.getAttribute("position"));
    expect(after.getAttribute("color")).toBeDefined();
    expect(after.getIndex()?.array).toBeDefined();
    expect(after.drawRange.count).toBe(3 * overlay.triangles);
    overlay.dispose();
    const third = m.geometry;
    third.dispose(); // No listener left behind to trip on.
  });

  it("clips_with_the_solid", () => {
    const m = host(2);
    const overlay = new MaskOverlay(m);
    const planes = [new THREE.Plane()];
    (m.material as THREE.Material).clippingPlanes = planes;
    overlay.sync([0]);
    expect((overlayMesh(m).material as THREE.Material).clippingPlanes).toBe(planes);
  });

  it("dispose_removes_the_child_but_keeps_the_hosts_buffers", () => {
    const m = host(2);
    const position = m.geometry.getAttribute("position");
    const overlay = new MaskOverlay(m);
    overlay.sync([0]);
    const child = overlayMesh(m);
    let disposedWith: unknown = "not disposed";
    child.geometry.addEventListener("dispose", () => {
      disposedWith = child.geometry.getAttribute("position");
    });
    overlay.dispose();
    expect(m.children).toHaveLength(0);
    // three drops the GL buffer of every attribute still on a disposed geometry.
    expect(disposedWith).toBeUndefined();
    expect(m.geometry.getAttribute("position")).toBe(position);
  });
});
