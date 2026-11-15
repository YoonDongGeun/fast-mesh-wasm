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

  it("frees_replaced_colour_and_index_buffers", () => {
    const m = host(400);
    const overlay = new MaskOverlay(m);
    overlay.sync([0]);
    const small = overlayMesh(m).geometry;
    let freed = false;
    const smallIndex = small.getIndex();
    small.addEventListener("dispose", () => {
      // three drops the GL buffer of each attribute still on a disposed geometry.
      freed = small.getIndex() === smallIndex;
    });
    // More triangles than the first index array holds: a new array, and a new attribute.
    overlay.sync(Uint32Array.from({ length: 802 }, (_, i) => i));
    expect(overlayMesh(m).geometry.getIndex()).not.toBe(smallIndex);
    expect(freed).toBe(true);
    // The grown slots: a new colour array too.
    const position = new THREE.BufferAttribute(new Float32Array(3 * 900), 3);
    m.geometry.setAttribute("position", position);
    const mid = overlayMesh(m).geometry;
    const midColor = mid.getAttribute("color");
    let colorFreed = false;
    mid.addEventListener("dispose", () => {
      colorFreed = mid.getAttribute("color") === midColor;
    });
    overlay.sync(Uint32Array.of(0));
    expect(overlayMesh(m).geometry.getAttribute("color")).not.toBe(midColor);
    expect(colorFreed).toBe(true);
    expect(overlayMesh(m).geometry.getAttribute("position")).toBe(position);
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

  it("keeps_live_gpu_buffers_across_host_dispose_and_split_stretched_passes", () => {
    // Simulates Three.js WebGLGeometries.onGeometryDispose + WebGLAttributes:
    // rendering populates `gpuBuffers`; geometry.dispose() removes every attribute still on that geometry.
    const gpuBuffers = new WeakSet<THREE.BufferAttribute | THREE.InterleavedBufferAttribute>();
    const listened = new WeakSet<THREE.BufferGeometry>();
    const simulateRender = (g: THREE.BufferGeometry) => {
      if (!listened.has(g)) {
        listened.add(g);
        g.addEventListener("dispose", () => {
          const idx = g.getIndex();
          if (idx !== null) gpuBuffers.delete(idx);
          for (const key of Object.keys(g.attributes)) {
            gpuBuffers.delete(g.attributes[key]);
          }
        });
      }
      const idx = g.getIndex();
      if (idx !== null) gpuBuffers.add(idx);
      for (const key of Object.keys(g.attributes)) {
        gpuBuffers.add(g.attributes[key]);
      }
    };

    const m = host(400);
    const overlay = new MaskOverlay(m);
    const mask = Uint32Array.of(0, 1, 2);
    overlay.sync(mask);
    simulateRender(overlayMesh(m).geometry);

    // 1. Structural host disposal without array reallocation (#hostDropped path).
    m.geometry.dispose();
    overlay.sync(mask);
    const g1 = overlayMesh(m).geometry;
    const c1 = g1.getAttribute("color");
    const i1 = g1.getIndex()!;
    expect(gpuBuffers.has(c1)).toBe(true);
    expect(gpuBuffers.has(i1)).toBe(true);
    simulateRender(g1);

    // 2. Host structural disposal + slot growth (splitStretched grows positions while index capacity holds).
    m.geometry.dispose();
    m.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * 900), 3));
    overlay.sync(mask);
    const g2 = overlayMesh(m).geometry;
    expect(g2).not.toBe(g1);
    // Old color buffer was replaced and freed; new color is freshly attached, and kept index buffer survived old.dispose().
    expect(gpuBuffers.has(c1)).toBe(false);
    expect(gpuBuffers.has(g2.getAttribute("color"))).toBe(false); // newly created wrapper ready for WebGLAttributes upload
    expect(g2.getAttribute("color")).not.toBe(c1);
    expect(gpuBuffers.has(i1)).toBe(true);
    expect(g2.getIndex()).toBe(i1);
    simulateRender(g2);

    // 3. Index buffer grows while slot capacity holds (repeated splitStretched passes + undo shrink).
    const c2 = g2.getAttribute("color");
    const wideMask = Uint32Array.from({ length: 802 }, (_, i) => i);
    m.geometry.setIndex(new THREE.BufferAttribute(strip(600), 1));
    overlay.sync(wideMask);
    const g3 = overlayMesh(m).geometry;
    expect(gpuBuffers.has(i1)).toBe(false); // old index buffer freed
    expect(g3.getIndex()).not.toBe(i1);
    expect(gpuBuffers.has(c2)).toBe(true); // kept color buffer survived
    expect(g3.getAttribute("color")).toBe(c2);
    expect(g3.drawRange.count).toBe(3 * overlay.triangles);
    simulateRender(g3);

    // 4. Undo shrinks triangle count back without wiping live buffers.
    overlay.sync(mask);
    expect(gpuBuffers.has(g3.getAttribute("color"))).toBe(true);
    expect(gpuBuffers.has(g3.getIndex()!)).toBe(true);
    expect(g3.drawRange.count).toBe(9);
    expect(g3.drawRange.count).toBe(3 * overlay.triangles);

    overlay.dispose();
  });
});
