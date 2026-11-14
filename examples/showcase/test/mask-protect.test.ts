// Per-vertex mask protection: a dab near the mask moves the free side.
// Bytes on a real engine: `scripts/showcase-mask-check.mjs`.
import { type MeshChange, type MeshHandle, type RepairOp, Scene } from "fast-mesh-wasm";
import { createSceneStore } from "fast-mesh-wasm/store";
import { describe, expect, it } from "vitest";
import {
  applyMaskPaintStep,
  inheritedSplits,
  MASK_SOFT_BAND,
  maskEdgeField,
  maskFieldCurrent,
  MaskLiveTracker,
  maskMeetsRemesh,
  protectDab,
  protectMaskedDab,
  updateMaskLiveFromRanges,
  updateMaskLiveFromSplits,
} from "../src/viewport/mask-protect";
import { GrabRegion, maskUsable } from "../src/viewport/sculpt";

/** A strip along x: vertices at (i, 0, 0) and (i, 1, 0), `2 * (n + 1)` slots. */
function strip(n: number): { positions: number[]; indices: number[]; slots: number } {
  const positions: number[] = [];
  for (let i = 0; i <= n; i++) {
    positions.push(i, 0, 0, i, 1, 0);
  }
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    indices.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  return { positions, indices, slots: 2 * (n + 1) };
}

/** The strip's slots at x < `x`. */
const leftOf = (x: number) => Array.from({ length: 2 * x }, (_, i) => i);

const inflateAt = (center: [number, number, number], radius: number): RepairOp => ({
  op: "inflate",
  center,
  radius,
  falloff: "smooth",
  strength: 0.1,
});

describe("maskEdgeField", () => {
  it("weights_masked_slots_zero_and_rises_across_the_band", () => {
    const { positions, indices, slots } = strip(6);
    const f = maskEdgeField(leftOf(2), positions, indices, slots, 2.5);
    const weightOf = (v: number) => f.weights[Array.from(f.ids).indexOf(v)];
    expect(weightOf(0)).toBe(0);
    expect(weightOf(3)).toBe(0);
    const one = weightOf(4); // x = 2, one edge from the mask.
    const two = weightOf(6); // x = 3, two edges from it.
    expect(one).toBeGreaterThan(0);
    expect(two).toBeGreaterThan(one);
    expect(two).toBeLessThan(1);
    // Past the band: not listed, so it moves freely.
    expect(Array.from(f.ids)).not.toContain(8);
  });

  it("flags_exactly_the_masked_slots", () => {
    const { positions, indices, slots } = strip(3);
    const f = maskEdgeField([1, 2, 99], positions, indices, slots, 0);
    expect(Array.from(f.flags)).toEqual([0, 1, 1, 0, 0, 0, 0, 0]);
    expect(Array.from(f.ids)).toEqual([1, 2]);
  });

  it("does_not_walk_through_dead_triangles", () => {
    const f = maskEdgeField([0], [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0], 3, 5);
    expect(Array.from(f.ids)).toEqual([0]);
  });

  it("measures_along_edges_not_through_space", () => {
    // A hairpin: slots 0 and 3 are close in space, several long edges apart.
    const positions = [0, 0, 0, 5, 0, 0, 5, 0.1, 0, 0, 0.1, 0, 5, 0.2, 0];
    const f = maskEdgeField([0], positions, [0, 1, 4, 1, 2, 4, 2, 3, 4], 5, 1);
    expect(Array.from(f.ids)).toEqual([0]);
  });
});

describe("a dab touching the mask's edge", () => {
  const { positions, indices, slots } = strip(8);
  const field = maskEdgeField(leftOf(3), positions, indices, slots, 1.5);

  it("protects_the_masked_vertices_it_reaches_at_weight_zero", () => {
    const op = protectDab(inflateAt([3.5, 0, 0], 2), field, positions) as RepairOp & {
      protect: { vertices: Uint32Array; weights: Float32Array };
    };
    const { vertices, weights } = op.protect;
    // Masked slots 4 and 5 (x = 2) are in the ball; slots 0-1 (x = 0) are past its reach.
    expect(Array.from(vertices)).toContain(4);
    expect(Array.from(vertices)).toContain(5);
    expect(Array.from(vertices)).not.toContain(0);
    expect(weights[Array.from(vertices).indexOf(4)]).toBe(0);
    // Free slots in the band are listed softly; those past it are not listed.
    const soft = Array.from(vertices).indexOf(6);
    expect(weights[soft]).toBeGreaterThan(0);
    expect(weights[soft]).toBeLessThan(1);
    expect(Array.from(vertices)).not.toContain(10);
  });

  it("keeps_the_brush_itself", () => {
    const op = inflateAt([3.5, 0, 0], 2);
    expect(protectDab(op, field, positions)).toMatchObject(op);
  });

  it("returns_the_op_itself_away_from_the_mask", () => {
    const op = inflateAt([8, 0, 0], 1);
    expect(protectDab(op, field, positions)).toBe(op);
  });

  it("leaves_a_non_brush_op_alone", () => {
    const op: RepairOp = { op: "remesh", center: [1, 0, 0], radius: 5 };
    expect(protectDab(op, field, positions)).toBe(op);
  });
});

describe("a dab's field lookup", () => {
  /** A bumpy grid large enough that the walk and the grid differ if they can. */
  function wavy(n: number) {
    const positions: number[] = [];
    for (let y = 0; y <= n; y++) {
      for (let x = 0; x <= n; x++) {
        positions.push(x * 0.1, y * 0.1, 0.2 * Math.sin(x * 0.7) * Math.cos(y * 0.4));
      }
    }
    const indices: number[] = [];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x;
        indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
      }
    }
    return { positions, indices, slots: (n + 1) * (n + 1) };
  }

  it("finds_what_a_walk_over_the_whole_field_finds", () => {
    const { positions, indices, slots } = wavy(60);
    const masked = Array.from({ length: slots }, (_, i) => i).filter((v) => positions[3 * v] < 3 && positions[3 * v + 1] > 1.5);
    const radius = 0.6;
    const field = maskEdgeField(masked, positions, indices, slots, radius * 0.25);
    expect(field.grid).not.toBeNull();
    for (const center of [[3, 1.5, 0], [0.2, 5.9, 0.1], [2.7, 3.3, -0.1], [5.5, 0.5, 0], [-1, -1, 0]] as [number, number, number][]) {
      const op = protectDab(inflateAt(center, radius), field, positions) as RepairOp & { protect?: { vertices: Uint32Array; weights: Float32Array } };
      const walk = protectDab(inflateAt(center, radius), { ...field, grid: null }, positions) as typeof op;
      const pairs = (o: typeof op) =>
        Array.from(o.protect?.vertices ?? [], (v, k) => [v, o.protect?.weights[k] ?? -1]).sort((a, b) => a[0] - b[0]);
      expect(pairs(op)).toEqual(pairs(walk));
    }
  });

  it("knows_when_something_else_moved_a_masked_vertex", () => {
    const grid = wavy(10);
    const { indices, slots } = grid;
    const positions = Float32Array.from(grid.positions);
    const field = maskEdgeField([0, 1, 2], positions, indices, slots, 0.1);
    expect(maskFieldCurrent(field, positions)).toBe(true);
    // A free vertex moving is a stroke at work; a masked one is someone else.
    const free = positions.slice();
    free[3 * 50] += 1;
    expect(maskFieldCurrent(field, free)).toBe(true);
    const moved = positions.slice();
    moved[3 * 1 + 2] += 1;
    expect(maskFieldCurrent(field, moved)).toBe(false);
  });
});

describe("a split between masked vertices", () => {
  /** A 4 x 4 flat grid, stretched by a grab with no split, and the vertices now under a mask. */
  function stretched() {
    const n = 60;
    const positions: number[] = [];
    for (let y = 0; y <= n; y++) {
      for (let x = 0; x <= n; x++) {
        positions.push((x / n) * 4, (y / n) * 4, 0);
      }
    }
    const indices: number[] = [];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x;
        indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
      }
    }
    const scene = new Scene();
    const mesh = scene.addMesh(positions, indices);
    const v = scene.views(mesh);
    scene.repair(mesh, { op: "captureRestSizes" });
    const first = new GrabRegion(v.positions(), v.indices(), [0, 0, 0], 1, "smooth", 1, scene.origin(mesh));
    scene.edit(mesh, [{ op: "setPositions", ...first.at([0, 0, 1.5]) }]);
    const local = v.positions();
    const masked: number[] = [];
    for (let i = 0; i < local.length / 3; i++) {
      if (Math.hypot(local[3 * i], local[3 * i + 1]) < 0.9) {
        masked.push(i);
      }
    }
    const report = scene.repair(mesh, { op: "splitStretched", center: [0, 0, 0], radius: 3, maxSplits: 400 });
    const set = new Set(masked);
    const fresh = inheritedSplits((id) => set.has(id), report.vertices);
    return { scene, mesh, v, masked, fresh };
  }

  const fieldOf = (s: ReturnType<typeof stretched>, ids: readonly number[]) =>
    maskEdgeField(ids, s.v.positions(), s.v.indices(), s.v.positions().length / 3, 0.25);

  const movedOf = (before: ArrayLike<number>, after: ArrayLike<number>, ids: readonly number[]) =>
    ids.filter((i) => [0, 1, 2].some((c) => before[3 * i + c] !== after[3 * i + c])).length;

  it("happens_on_an_edge_that_stretched_before_the_mask", () => {
    expect(stretched().fresh.length).toBeGreaterThan(100);
  });

  it("follows_chains_through_new_vertices", () => {
    // 10 between masked 0 and 1; 11 between 10 and masked 2; 12 has a free parent.
    expect(inheritedSplits((id) => id < 3, [10, 0, 1, 11, 10, 2, 12, 11, 5])).toEqual([10, 11]);
  });

  it("holds_the_new_vertices_under_a_dab", () => {
    const s = stretched();
    const all = [...s.masked, ...s.fresh];
    const before = Float32Array.from(s.v.positions());
    const op = { op: "grab", center: [0, 0, 0.7], radius: 1.5, falloff: "smooth", strength: 1, direction: [0.3, 0, 0] } as const;
    // What the owner would see without them: the mask holds, its new vertices do not.
    s.scene.repair(s.mesh, protectDab(op, fieldOf(s, s.masked), s.v.positions()));
    expect(movedOf(before, s.v.positions(), s.fresh)).toBeGreaterThan(0);
    s.scene.undo();
    s.scene.repair(s.mesh, protectDab(op, fieldOf(s, all), s.v.positions()));
    expect(movedOf(before, s.v.positions(), all)).toBe(0);
    expect(Array.from(s.v.positions())).not.toEqual(Array.from(before));
  });

  it("keeps_a_grab_out_through_each_rebase", () => {
    const s = stretched();
    const all = [...s.masked, ...s.fresh];
    const before = Float32Array.from(s.v.positions());
    const center: [number, number, number] = [0, 0, 0.7];
    // The field the region reads now, then one that knows the next split's vertices.
    let held = fieldOf(s, s.masked);
    const region = new GrabRegion(s.v.positions(), s.v.indices(), center, 1.5, "smooth", 1, s.scene.origin(s.mesh), () => held);
    held = fieldOf(s, all);
    region.rebase(s.v.positions(), s.v.indices());
    const inRegion = new Set(region.ids);
    expect(all.some((id) => inRegion.has(id))).toBe(false);
    for (let k = 1; k <= 6; k++) {
      s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at([0.1 * k, 0, 0]) }]);
      const report = s.scene.repair(s.mesh, { op: "splitStretched", center, radius: 3, maxSplits: 200 });
      // In the order the stroke goes: the split joins the mask, then the region is taken again.
      const set = new Set(all);
      const more = inheritedSplits((id) => set.has(id), report.vertices);
      all.push(...more);
      held = fieldOf(s, all);
      region.rebase(s.v.positions(), s.v.indices(), report.vertices);
      // And again with no split list, as after a pass that reports nothing.
      region.rebase(s.v.positions(), s.v.indices());
      const taken = new Set(region.ids);
      expect(all.some((id) => taken.has(id))).toBe(false);
      expect(movedOf(before, s.v.positions(), [...s.masked, ...s.fresh])).toBe(0);
    }
    expect(region.ids.length).toBeGreaterThan(0);
  });

  it("does_not_make_the_mask_stale_when_undo_takes_them", () => {
    const s = stretched();
    const vertices = Uint32Array.from([...s.masked, ...s.fresh].sort((a, b) => a - b));
    const slots = s.v.positions().length / 3;
    s.scene.undo();
    const now = s.v.indices();
    const inherited = new Set(s.fresh);
    expect(maskUsable({ vertices, slots, inherited }, now, s.v.positions().length / 3)).toBe(true);
    // Without the note it would read as a topology edit taking a masked vertex.
    expect(maskUsable({ vertices, slots }, now, s.v.positions().length / 3)).toBe(false);
  });
});

describe("a stroke-end remesh near the mask", () => {
  /** A flat `n x n` grid over [0, 1]². */
  function grid(n: number) {
    const positions: number[] = [];
    for (let y = 0; y <= n; y++) {
      for (let x = 0; x <= n; x++) {
        positions.push(x / n, y / n, 0);
      }
    }
    const indices: number[] = [];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x;
        indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
      }
    }
    return { positions, indices };
  }

  it("is_refused_when_it_can_collapse_a_masked_vertex_just_outside_the_region", () => {
    const n = 40;
    const { positions, indices } = grid(n);
    const scene = new Scene();
    const mesh = scene.addMesh(positions, indices);
    // Brush centers are local: the mesh's own origin is at the grid's middle.
    const local = Float32Array.from(scene.views(mesh).positions());
    const center: [number, number, number] = [local[3 * 840], local[3 * 840 + 1], local[3 * 840 + 2]];
    const radius = 0.2;
    // Vertices past the region, within half an edge of it: edges reaching them have midpoints inside.
    const masked: number[] = [];
    for (let v = 0; v < local.length / 3; v++) {
      const d = Math.hypot(local[3 * v] - center[0], local[3 * v + 1] - center[1]);
      if (d > radius && d < radius + 0.4 / n) {
        masked.push(v);
      }
    }
    expect(masked.length).toBeGreaterThan(0);
    const guard = maskMeetsRemesh(masked, local, indices, center, radius);
    scene.repair(mesh, { op: "remesh", center, radius, targetLength: 0.2, maxRounds: 4 });
    // What the guard is for: the engine really takes one of them.
    const now = scene.views(mesh);
    const mask = { vertices: Uint32Array.from(masked), slots: local.length / 3 };
    expect(maskUsable(mask, now.indices(), now.positions().length / 3)).toBe(false);
    expect(guard).toBe(true);
  });

  it("is_allowed_well_away_from_the_mask", () => {
    const { positions, indices } = grid(40);
    expect(maskMeetsRemesh([0], positions, indices, [0.5, 0.5, 0], 0.2)).toBe(false);
    expect(maskMeetsRemesh([], positions, indices, [0.5, 0.5, 0], 0.2)).toBe(false);
  });
});

describe("engine-localized protectMaskedDab and incremental maskCache (#197)", () => {
  it("filters_only_masked_vertices_near_active_symmetry_spheres_with_soft_band", () => {
    // Vertices at x = -4, -1, 1, 4 along x axis
    const positions = new Float32Array([-4, 0, 0, -1, 0, 0, 1, 0, 0, 4, 0, 0]);
    const live = new Uint8Array([1, 1, 1, 1]);
    const masked = new Uint32Array([0, 1, 2, 3]);
    const baseOp: RepairOp = {
      op: "inflate",
      center: [1.1, 0, 0],
      radius: 0.4,
      falloff: "smooth",
      strength: 0.05,
      symmetry: { x: true, y: false, z: false },
    };
    const decorated = protectMaskedDab(baseOp, masked, positions, live) as Extract<RepairOp, { op: "inflate" }>;
    expect(decorated.protect).toBeDefined();
    // Both +1 and reflected -1 are within (0.4 + 0.1) * 1.25 = 0.625 of [1.1,0,0] / [-1.1,0,0]; +/-4 are excluded.
    expect(Array.from(decorated.protect?.vertices ?? [])).toEqual([1, 2]);
    expect(Array.from(decorated.protect?.weights ?? [])).toEqual([0, 0]);
    expect(decorated.protect?.band).toBeCloseTo(0.4 * MASK_SOFT_BAND, 6);

    // Dead inherited split vertex is excluded so check_protection never rejects after undo.
    const liveAfterUndo = new Uint8Array([1, 0, 1, 1]);
    const afterUndo = protectMaskedDab(baseOp, masked, positions, liveAfterUndo) as Extract<RepairOp, { op: "inflate" }>;
    expect(Array.from(afterUndo.protect?.vertices ?? [])).toEqual([2]);
  });

  it("updates_mask_live_incrementally_on_ranges_and_splits_and_invalidates_on_removal", () => {
    const live = new Uint8Array([1, 1, 1, 0]);
    const indices = new Uint32Array([0, 1, 2, 1, 2, 3, 2, 3, 4]);
    const grown = updateMaskLiveFromRanges({ slots: 4, live }, indices, 5, [[1, 2]]);
    expect(grown).not.toBeNull();
    expect(grown?.slots).toBe(5);
    expect(Array.from(grown?.live ?? [])).toEqual([1, 1, 1, 1, 1]);

    // A removed triangle (0, 0, 0) in a dirty range invalidates the cache.
    const withDead = new Uint32Array([0, 1, 2, 0, 0, 0, 2, 3, 4]);
    expect(updateMaskLiveFromRanges(grown!, withDead, 5, [[1, 1]])).toBeNull();

    // Split growth marks new split slots live in O(splits) without scanning indices.
    const splitGrown = updateMaskLiveFromSplits({ slots: 3, live: new Uint8Array([1, 1, 1]) }, 5, [3, 0, 1, 4, 1, 2]);
    expect(splitGrown.slots).toBe(5);
    expect(Array.from(splitGrown.live)).toEqual([1, 1, 1, 1, 1]);
  });

  it("retains_live_cache_across_post_split_commitFrame_without_rescanning_all_triangles", () => {
    const scene = new Scene();
    const s = strip(3);
    const mesh = scene.addMesh(Float64Array.from(s.positions), Uint32Array.from(s.indices));
    scene.commitFrame();
    const mask = { mesh, vertices: Uint32Array.from([0, 1]) };

    // Count full-triangle scans by observing index element reads via Proxy once cache is active.
    let indexElementReads = 0;
    const origViews = scene.views.bind(scene);
    const wrappedScene = {
      views(m: typeof mesh) {
        const v = origViews(m);
        return {
          positions: () => v.positions(),
          indices: () =>
            new Proxy(v.indices(), {
              get(target, prop, receiver) {
                if (typeof prop === "string" && /^\d+$/.test(prop)) {
                  indexElementReads++;
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
              },
            }),
        };
      },
      onChanges: (cb: Parameters<typeof scene.onChanges>[0]) => scene.onChanges(cb),
    };

    const tracker = new MaskLiveTracker(wrappedScene);
    tracker.attach();
    const first = tracker.liveSnapshot(mesh, mask);
    expect(first).not.toBeNull();
    expect(first).not.toBe("stale");
    expect(indexElementReads).toBeGreaterThan(0);

    // A position-only dab keeps the cache with zero index element reads.
    indexElementReads = 0;
    scene.repair(mesh, { op: "inflate", center: [2, 0.5, 0], radius: 0.8, falloff: "smooth", strength: 0.2 });
    scene.commitFrame();
    const afterDab = tracker.liveSnapshot(mesh, mask);
    expect(afterDab).not.toBe("stale");
    expect(indexElementReads).toBe(0);

    // Split + capacity change delivered at a later commitFrame() after endSplit keeps cache live without rescanning.
    indexElementReads = 0;
    tracker.beginSplit(mesh);
    scene.repair(mesh, { op: "captureRestSizes", onlyIfMissing: false });
    scene.repair(mesh, {
      op: "grab",
      center: [1.5, 0.5, 0],
      radius: 1.2,
      falloff: "smooth",
      strength: 1,
      direction: [0, 0, 2.0],
    });
    const splitRep = scene.repair(mesh, { op: "splitStretched", center: [1.5, 0.5, 0], radius: 3, maxSplits: 16 });
    expect(splitRep.splits).toBeGreaterThan(0);
    const newSlots = origViews(mesh).positions().length / 3;
    tracker.endSplit(mesh, true, newSlots, splitRep.vertices);
    scene.commitFrame();
    const afterSplit = tracker.liveSnapshot(mesh, mask);
    expect(afterSplit).not.toBeNull();
    expect(afterSplit).not.toBe("stale");
    expect(typeof afterSplit === "object" && afterSplit !== null ? afterSplit.live.length : 0).toBe(newSlots);
    expect(indexElementReads).toBe(0);

    // maskUsable still agrees on the rebuilt mesh, and a real undo invalidates cleanly without crash.
    const rawViews = origViews(mesh);
    expect(maskUsable({ vertices: mask.vertices, slots: rawViews.positions().length / 3 }, rawViews.indices(), rawViews.positions().length / 3)).toBe(true);
    scene.undo();
    scene.undo();
    scene.commitFrame();
    tracker.liveSnapshot(mesh, mask);
    expect(indexElementReads).toBeGreaterThan(0);
    tracker.dispose();
  });

  it("trusts_shared_mask_paint_and_split_adoption_with_zero_triangle_scans_until_undo", () => {
    const scene = new Scene();
    const s = strip(3);
    const mesh = scene.addMesh(Float64Array.from(s.positions), Uint32Array.from(s.indices));
    scene.commitFrame();
    const origViews = scene.views.bind(scene);
    let indexElementReads = 0;
    const shared = new MaskLiveTracker({
      views(m) {
        const v = origViews(m);
        return {
          positions: () => v.positions(),
          indices: () =>
            new Proxy(v.indices(), {
              get(target, prop) {
                if (typeof prop === "string" && /^\d+$/.test(prop)) {
                  indexElementReads++;
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
              },
            }),
        };
      },
      onChanges: (cb) => scene.onChanges(cb),
    });
    shared.attach();

    const slots = origViews(mesh).positions().length / 3;
    const picked = Uint32Array.from(scene.verticesInSphere(mesh, [0.5, 0.5, 0], 1.1));
    expect(picked.length).toBeGreaterThan(0);
    shared.trust(mesh, picked, slots, true);

    // MaskLayer checkMask and SculptLayer liveSnapshot both run with 0 triangle index reads.
    expect(shared.checkMask(mesh, { mesh, vertices: picked })).toEqual({ ok: true, slots });
    const snap = shared.liveSnapshot(mesh, { mesh, vertices: picked });
    expect(snap).not.toBeNull();
    expect(snap).not.toBe("stale");
    expect(indexElementReads).toBe(0);

    // Live splits update trusted state cleanly with 0 triangle scans too.
    shared.beginSplit(mesh);
    scene.repair(mesh, { op: "captureRestSizes", onlyIfMissing: false });
    scene.repair(mesh, {
      op: "grab",
      center: [1.5, 0.5, 0],
      radius: 1.2,
      falloff: "smooth",
      strength: 1,
      direction: [0, 0, 2.0],
    });
    const sr = scene.repair(mesh, { op: "splitStretched", center: [1.5, 0.5, 0], radius: 3, maxSplits: 16 });
    const afterSlots = origViews(mesh).positions().length / 3;
    shared.endSplit(mesh, sr.splits > 0, afterSlots, sr.vertices);
    const nextMask = Uint32Array.from(new Set([...picked, ...Array.from(sr.vertices).filter((_, k) => k % 3 === 0)])).sort(
      (a, b) => a - b,
    );
    shared.trustSplits(mesh, picked, nextMask, afterSlots);
    scene.commitFrame();

    const inherited = new Set(Array.from(sr.vertices).filter((_, k) => k % 3 === 0));
    const afterSplitSnap = shared.liveSnapshot(mesh, { mesh, vertices: nextMask, inherited });
    expect(afterSplitSnap).not.toBeNull();
    expect(afterSplitSnap).not.toBe("stale");
    expect(indexElementReads).toBe(0);

    // Undo drops trust and forces full triangle validation so dead inherited splits stay live===0.
    scene.undo();
    scene.undo();
    scene.commitFrame();
    const afterUndo = shared.liveSnapshot(mesh, { mesh, vertices: nextMask, inherited });
    expect(afterUndo).not.toBeNull();
    expect(afterUndo).not.toBe("stale");
    if (typeof afterUndo === "object" && afterUndo !== null) {
      for (const id of inherited) {
        expect(afterUndo.live[id]).toBe(0);
      }
    }
    expect(indexElementReads).toBeGreaterThan(0);
    shared.dispose();
  });

  it("reattaches_cleanly_after_dispose_in_strict_mode_without_double_subscription", () => {
    const scene = new Scene();
    const s = strip(3);
    const mesh = scene.addMesh(Float64Array.from(s.positions), Uint32Array.from(s.indices));
    scene.commitFrame();
    const origViews = scene.views.bind(scene);
    const listeners = new Set<(changes: MeshChange[]) => void>();
    const unsub = scene.onChanges((cs) => {
      for (const l of [...listeners]) {
        l(cs);
      }
    });
    let indexElementReads = 0;
    const tracker = new MaskLiveTracker({
      views(m) {
        const v = origViews(m);
        return {
          positions: () => v.positions(),
          indices: () =>
            new Proxy(v.indices(), {
              get(target, prop) {
                if (typeof prop === "string" && /^\d+$/.test(prop)) {
                  indexElementReads++;
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
              },
            }),
        };
      },
      onChanges(cb) {
        listeners.add(cb);
        return () => {
          listeners.delete(cb);
        };
      },
    });

    // Side-effect-free constructor does not subscribe before attach().
    expect(listeners.size).toBe(0);

    // Mount 1 + double attach guard.
    tracker.attach();
    tracker.attach();
    expect(listeners.size).toBe(1);

    const mask = { mesh, vertices: Uint32Array.from([0, 1]) };
    const snap1 = tracker.liveSnapshot(mesh, mask);
    expect(snap1).not.toBeNull();
    expect(snap1).not.toBe("stale");
    expect(indexElementReads).toBeGreaterThan(0);

    // StrictMode unmount disposes listener and clears cache/trust.
    const epochBeforeDispose = tracker.topologyVersion(mesh);
    tracker.dispose();
    expect(listeners.size).toBe(0);

    // Mutation while detached happens silently (removed triangles leave vertex 0 dead).
    scene.edit(mesh, [{ op: "removeTriangles", triangles: [0] }]);
    scene.commitFrame();

    // StrictMode remount re-attaches, bumps topology epoch, invalidates stale state, and receives events again.
    tracker.attach();
    expect(listeners.size).toBe(1);
    expect(tracker.topologyVersion(mesh)).toBeGreaterThan(epochBeforeDispose);

    indexElementReads = 0;
    const afterReattach = tracker.liveSnapshot(mesh, mask);
    expect(afterReattach).toBe("stale");
    expect(indexElementReads).toBeGreaterThan(0);

    // Still receives subsequent commitFrame() events cleanly (position dab retains cache).
    const liveMask = { mesh, vertices: Uint32Array.from([4, 5]) };
    expect(tracker.liveSnapshot(mesh, liveMask)).not.toBe("stale");
    indexElementReads = 0;
    scene.repair(mesh, { op: "inflate", center: [2, 0.5, 0], radius: 0.8, falloff: "smooth", strength: 0.1 });
    scene.commitFrame();
    expect(tracker.liveSnapshot(mesh, liveMask)).not.toBe("stale");
    expect(indexElementReads).toBe(0);

    tracker.dispose();
    expect(listeners.size).toBe(0);
    unsub();
  });

  it("invalidates_when_split_coalesces_with_non_split_all_or_dead_range_edits", () => {
    // 1) Mock scene covering coalesced split + non-split "all" / capacity mismatch / dead dirty ranges.
    let positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    let indices = new Uint32Array([0, 1, 2, 1, 3, 2]);
    let listener: ((changes: MeshChange[]) => void) | null = null;
    let indexReads = 0;
    const fakeMesh = { id: 1 } as unknown as MeshHandle;
    const mockTracker = new MaskLiveTracker({
      views() {
        return {
          positions: () => positions,
          indices: () =>
            new Proxy(indices, {
              get(target, prop) {
                if (typeof prop === "string" && /^\d+$/.test(prop)) {
                  indexReads++;
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
              },
            }),
        };
      },
      onChanges(cb) {
        listener = cb;
        return () => {
          listener = null;
        };
      },
    });
    mockTracker.attach();

    const mask = { mesh: fakeMesh, vertices: new Uint32Array([0, 1, 3]) };
    expect(mockTracker.liveSnapshot(fakeMesh, mask)).not.toBe("stale");

    // Case A: endSplit ran (4 -> 5 slots, 2 -> 4 tris), then coalesced "all" rebuild WITHOUT capacityChanged.
    mockTracker.beginSplit(fakeMesh);
    positions = new Float32Array(5 * 3);
    indices = new Uint32Array([0, 1, 4, 1, 3, 4, 4, 3, 2, 0, 4, 2]);
    mockTracker.endSplit(fakeMesh, true, 5, [4, 1, 3]);
    // Coalesced edit later removes triangles referencing vertex 3 with capacityChanged: false, indices: "all".
    indices = new Uint32Array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    indexReads = 0;
    listener!([
      {
        mesh: fakeMesh,
        added: false,
        removed: false,
        placementChanged: false,
        capacityChanged: false,
        positions: "all",
        normals: "all",
        colors: [],
        indices: "all",
        fieldsChanged: [],
        sectionsChanged: [],
      },
    ]);
    expect(mockTracker.liveSnapshot(fakeMesh, mask)).toBe("stale");
    expect(indexReads).toBeGreaterThan(0);

    // Case B: Dirty range with (0,0,0) dead triangle coalesced after endSplit must invalidate (never skip via fromSplit).
    mockTracker.invalidate(fakeMesh);
    positions = new Float32Array(5 * 3);
    indices = new Uint32Array([0, 1, 4, 1, 3, 4]);
    expect(mockTracker.liveSnapshot(fakeMesh, mask)).not.toBe("stale");
    mockTracker.beginSplit(fakeMesh);
    mockTracker.endSplit(fakeMesh, true, 5, [4, 1, 3]);
    indices = new Uint32Array([0, 1, 4, 0, 0, 0]);
    indexReads = 0;
    listener!([
      {
        mesh: fakeMesh,
        added: false,
        removed: false,
        placementChanged: false,
        capacityChanged: false,
        positions: [[3, 1]],
        normals: "all",
        colors: [],
        indices: [[1, 1]],
        fieldsChanged: [],
        sectionsChanged: [],
      },
    ]);
    expect(mockTracker.liveSnapshot(fakeMesh, mask)).toBe("stale");
    expect(indexReads).toBeGreaterThan(0);

    // Out-of-range triangle ranges or vertex IDs return null from updateMaskLiveFromRanges.
    expect(updateMaskLiveFromRanges({ slots: 5, live: new Uint8Array(5).fill(1) }, indices, 5, [[5, 2]])).toBeNull();
    expect(
      updateMaskLiveFromRanges({ slots: 5, live: new Uint8Array(5).fill(1) }, new Uint32Array([0, 1, 99, 1, 2, 3]), 5, [
        [0, 1],
      ]),
    ).toBeNull();

    // Case C: Real Scene where splitStretched coalesces with non-split edit before commitFrame().
    const scene = new Scene();
    const s = strip(3);
    const mesh = scene.addMesh(Float64Array.from(s.positions), Uint32Array.from(s.indices));
    scene.commitFrame();
    const realMaskVerts = Uint32Array.from(scene.verticesInSphere(mesh, [-1.5, -0.5, 0], 0.2));
    expect(Array.from(realMaskVerts)).toContain(0);
    const tracker = new MaskLiveTracker(scene);
    tracker.attach();
    tracker.trust(mesh, realMaskVerts, scene.views(mesh).positions().length / 3, true);

    tracker.beginSplit(mesh);
    scene.repair(mesh, { op: "captureRestSizes", onlyIfMissing: false });
    scene.repair(mesh, { op: "grab", center: [1.5, 0.5, 0], radius: 1.2, falloff: "smooth", strength: 1, direction: [0, 0, 2] });
    const sr = scene.repair(mesh, { op: "splitStretched", center: [1.5, 0.5, 0], radius: 3, maxSplits: 16 });
    expect(sr.splits).toBeGreaterThan(0);
    const afterSplitSlots = scene.views(mesh).positions().length / 3;
    tracker.endSplit(mesh, true, afterSplitSlots, sr.vertices);
    // Coalesce a non-split edit (removing triangle 0 so vertex 0 becomes dead) in the same frame before commitFrame().
    scene.edit(mesh, [{ op: "removeTriangles", triangles: [0] }]);
    scene.commitFrame();
    expect(tracker.checkMask(mesh, { mesh, vertices: realMaskVerts })).toEqual({
      ok: false,
      slots: scene.views(mesh).positions().length / 3,
    });
    tracker.dispose();

    // Case D1: Real Scene where splitStretched is undone in the same frame before commitFrame() (slots stay 10!).
    const s2 = strip(3);
    const mesh2 = scene.addMesh(Float64Array.from(s2.positions), Uint32Array.from(s2.indices));
    scene.commitFrame();
    const mask2Verts = Uint32Array.from(scene.verticesInSphere(mesh2, [-1.5, -0.5, 0], 0.2));
    const origD1Views = scene.views.bind(scene);
    let d1Reads = 0;
    const hookedScene = {
      views: (m: MeshHandle) => {
        const v = origD1Views(m);
        return {
          positions: () => v.positions(),
          indices: () =>
            new Proxy(v.indices(), {
              get(target, prop) {
                if (typeof prop === "string" && /^\d+$/.test(prop)) {
                  d1Reads++;
                }
                const value = Reflect.get(target, prop, target);
                return typeof value === "function" ? value.bind(target) : value;
              },
            }),
        };
      },
      onChanges: scene.onChanges.bind(scene),
      undo: scene.undo.bind(scene),
      repair: scene.repair.bind(scene),
    };
    const d1Tracker = new MaskLiveTracker(hookedScene);
    d1Tracker.attach();
    const cleanSlots = origD1Views(mesh2).positions().length / 3;
    d1Tracker.trust(mesh2, mask2Verts, cleanSlots, true);
    d1Tracker.beginSplit(mesh2);
    hookedScene.repair(mesh2, { op: "captureRestSizes", onlyIfMissing: false });
    hookedScene.repair(mesh2, {
      op: "grab",
      center: [1.5, 0.5, 0],
      radius: 1.2,
      falloff: "smooth",
      strength: 1,
      direction: [0, 0, 2],
    });
    const sr2 = hookedScene.repair(mesh2, { op: "splitStretched", center: [1.5, 0.5, 0], radius: 3, maxSplits: 16 });
    expect(sr2.splits).toBeGreaterThan(0);
    const grownSlots = origD1Views(mesh2).positions().length / 3;
    d1Tracker.endSplit(mesh2, true, grownSlots, sr2.vertices);
    const splitInherited = new Set(Array.from(sr2.vertices).filter((_, k) => k % 3 === 0));
    const expandedMask = Uint32Array.from(new Set([...mask2Verts, ...splitInherited])).sort((a, b) => a - b);
    d1Tracker.trustSplits(mesh2, mask2Verts, expandedMask, grownSlots);
    // Coalesced undo in same frame keeps slots === 10 while turning split triangles into (0, 0, 0).
    hookedScene.undo();
    scene.commitFrame();
    d1Reads = 0;
    const d1Snap = d1Tracker.liveSnapshot(mesh2, { mesh: mesh2, vertices: expandedMask, inherited: splitInherited });
    expect(d1Snap).not.toBe("stale");
    expect(d1Reads).toBeGreaterThan(0);
    if (typeof d1Snap === "object" && d1Snap !== null) {
      for (const id of splitInherited) {
        expect(d1Snap.live[id]).toBe(0);
      }
    }
    d1Tracker.dispose();

    // Case D2: Unhooked mock where capacityChanged: true, indices: "all" arrives with changed slots/tris after endSplit.
    mockTracker.invalidate(fakeMesh);
    positions = new Float32Array(5 * 3);
    indices = new Uint32Array([0, 1, 4, 1, 3, 4]);
    expect(mockTracker.liveSnapshot(fakeMesh, mask)).not.toBe("stale");
    mockTracker.beginSplit(fakeMesh);
    mockTracker.endSplit(fakeMesh, true, 5, [4, 1, 3]);
    // Coalesced non-split shrink/edit changes slots (6 !== 5) and removes vertex 3 before flush.
    positions = new Float32Array(6 * 3);
    indices = new Uint32Array([0, 1, 4, 1, 2, 4, 2, 5, 4]);
    indexReads = 0;
    listener!([
      {
        mesh: fakeMesh,
        added: false,
        removed: false,
        placementChanged: false,
        capacityChanged: true,
        positions: "all",
        normals: "all",
        colors: [],
        indices: "all",
        fieldsChanged: [],
        sectionsChanged: [],
      },
    ]);
    expect(mockTracker.liveSnapshot(fakeMesh, mask)).toBe("stale");
    expect(indexReads).toBeGreaterThan(0);
    mockTracker.dispose();
  });

  it("blocks_microtask_paint_trust_when_undo_or_non_split_repair_runs_before_commitFrame_with_identical_slots", async () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    // 6 vertex slots; triangle 2 references vertex 4 while vertex 5 is dead -> vertexCount === 5, slots === 6.
    const mesh = store.addMesh(
      Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 2, 0.2, 0]),
      Uint32Array.from([0, 1, 2, 1, 3, 2, 2, 3, 4]),
    );
    scene.commitFrame();

    // Edit step swaps triangle-used vertex 4 -> 5 while keeping vertexCount (6) and slots (6) identical.
    await store.edit(mesh, [
      { op: "removeTriangles", triangles: [2] },
      { op: "addTriangles", triangles: [2, 3, 5] },
    ]);
    scene.commitFrame();
    const slotsBefore = scene.views(mesh).positions().length / 3;
    expect(scene.vertexCount(mesh)).toBe(6);
    expect(slotsBefore).toBe(6);

    let viewsCalls = 0;
    const spiedViews = (m: MeshHandle) => {
      viewsCalls++;
      return scene.views(m);
    };
    const tracker = new MaskLiveTracker(scene, store);
    tracker.attach();

    // Pre-existing mask containing vertex 5 (live right now before undo).
    const existingMask = { mesh, vertices: new Uint32Array([2, 3, 5]), slots: slotsBefore };
    tracker.trust(mesh, existingMask.vertices, slotsBefore, true);

    // Start async paintAt querying verticesInSphere; trigger store.undo() in the microtask window BEFORE commitFrame().
    let resolveQuery!: (ids: Uint32Array) => void;
    const queryPromise = new Promise<Uint32Array>((r) => {
      resolveQuery = r;
    });
    let setMaskCalled = 0;
    let checkMaskCalled = 0;
    let trustCalled = 0;
    const origCheck = tracker.checkMask.bind(tracker);
    const origTrust = tracker.trust.bind(tracker);
    tracker.checkMask = (...a) => {
      checkMaskCalled++;
      return origCheck(...a);
    };
    tracker.trust = (...a) => {
      trustCalled++;
      return origTrust(...a);
    };

    const paintPromise = applyMaskPaintStep({
      mesh,
      centers: [[2, 0.2, 0]],
      radius: 0.5,
      verticesInSphere: () => queryPromise,
      views: spiedViews,
      tracker,
      isAlive: () => true,
      getPrevMask: () => existingMask,
      setMask: () => {
        setMaskCalled++;
      },
      onStale: () => {},
    });

    // Undo runs BEFORE commitFrame(), keeping vertexCount === 6 and slots === 6 identical while killing vertex 5.
    store.undo();
    expect(scene.vertexCount(mesh)).toBe(6);
    expect(scene.views(mesh).positions().length / 3).toBe(6);

    viewsCalls = 0;
    resolveQuery(new Uint32Array([5]));
    const applied = await paintPromise;
    expect(applied).toBe(false);
    expect(viewsCalls).toBe(0);
    expect(checkMaskCalled).toBe(0);
    expect(trustCalled).toBe(0);
    expect(setMaskCalled).toBe(0);
    // Even BEFORE commitFrame(), tracker must never return stale trust for vertex 5.
    expect(origCheck(mesh, existingMask)).toEqual({ ok: false, slots: 6 });

    // Same microtask guard protects against scene.cancelStep(), store.redo(), and async non-split repair.
    store.redo();
    scene.commitFrame();
    const redoMask = { mesh, vertices: new Uint32Array([2, 3, 5]), slots: 6 };
    origTrust(mesh, redoMask.vertices, 6, true);
    let resolveCancel!: (ids: Uint32Array) => void;
    const pCancel = applyMaskPaintStep({
      mesh,
      centers: [[2, 0.2, 0]],
      radius: 0.5,
      verticesInSphere: () => new Promise<Uint32Array>((r) => (resolveCancel = r)),
      views: spiedViews,
      tracker,
      isAlive: () => true,
      getPrevMask: () => redoMask,
      setMask: () => {
        setMaskCalled++;
      },
      onStale: () => {},
    });
    scene.beginStep();
    scene.edit(mesh, [{ op: "removeTriangles", triangles: [0] }]);
    scene.cancelStep();
    resolveCancel(new Uint32Array([5]));
    expect(await pCancel).toBe(false);
    expect(setMaskCalled).toBe(0);

    // Async non-split repair (remesh/removeDegenerate) dispatched/settled before commitFrame() also blocks paint trust.
    scene.commitFrame();
    origTrust(mesh, redoMask.vertices, 6, true);
    let resolveRepair!: (ids: Uint32Array) => void;
    const pRepair = applyMaskPaintStep({
      mesh,
      centers: [[2, 0.2, 0]],
      radius: 0.5,
      verticesInSphere: () => new Promise<Uint32Array>((r) => (resolveRepair = r)),
      views: spiedViews,
      tracker,
      isAlive: () => true,
      getPrevMask: () => null,
      setMask: () => {
        setMaskCalled++;
      },
      onStale: () => {},
    });
    await store.repair(mesh, { op: "removeDegenerate" });
    resolveRepair(new Uint32Array([5]));
    expect(await pRepair).toBe(false);
    expect(setMaskCalled).toBe(0);

    tracker.dispose();

    // Case E: Async Worker-style store undo dispatched BEFORE paintAt and settling via subscribe BEFORE commitFrame().
    let asyncStatus = "idle";
    const subs = new Set<() => void>();
    const asyncStore = {
      getSnapshot: () => ({ status: asyncStatus }),
      subscribe(fn: () => void) {
        subs.add(fn);
        return () => subs.delete(fn);
      },
      undo() {
        asyncStatus = "computing";
        for (const fn of [...subs]) fn();
      },
    };
    const asyncTracker = new MaskLiveTracker(scene, asyncStore);
    asyncTracker.attach();
    asyncStore.undo(); // dispatched beforehand
    let resolveAsyncPaint!: (ids: Uint32Array) => void;
    let asyncSetMask = 0;
    const pWorker = applyMaskPaintStep({
      mesh,
      centers: [[2, 0.2, 0]],
      radius: 0.5,
      verticesInSphere: () => new Promise<Uint32Array>((r) => (resolveAsyncPaint = r)),
      views: spiedViews,
      tracker: asyncTracker,
      isAlive: () => true,
      getPrevMask: () => redoMask,
      setMask: () => {
        asyncSetMask++;
      },
      onStale: () => {},
    });
    // Worker reply lands in microtask before rAF commitFrame(), flipping status computing -> idle and notifying subscribers.
    asyncStatus = "idle";
    for (const fn of [...subs]) fn();
    resolveAsyncPaint(new Uint32Array([5]));
    expect(await pWorker).toBe(false);
    expect(asyncSetMask).toBe(0);
    asyncTracker.dispose();

    // Case F: Pre-existing unrelated store error (`status: "error"`) must NOT short-circuit
    // `#pendingAsyncHistory` while history status (`history.status` or `scene.status("history")`)
    // is `"computing"`.
    {
      let historyStatus = "idle";
      const errSubs = new Set<() => void>();
      const errStore = {
        getSnapshot: () => ({ status: "error", history: { status: historyStatus } }),
        subscribe(fn: () => void) {
          errSubs.add(fn);
          return () => errSubs.delete(fn);
        },
        undo() {
          historyStatus = "computing";
          for (const fn of [...errSubs]) fn();
        },
      };
      const trackerWithHistoryField = new MaskLiveTracker(scene, errStore);
      trackerWithHistoryField.attach();
      origTrust(mesh, redoMask.vertices, 6, true);
      errStore.undo();
      let resolveHistoryField!: (ids: Uint32Array) => void;
      let errSetMask = 0;
      const pHistoryField = applyMaskPaintStep({
        mesh,
        centers: [[2, 0.2, 0]],
        radius: 0.5,
        verticesInSphere: () => new Promise<Uint32Array>((r) => (resolveHistoryField = r)),
        views: spiedViews,
        tracker: trackerWithHistoryField,
        isAlive: () => true,
        getPrevMask: () => redoMask,
        setMask: () => {
          errSetMask++;
        },
        onStale: () => {},
      });
      historyStatus = "idle";
      for (const fn of [...errSubs]) fn();
      resolveHistoryField(new Uint32Array([5]));
      expect(await pHistoryField).toBe(false);
      expect(errSetMask).toBe(0);
      trackerWithHistoryField.dispose();

      // WorkerScene-style `scene.status("history")` + `scene.onStatus` alongside unrelated store error.
      let workerHistoryStatus = "idle";
      const workerStatusListeners = new Set<() => void>();
      const workerStyleScene = {
        views: (h: typeof mesh) => scene.views(h),
        onChanges: (fn: (changes: Parameters<Parameters<typeof scene.onChanges>[0]>[0]) => void) => scene.onChanges(fn),
        status: (key: string) => ({ status: key === "history" ? workerHistoryStatus : "error" }),
        onStatus: (fn: () => void) => {
          workerStatusListeners.add(fn);
          return () => workerStatusListeners.delete(fn);
        },
      };
      const workerStoreSubs = new Set<() => void>();
      const workerErrStore = {
        getSnapshot: () => ({ status: "error" }),
        subscribe(fn: () => void) {
          workerStoreSubs.add(fn);
          return () => workerStoreSubs.delete(fn);
        },
        undo() {
          workerHistoryStatus = "computing";
          for (const fn of [...workerStatusListeners]) fn();
        },
      };
      const workerTracker = new MaskLiveTracker(workerStyleScene, workerErrStore);
      workerTracker.attach();
      workerErrStore.undo();
      let resolveWorkerErr!: (ids: Uint32Array) => void;
      let workerErrSetMask = 0;
      const pWorkerErr = applyMaskPaintStep({
        mesh,
        centers: [[2, 0.2, 0]],
        radius: 0.5,
        verticesInSphere: () => new Promise<Uint32Array>((r) => (resolveWorkerErr = r)),
        views: spiedViews,
        tracker: workerTracker,
        isAlive: () => true,
        getPrevMask: () => redoMask,
        setMask: () => {
          workerErrSetMask++;
        },
        onStale: () => {},
      });
      workerHistoryStatus = "idle";
      for (const fn of [...workerStatusListeners]) fn();
      resolveWorkerErr(new Uint32Array([5]));
      expect(await pWorkerErr).toBe(false);
      expect(workerErrSetMask).toBe(0);
      workerTracker.dispose();
    }
  });
});
