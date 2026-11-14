// The stroke against the real engine (#174): the same path sampled at
// different event rates gives the same shape, the same relative stroke gives
// the same relative shape at any model size, one undo step per stroke on the
// sync scene, and a remesh in the middle of a grab keeps it continuous.
import { type RepairOp, Scene } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { inheritedSplits, MASK_SOFT_BAND, maskEdgeField, protectMaskedDab } from "../src/viewport/mask-protect";
import { dabOperation, GRAB_MAX_STRETCH, GrabRegion, SculptStroke, sculptBrush, unionMaskIds, type SculptBrush } from "../src/viewport/sculpt";

type Vec = [number, number, number];

/** A UV sphere of radius `r`. */
function sphere(rings: number, segs: number, r: number) {
  const positions: number[] = [0, r, 0];
  const indices: number[] = [];
  for (let i = 1; i < rings; i++) {
    for (let j = 0; j < segs; j++) {
      const t = (Math.PI * i) / rings;
      const f = (2 * Math.PI * j) / segs;
      positions.push(r * Math.sin(t) * Math.cos(f), r * Math.cos(t), r * Math.sin(t) * Math.sin(f));
    }
  }
  positions.push(0, -r, 0);
  const at = (i: number, j: number) => (i === 0 ? 0 : i === rings ? 1 + (rings - 1) * segs : 1 + (i - 1) * segs + (j % segs));
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segs; j++) {
      if (i > 0) {
        indices.push(at(i, j), at(i, j + 1), at(i + 1, j));
      }
      if (i < rings - 1) {
        indices.push(at(i, j + 1), at(i + 1, j + 1), at(i + 1, j));
      }
    }
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/** The farthest any vertex is between two position buffers. */
function maxDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let v = 0; v < a.length / 3; v++) {
    worst = Math.max(worst, Math.hypot(a[3 * v] - b[3 * v], a[3 * v + 1] - b[3 * v + 1], a[3 * v + 2] - b[3 * v + 2]));
  }
  return worst;
}

const R = 2;

function setup(scale = 1) {
  const { positions, indices } = sphere(80, 160, R * scale);
  const scene = new Scene();
  const mesh = scene.addMesh(positions, indices);
  const apply = (ops: readonly RepairOp[]) => {
    for (const op of ops) {
      scene.repair(mesh, op);
    }
    return Promise.resolve();
  };
  return { scene, mesh, apply, start: positions };
}

/** A point on the sphere's equator; `u` runs from 0 to 1 along a curved path. */
const onPath = (u: number, scale = 1): Vec => {
  const a = 0.2 + u;
  return [scale * R * Math.cos(a), 0, scale * R * Math.sin(a)];
};

/** Runs an inflate stroke along the path, `events` pointer samples long. */
async function inflateStroke(events: number, scale = 1, wait: () => Promise<unknown> = () => Promise.resolve()) {
  const s = setup(scale);
  const brush: SculptBrush = sculptBrush("inflate", s.mesh, 0.5 * scale, 0.1, "smooth");
  const stroke = new SculptStroke(brush, {}, s.apply);
  for (let k = 0; k <= events; k++) {
    stroke.dab(onPath(k / events, scale));
    await wait();
  }
  stroke.end();
  await stroke.done;
  return { out: Float32Array.from(s.scene.views(s.mesh).positions()), start: s.start, dabs: stroke.dabs };
}

/** The grab drag at `u`: out and sideways, along a curve. */
const dragAt = (u: number): Vec => [u, Math.sin(u * Math.PI) * 0.5, 0];

/** Runs a grab stroke, `events` pointer samples long, through one captured region. */
async function grabStroke(events: number) {
  const s = setup();
  const v = s.scene.views(s.mesh);
  const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1);
  for (let k = 1; k <= events; k++) {
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(dragAt(k / events)) }]);
  }
  return { out: Float32Array.from(v.positions()), start: s.start };
}

describe("event-rate independence", () => {
  it("grab_gives_the_same_shape_at_any_event_rate", async () => {
    const reference = await grabStroke(240);
    // Absolute by construction: the last drag alone decides the shape.
    for (const events of [120, 30, 4, 1]) {
      const { out } = await grabStroke(events);
      expect(maxDifference(out, reference.out), `${events} events`).toBeLessThan(1e-5);
    }
    expect(maxDifference(reference.out, reference.start)).toBeCloseTo(1, 3);
  });

  it("inflate_gives_the_same_shape_within_two_percent_of_the_radius", async () => {
    const reference = await inflateStroke(480);
    const peak = maxDifference(reference.out, reference.start);
    expect(peak).toBeGreaterThan(0.1);
    // Curved path: coarser samples cut its corners by a little, nothing more.
    // (Before #174, 30 events lost 37% of the height and 10 lost 78%.)
    for (const events of [240, 120, 60, 30, 10]) {
      const { out, dabs } = await inflateStroke(events);
      expect(maxDifference(out, reference.out) / 0.5, `${events} events`).toBeLessThan(0.02);
      expect(Math.abs(dabs - reference.dabs), `${events} events`).toBeLessThanOrEqual(1);
    }
  });

  it("a_slow_scene_loses_no_dabs", async () => {
    const free = await inflateStroke(120);
    // Every applied dab takes a while: the pointer's samples queue behind it.
    const s = setup();
    const brush = sculptBrush("inflate", s.mesh, 0.5, 0.1, "smooth");
    let applies = 0;
    const stroke = new SculptStroke(brush, {}, async (ops) => {
      applies++;
      await new Promise((r) => setTimeout(r, 2));
      for (const op of ops) {
        s.scene.repair(s.mesh, op);
      }
    });
    for (let k = 0; k <= 120; k++) {
      stroke.dab(onPath(k / 120));
    }
    stroke.end();
    await stroke.done;
    expect(stroke.dabs).toBe(free.dabs);
    expect(applies).toBeLessThan(free.dabs);
    expect(maxDifference(s.scene.views(s.mesh).positions(), free.out)).toBeLessThan(1e-4);
  });
});

describe("model size", () => {
  it("the_same_relative_stroke_gives_the_same_relative_shape", async () => {
    // Before #174 the peak over the radius was 10.4, 0.57 and 0.007 at 0.01, 1 and 100.
    const unit = await inflateStroke(60, 1);
    const unitPeak = maxDifference(unit.out, unit.start) / 0.5;
    expect(unitPeak).toBeGreaterThan(0.2);
    for (const scale of [0.01, 100]) {
      const { out, start } = await inflateStroke(60, scale);
      const peak = maxDifference(out, start) / (0.5 * scale);
      expect(Math.abs(peak - unitPeak) / unitPeak, `scale ${scale}`).toBeLessThan(0.01);
    }
  });
});

describe("undo", () => {
  it("is_one_step_per_stroke_on_the_sync_scene", async () => {
    const s = setup();
    const before = Float32Array.from(s.scene.views(s.mesh).positions());
    const brush = sculptBrush("inflate", s.mesh, 0.5, 0.1, "smooth");
    const stroke = new SculptStroke(
      brush,
      { beginStep: () => s.scene.beginStep(), endStep: () => s.scene.endStep(), cancelStep: () => s.scene.cancelStep() },
      s.apply,
    );
    for (let k = 0; k <= 60; k++) {
      stroke.dab(onPath(k / 60));
    }
    stroke.end();
    await stroke.done;
    expect(stroke.dabs).toBeGreaterThan(5);
    expect(maxDifference(s.scene.views(s.mesh).positions(), before)).toBeGreaterThan(0.1);
    s.scene.undo();
    expect(maxDifference(s.scene.views(s.mesh).positions(), before)).toBe(0);
    expect(s.scene.canUndo()).toBe(false);
  });

  it("holds_for_a_grab_stroke_too", async () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const before = Float32Array.from(v.positions());
    const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1);
    const stroke = new SculptStroke(
      sculptBrush("grab", s.mesh, 0.5, 1, "smooth"),
      { beginStep: () => s.scene.beginStep(), endStep: () => s.scene.endStep(), cancelStep: () => s.scene.cancelStep() },
      s.apply,
    );
    for (let k = 1; k <= 20; k++) {
      const drag = dragAt(k / 20);
      stroke.move(
        () => {
          s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(drag) }]);
          return Promise.resolve();
        },
        [R + drag[0], drag[1], drag[2]],
      );
      await Promise.resolve();
      await Promise.resolve();
    }
    stroke.end();
    await stroke.done;
    expect(maxDifference(v.positions(), before)).toBeGreaterThan(0.5);
    s.scene.undo();
    expect(maxDifference(v.positions(), before)).toBe(0);
    expect(s.scene.canUndo()).toBe(false);
  });
});

describe("a remesh during a grab", () => {
  it("rebases_the_new_vertices_without_a_pop_and_keeps_the_shape", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const center: Vec = [R, 0, 0];
    const region = new GrabRegion(v.positions(), v.indices(), center, 0.5, "smooth", 1);
    const half: Vec = [0.3, 0.1, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(half) }]);
    // The live pass: splits only, here over the grabbed region.
    const slotsBefore = v.positions().length / 3;
    s.scene.repair(s.mesh, { op: "remesh", center: [R + 0.3, 0.1, 0], radius: 1, maxRounds: 2, splitOnly: true, targetLength: 0.02 });
    expect(v.positions().length / 3).toBeGreaterThan(slotsBefore);
    const settled = Float32Array.from(v.positions());
    region.rebase(v.positions(), v.indices());
    // No pop: the same drag writes the same positions back.
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(half) }]);
    expect(maxDifference(v.positions(), settled)).toBeLessThan(1e-3);
    // New vertices follow on: every vertex in the middle of the grabbed lump
    // moves with the drag, old and new.
    const further: Vec = [0.6, 0.2, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(further) }]);
    const now = v.positions();
    const slots = settled.length / 3;
    let followers = 0;
    for (let k = slotsBefore; k < slots; k++) {
      const moved = Math.hypot(now[3 * k] - settled[3 * k], now[3 * k + 1] - settled[3 * k + 1], now[3 * k + 2] - settled[3 * k + 2]);
      followers += moved > 0.01 ? 1 : 0;
    }
    expect(followers).toBeGreaterThan(0);
    // The tip reaches the drag, as it would have without the remesh.
    let tip = 0;
    for (let k = 0; k < slots; k++) {
      tip = Math.max(tip, now[3 * k]);
    }
    expect(tip).toBeCloseTo(R + 0.6, 1);
  });
});

describe("a live split during a long grab", () => {
  it("moves_the_new_vertices_with_the_region", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const center: Vec = [R, 0, 0];
    s.scene.repair(s.mesh, { op: "captureRestSizes" });
    const region = new GrabRegion(v.positions(), v.indices(), center, 0.5, "smooth", 1);
    // A drag several radii long: the lump's sides stretch far past the limit.
    const far: Vec = [2.5, 0, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(far) }]);
    const slotsBefore = v.positions().length / 3;
    const r = s.scene.repair(s.mesh, { op: "splitStretched", center: [R + 1, 0, 0], radius: 4 });
    expect(r.splits).toBeGreaterThan(10);
    const settled = Float32Array.from(v.positions());
    region.rebase(v.positions(), v.indices(), r.vertices);
    // No pop: the same drag writes the same positions back.
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(far) }]);
    expect(maxDifference(v.positions(), settled)).toBeLessThan(1e-3);
    // Dragging on takes the new vertices along, each by its own part of the drag.
    const further: Vec = [3, 0, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(further) }]);
    const now = v.positions();
    const slots = settled.length / 3;
    let moved = 0;
    for (let k = slotsBefore; k < slots; k++) {
      const by = now[3 * k] - settled[3 * k];
      expect(by).toBeLessThanOrEqual(0.5 + 1e-3);
      expect(by).toBeGreaterThanOrEqual(-1e-3);
      moved += by > 1e-3 ? 1 : 0;
    }
    // On the lump's sides, weight follows height: a new vertex's share of the
    // drag is about how far along the drag it sits.
    expect(moved / (slots - slotsBefore)).toBeGreaterThan(0.9);
  });
});

describe("a live split during a grab along the surface", () => {
  it("gives_new_vertices_their_edge_weights_so_the_mesh_does_not_explode", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const center: Vec = [R, 0, 0];
    s.scene.repair(s.mesh, { op: "captureRestSizes" });
    const before = v.indices().length / 3;
    const region = new GrabRegion(v.positions(), v.indices(), center, 0.5, "smooth", 1);
    // Along the surface the front folds over itself, so a position alone does
    // not say which part of the sheet a new vertex belongs to.
    for (let k = 1; k <= 60; k++) {
      const drag: Vec = [0, 0.05 * k, 0];
      s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(drag) }]);
      if (k % 4 === 0) {
        const r = s.scene.repair(s.mesh, { op: "splitStretched", center: [R, 0.025 * k, 0], radius: 4 });
        const settled = Float32Array.from(v.positions());
        region.rebase(v.positions(), v.indices(), r.vertices);
        s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(drag) }]);
        expect(maxDifference(v.positions(), settled)).toBeLessThan(1e-3);
      }
    }
    const after = v.indices().length / 3;
    // A single-sheet lump of about this much area needs a few thousand.
    expect(after - before).toBeLessThan(8000);
  });

  it("carries_every_split_vertex_with_a_moving_parent", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    s.scene.repair(s.mesh, { op: "captureRestSizes" });
    const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1);
    const drag: Vec = [0, 1.5, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(drag) }]);
    const moving = new Set(region.ids);
    const r = s.scene.repair(s.mesh, { op: "splitStretched", center: [R, 0.75, 0], radius: 4 });
    expect(r.vertices.length / 3).toBe(r.splits);
    expect(r.splits).toBeGreaterThan(0);
    region.rebase(v.positions(), v.indices(), r.vertices);
    const held = new Set(region.ids);
    let parented = 0;
    for (let k = 0; k < r.vertices.length; k += 3) {
      if (moving.has(r.vertices[k + 1]) || moving.has(r.vertices[k + 2])) {
        moving.add(r.vertices[k]);
        parented++;
        expect(held.has(r.vertices[k])).toBe(true);
      }
    }
    expect(parented).toBe(r.splits);
  });
});

describe("a grab on a mesh away from the world origin", () => {
  it("moves_only_by_the_drag", () => {
    const { positions, indices } = sphere(20, 40, R);
    const offset: Vec = [5, 3, 1];
    const away = positions.map((x, k) => x + offset[k % 3]);
    const scene = new Scene();
    const mesh = scene.addMesh(away, indices);
    const v = scene.views(mesh);
    const before = Float32Array.from(v.positions());
    const center: Vec = [before[3], before[4], before[5]];
    const region = new GrabRegion(v.positions(), v.indices(), center, 0.5, "smooth", 1, scene.origin(mesh));
    scene.edit(mesh, [{ op: "setPositions", ...region.at([0, 0, 0]) }]);
    expect(maxDifference(v.positions(), before)).toBeLessThan(1e-5);
    scene.edit(mesh, [{ op: "setPositions", ...region.at([0, 0.2, 0]) }]);
    expect(maxDifference(v.positions(), before)).toBeCloseTo(0.2, 3);
  });
});

describe("a remesh during a long grab", () => {
  it("new_vertices_stay_with_their_neighbours_past_the_drag_where_the_warp_folds", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    s.scene.repair(s.mesh, { op: "captureRestSizes" });
    const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1);
    // Three radii out: the warp folds here, so no position maps back to one rest.
    const far: Vec = [1.5, 0, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(far) }]);
    const slots = v.positions().length / 3;
    const r = s.scene.repair(s.mesh, { op: "splitStretched", center: [R + 0.8, 0, 0], radius: 3 });
    expect(v.positions().length / 3).toBeGreaterThan(slots);
    region.rebase(v.positions(), v.indices(), r.vertices);
    const further: Vec = [2.5, 0, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at(further) }]);
    // Each new vertex stays between its neighbours as the lump goes on.
    const positions = v.positions();
    const indices = v.indices();
    const sums = new Float64Array((positions.length / 3) * 4);
    for (let t = 0; t + 2 < indices.length; t += 3) {
      const corners = [indices[t], indices[t + 1], indices[t + 2]];
      for (const a of corners) {
        for (const b of corners) {
          if (a !== b) {
            for (let c = 0; c < 3; c++) {
              sums[4 * a + c] += positions[3 * b + c];
            }
            sums[4 * a + 3]++;
          }
        }
      }
    }
    const offAlongDrag = (from: number, to: number) => {
      let worst = 0;
      for (let k = from; k < to; k++) {
        if (sums[4 * k + 3] > 0) {
          worst = Math.max(worst, Math.abs(sums[4 * k] / sums[4 * k + 3] - positions[3 * k]));
        }
      }
      return worst;
    };
    const fresh = offAlongDrag(slots, positions.length / 3);
    const grabbed = offAlongDrag(0, slots);
    // A steep lump puts every vertex off its neighbours: the new ones must not
    // stand out from the old (weights from going back through the drag: three times).
    expect(fresh).toBeLessThan(1.5 * grabbed);
  });
});

describe("a grab released while a live pass is out", () => {
  it("takes_the_new_vertices_along", async () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1);
    // As on the Worker: requests run in the order sent, and answer later.
    const fifo: (() => void)[] = [];
    const send = (work: () => void) =>
      new Promise<void>((resolve) =>
        fifo.push(() => {
          work();
          resolve();
        }),
      );
    const run = async () => {
      while (fifo.length > 0) {
        fifo.shift()?.();
        await new Promise((r) => setTimeout(r, 0));
      }
    };
    s.scene.repair(s.mesh, { op: "captureRestSizes" });
    const stroke = new SculptStroke(sculptBrush("grab", s.mesh, 0.5, 1, "smooth"), {}, () => Promise.resolve());
    const moveTo = (drag: Vec) =>
      stroke.move(
        () => {
          // The ids and positions are read when it is sent, not when it runs.
          const { vertices, positions } = region.at(drag);
          return send(() => s.scene.edit(s.mesh, [{ op: "setPositions", vertices, positions }]));
        },
        [R + drag[0], drag[1], drag[2]],
      );
    moveTo([0.3, 0, 0]);
    await run();
    const slots = v.positions().length / 3;
    let splits: ArrayLike<number> | undefined;
    void stroke.pass(() =>
      send(() => {
        splits = s.scene.repair(s.mesh, { op: "splitStretched", center: [R, 0, 0], radius: 2 }).vertices;
      }).then(() => region.rebase(v.positions(), v.indices(), splits)),
    );
    // The drag goes on during the pass, and the pointer lets go.
    moveTo([0.9, 0, 0]);
    stroke.flush();
    stroke.end();
    await run();
    await stroke.done;
    await run();
    const now = v.positions();
    expect(now.length / 3).toBeGreaterThan(slots);
    // The tip of the lump is new vertices as well as old: both reach the drag.
    let oldTip = 0;
    let newTip = 0;
    for (let k = 0; k < now.length / 3; k++) {
      if (k < slots) {
        oldTip = Math.max(oldTip, now[3 * k]);
      } else {
        newTip = Math.max(newTip, now[3 * k]);
      }
    }
    expect(oldTip).toBeGreaterThan(R + 0.8);
    // Without the rebase behind the pass they stay where the split left them.
    expect(newTip).toBeGreaterThan(R + 0.4);

  });
});

describe("GrabRegion over a mask", () => {
  /** A mask over the sphere's slots with x > 0.2, and what the stroke holds it with. */
  function masked(positions: ArrayLike<number>, indices: ArrayLike<number>) {
    const slots = positions.length / 3;
    const ids: number[] = [];
    for (let v = 0; v < slots; v++) {
      if (positions[3 * v + 2] > 0) {
        ids.push(v);
      }
    }
    return { ids, field: maskEdgeField(ids, positions, indices, slots, 0.1) };
  }

  it("leaves_masked_vertices_byte_identical_through_a_grab", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const { ids, field } = masked(v.positions(), v.indices());
    const before = Float32Array.from(v.positions());
    const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1, [0, 0, 0], field);
    expect(region.ids.length).toBeGreaterThan(0);
    for (let k = 1; k <= 8; k++) {
      s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at([0, 0.05 * k, 0]) }]);
      const now = new Uint8Array(v.positions().buffer, v.positions().byteOffset, v.positions().byteLength);
      const was = new Uint8Array(before.buffer);
      for (const id of ids) {
        for (let b = 12 * id; b < 12 * id + 12; b++) {
          expect(now[b]).toBe(was[b]);
        }
      }
    }
    expect(Array.from(v.positions())).not.toEqual(Array.from(before));
  });

  it("scales_the_soft_band_and_keeps_it_through_a_rebase", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const { ids, field } = masked(v.positions(), v.indices());
    const region = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1, [0, 0, 0], field);
    const free = new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1);
    const held = new Set(ids);
    expect(Array.from(region.ids).some((id) => held.has(id))).toBe(false);
    expect(region.ids.length).toBeLessThan(free.ids.length);
    region.at([0, 0.2, 0]).positions;
    s.scene.edit(s.mesh, [{ op: "setPositions", ...region.at([0, 0.2, 0]) }]);
    region.rebase(v.positions(), v.indices());
    expect(Array.from(region.ids).some((id) => held.has(id))).toBe(false);
  });

  it("takes_nothing_when_the_press_is_fully_masked", () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    const slots = v.positions().length / 3;
    const all = Array.from({ length: slots }, (_, i) => i);
    const field = maskEdgeField(all, v.positions(), v.indices(), slots, 0.1);
    expect(new GrabRegion(v.positions(), v.indices(), [R, 0, 0], 0.5, "smooth", 1, [0, 0, 0], field).ids.length).toBe(0);
  });
});

describe("GrabRegion", () => {
  const { positions, indices } = sphere(40, 80, R);

  it("captures_the_vertices_under_the_brush_with_their_weights", () => {
    const region = new GrabRegion(positions, indices, [R, 0, 0], 0.5, "smooth", 1);
    expect(region.ids.length).toBeGreaterThan(5);
    const out = region.at([0, 0, 0]).positions;
    for (let k = 0; k < region.ids.length; k++) {
      const id = region.ids[k];
      expect(out[3 * k]).toBeCloseTo(positions[3 * id], 5);
      expect(Math.hypot(positions[3 * id] - R, positions[3 * id + 1], positions[3 * id + 2])).toBeLessThan(0.5);
    }
  });

  it("scales_the_drag_by_strength_and_falloff", () => {
    const full = new GrabRegion(positions, indices, [R, 0, 0], 0.5, "smooth", 1);
    const half = new GrabRegion(positions, indices, [R, 0, 0], 0.5, "smooth", 0.5);
    const a = full.at([1, 0, 0]).positions;
    const b = half.at([1, 0, 0]).positions;
    expect(full.ids).toEqual(half.ids);
    for (let k = 0; k < full.ids.length; k++) {
      const rest = positions[3 * full.ids[k]];
      expect(b[3 * k] - rest).toBeCloseTo((a[3 * k] - rest) / 2, 5);
    }
  });

  it("is_empty_away_from_the_surface", () => {
    expect(new GrabRegion(positions, indices, [50, 0, 0], 0.5, "smooth", 1).ids.length).toBe(0);
  });
});

describe("a grab never folds the surface", () => {
  // A flat sheet with its normals along +z: a triangle turned over points the other way.
  const sheet = (n: number, size: number) => {
    const positions: number[] = [];
    const indices: number[] = [];
    for (let j = 0; j <= n; j++) {
      for (let i = 0; i <= n; i++) {
        positions.push((i / n - 0.5) * size, (j / n - 0.5) * size, 0);
      }
    }
    const at = (i: number, j: number) => j * (n + 1) + i;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        indices.push(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1));
      }
    }
    return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
  };
  const moved = (s: { positions: Float32Array }, region: GrabRegion, drag: [number, number, number]) => {
    const out = Float32Array.from(s.positions);
    const { vertices, positions } = region.at(drag);
    vertices.forEach((v, k) => {
      out[3 * v] = positions[3 * k];
      out[3 * v + 1] = positions[3 * k + 1];
      out[3 * v + 2] = positions[3 * k + 2];
    });
    return out;
  };
  const turnedOver = (p: Float32Array, indices: Uint32Array) => {
    let count = 0;
    for (let t = 0; t < indices.length; t += 3) {
      const [a, b, c] = [indices[t], indices[t + 1], indices[t + 2]];
      const area =
        (p[3 * b] - p[3 * a]) * (p[3 * c + 1] - p[3 * a + 1]) - (p[3 * b + 1] - p[3 * a + 1]) * (p[3 * c] - p[3 * a]);
      if (area <= 0) count++;
    }
    return count;
  };

  it.each(["smooth", "linear"] as const)("along the surface, at any length (%s)", (falloff) => {
    const s = sheet(80, 8);
    const radius = 0.5;
    for (const length of [0.2, 0.5, 1, 2, 3]) {
      const region = new GrabRegion(s.positions, s.indices, [0, 0, 0], radius, falloff, 1);
      expect(turnedOver(moved(s, region, [length, 0, 0]), s.indices)).toBe(0);
    }
  });

  it("the old fixed radius does fold at these lengths", () => {
    // What the widening is for: smoothstep over a fixed 0.5 radius, dragged 1.
    const s = sheet(80, 8);
    const out = Float32Array.from(s.positions);
    for (let v = 0; v < out.length / 3; v++) {
      const t = Math.hypot(out[3 * v], out[3 * v + 1]) / 0.5;
      if (t < 1) {
        const u = 1 - t;
        out[3 * v] += u * u * (3 - 2 * u);
      }
    }
    expect(turnedOver(out, s.indices)).toBeGreaterThan(0);
  });

  it("the press follows the drag, and a short drag keeps the brush radius", () => {
    const s = sheet(80, 8);
    const region = new GrabRegion(s.positions, s.indices, [0, 0, 0], 0.5, "smooth", 1);
    expect(region.influence([0.1, 0, 0])).toBe(0.5);
    expect(region.influence([2, 0, 0])).toBeCloseTo((1.5 * 2) / GRAB_MAX_STRETCH, 5);
    const out = moved(s, region, [2, 1, 0]);
    const center = 40 * 81 + 40;
    expect(out[3 * center]).toBeCloseTo(2, 5);
    expect(out[3 * center + 1]).toBeCloseTo(1, 5);
  });

  it("depends on the total drag alone, also after it comes back", () => {
    const s = sheet(40, 8);
    const region = new GrabRegion(s.positions, s.indices, [0, 0, 0], 0.5, "smooth", 1);
    moved(s, region, [3, 0, 0]);
    const back = moved(s, region, [0.1, 0, 0]);
    const direct = moved(s, new GrabRegion(s.positions, s.indices, [0, 0, 0], 0.5, "smooth", 1), [0.1, 0, 0]);
    expect(Array.from(back)).toEqual(Array.from(direct));
  });

  it("pulling straight off the surface keeps the brush radius, however far", () => {
    const s = sheet(80, 8);
    const region = new GrabRegion(s.positions, s.indices, [0, 0, 0], 0.5, "smooth", 1);
    expect(region.influence([0, 0, 3])).toBe(0.5);
  });

  it("surface ahead of a pull off it widens the influence", () => {
    // A wall ahead of the press, at x = 1, which the lump's front would run over.
    const s = sheet(40, 8);
    const wall = s.positions.length / 3;
    const positions = Float32Array.from([...s.positions, 1, 0, 2, 1, 1, 2, 1, 0, 3]);
    const indices = Uint32Array.from([...s.indices, wall, wall + 1, wall + 2]);
    const region = new GrabRegion(positions, indices, [0, 0, 2], 0.5, "smooth", 1);
    expect(region.influence([3, 0, 0])).toBeGreaterThan(0.5);
  });

  it("a vertex added mid-drag stays where it landed, then moves with its edge", () => {
    const s = sheet(40, 8);
    const region = new GrabRegion(s.positions, s.indices, [0, 0, 0], 0.5, "smooth", 1);
    const drag: [number, number, number] = [1.5, 0, 0];
    const landed = moved(s, region, drag);
    const [a, b] = [20 * 41 + 20, 20 * 41 + 21];
    const fresh = landed.length / 3;
    const at = [0, 1, 2].map((c) => (landed[3 * a + c] + landed[3 * b + c]) / 2);
    const positions = Float32Array.from([...landed, ...at]);
    const indices = Uint32Array.from([...s.indices, fresh, a, b]);
    region.rebase(positions, indices, [fresh, a, b]);
    const read = (edit: { vertices: Uint32Array; positions: Float64Array }, v: number) => {
      const k = Array.from(edit.vertices).indexOf(v);
      expect(k).toBeGreaterThanOrEqual(0);
      return Array.from(edit.positions.subarray(3 * k, 3 * k + 3));
    };
    const same = read(region.at(drag), fresh);
    for (const c of [0, 1, 2]) {
      expect(same[c]).toBeCloseTo(at[c], 4);
    }
    const further = region.at([1.8, 0, 0]);
    const [pa, pb, pf] = [read(further, a), read(further, b), read(further, fresh)];
    // Its share of the extra drag is its edge's mean, whatever the widening does.
    expect(pf[0] - at[0]).toBeCloseTo((pa[0] - landed[3 * a] + (pb[0] - landed[3 * b])) / 2, 4);
  });
});

describe("engine-driven Grab + masked dabs + live splits (#177, #192, #197)", () => {
  it("preserves_masked_and_inherited_split_vertices_and_undoes_in_one_Ctrl_Z_step_on_sync_scene", async () => {
    const s = setup();
    const v = s.scene.views(s.mesh);
    s.scene.repair(s.mesh, { op: "captureRestSizes", onlyIfMissing: true });

    // Paint mask via engine BVH verticesInSphere near [R, 0.25, 0.25]
    let masked = s.scene.verticesInSphere(s.mesh, [R, 0.25, 0.25], 0.35);
    expect(masked.length).toBeGreaterThan(5);
    let inherited = new Set<number>();

    const before = Float32Array.from(v.positions());
    const press: Vec = [R, 0, 0];
    const radius = 0.6;
    const stroke = new SculptStroke(
      sculptBrush("grab", s.mesh, radius, 1, "smooth"),
      {
        beginStep: () => s.scene.beginStep(),
        endStep: () => s.scene.endStep(),
        cancelStep: () => s.scene.cancelStep(),
      },
      () => Promise.resolve(),
    );

    // Press queues captureStrokeState in-order before moves, without opening an undo step on its own.
    void stroke.pass(() => {
      s.scene.repair(s.mesh, { op: "captureStrokeState", onlyIfMissing: false });
      return Promise.resolve();
    });

    const moveGrab = (drag: Vec) =>
      stroke.move(
        () => {
          const curPos = v.positions();
          const live = new Uint8Array(curPos.length / 3).fill(1);
          const grabOp = dabOperation(
            { ...sculptBrush("grab", s.mesh, radius, 1, "smooth"), direction: drag },
            press,
          );
          const heldOp = protectMaskedDab(grabOp, masked, curPos, live);
          s.scene.repair(s.mesh, heldOp);
          return Promise.resolve();
        },
        [press[0] + drag[0], press[1] + drag[1], press[2] + drag[2]],
      );

    moveGrab([0.6, 0, 0]);
    await stroke.pass(() => {
      const rep = s.scene.repair(s.mesh, { op: "splitStretched", center: [R + 0.3, 0, 0], radius: 2 });
      const set = new Set(masked);
      const fresh = inheritedSplits((id) => set.has(id), rep.vertices);
      if (fresh.length > 0) {
        masked = unionMaskIds(masked, Uint32Array.from(fresh).sort((a, b) => a - b));
        inherited = new Set([...inherited, ...fresh]);
      }
      return Promise.resolve();
    });
    const afterSplitPos = Float32Array.from(v.positions());
    const splitSlots = afterSplitPos.length / 3;
    expect(splitSlots).toBeGreaterThan(before.length / 3);

    moveGrab([1.2, 0.2, 0]);
    stroke.end();
    await stroke.done;

    const finalPos = v.positions();
    expect(maxDifference(finalPos.subarray(0, before.length), before)).toBeGreaterThan(0.8);

    // Every original masked vertex stayed byte-identical throughout the grab + live split.
    const wasBytes = new Uint8Array(before.buffer);
    const nowBytes = new Uint8Array(finalPos.buffer, finalPos.byteOffset, finalPos.byteLength);
    for (const id of masked) {
      if (3 * id < before.length) {
        for (let b = 12 * id; b < 12 * id + 12; b++) {
          expect(nowBytes[b]).toBe(wasBytes[b]);
        }
      } else {
        // Inherited split midpoint stayed byte-identical after its split created it.
        for (let c = 0; c < 3; c++) {
          expect(finalPos[3 * id + c]).toBe(afterSplitPos[3 * id + c]);
        }
      }
    }

    // Band parameter check
    expect(MASK_SOFT_BAND).toBeGreaterThan(0);

    // Single Ctrl+Z undoes the entire grab stroke + live splits back to original mesh!
    expect(s.scene.canUndo()).toBe(true);
    s.scene.undo();
    expect(s.scene.vertexCount(s.mesh)).toBe(before.length / 3);
    expect(v.positions().length).toBeGreaterThanOrEqual(before.length);
    expect(maxDifference(v.positions().subarray(0, before.length), before)).toBe(0);
    expect(s.scene.canUndo()).toBe(false);
  });
});
