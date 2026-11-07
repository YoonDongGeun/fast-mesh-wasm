// The stroke against the real engine (#174): the same path sampled at
// different event rates gives the same shape, the same relative stroke gives
// the same relative shape at any model size, one undo step per stroke on the
// sync scene, and a remesh in the middle of a grab keeps it continuous.
import { type RepairOp, Scene } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { GrabRegion, SculptStroke, sculptBrush, type SculptBrush } from "../src/viewport/sculpt";

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
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(dragAt(k / events)) }]);
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
          s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(drag) }]);
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
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(half) }]);
    // The live pass: splits only, here over the grabbed region.
    const slotsBefore = v.positions().length / 3;
    s.scene.repair(s.mesh, { op: "remesh", center: [R + 0.3, 0.1, 0], radius: 1, maxRounds: 2, splitOnly: true, targetLength: 0.02 });
    expect(v.positions().length / 3).toBeGreaterThan(slotsBefore);
    const settled = Float32Array.from(v.positions());
    region.rebase(v.positions(), v.indices());
    // No pop: the same drag writes the same positions back.
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(half) }]);
    expect(maxDifference(v.positions(), settled)).toBeLessThan(1e-3);
    // New vertices follow on: every vertex in the middle of the grabbed lump
    // moves with the drag, old and new.
    const further: Vec = [0.6, 0.2, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(further) }]);
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
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(far) }]);
    const slotsBefore = v.positions().length / 3;
    const r = s.scene.repair(s.mesh, { op: "splitStretched", center: [R + 1, 0, 0], radius: 4 });
    expect(r.splits).toBeGreaterThan(10);
    const settled = Float32Array.from(v.positions());
    region.rebase(v.positions(), v.indices(), r.vertices);
    // No pop: the same drag writes the same positions back.
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(far) }]);
    expect(maxDifference(v.positions(), settled)).toBeLessThan(1e-3);
    // Dragging on takes the new vertices along, each by its own part of the drag.
    const further: Vec = [3, 0, 0];
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(further) }]);
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
      s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(drag) }]);
      if (k % 4 === 0) {
        const r = s.scene.repair(s.mesh, { op: "splitStretched", center: [R, 0.025 * k, 0], radius: 4 });
        const settled = Float32Array.from(v.positions());
        region.rebase(v.positions(), v.indices(), r.vertices);
        s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(drag) }]);
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
    s.scene.edit(s.mesh, [{ op: "setPositions", vertices: region.ids, positions: region.positionsAt(drag) }]);
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

describe("GrabRegion", () => {
  const { positions, indices } = sphere(40, 80, R);

  it("captures_the_vertices_under_the_brush_with_their_weights", () => {
    const region = new GrabRegion(positions, indices, [R, 0, 0], 0.5, "smooth", 1);
    expect(region.ids.length).toBeGreaterThan(5);
    const out = region.positionsAt([0, 0, 0]);
    for (let k = 0; k < region.ids.length; k++) {
      const id = region.ids[k];
      expect(out[3 * k]).toBeCloseTo(positions[3 * id], 5);
      expect(Math.hypot(positions[3 * id] - R, positions[3 * id + 1], positions[3 * id + 2])).toBeLessThan(0.5);
    }
  });

  it("scales_the_drag_by_strength_and_falloff", () => {
    const full = new GrabRegion(positions, indices, [R, 0, 0], 0.5, "smooth", 1);
    const half = new GrabRegion(positions, indices, [R, 0, 0], 0.5, "smooth", 0.5);
    const a = full.positionsAt([1, 0, 0]);
    const b = half.positionsAt([1, 0, 0]);
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
