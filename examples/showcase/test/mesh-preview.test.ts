import { describe, expect, it } from "vitest";
import { PREVIEW_SIZE, PreviewQueue, type PreviewHost, PreviewRender, type PreviewSource } from "../src/panels/mesh-preview";

/** A closed axis-aligned box, its corners about `at`. */
function box(at: [number, number, number], half: [number, number, number]) {
  const positions = new Float32Array(24);
  for (let i = 0; i < 8; i++) {
    positions[3 * i] = at[0] + (i & 1 ? half[0] : -half[0]);
    positions[3 * i + 1] = at[1] + (i & 2 ? half[1] : -half[1]);
    positions[3 * i + 2] = at[2] + (i & 4 ? half[2] : -half[2]);
  }
  const quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
  const indices = new Uint32Array(quads.flatMap(([a, b, c, d]) => [a, b, c, a, c, d]));
  return { positions, indices };
}

function source(mesh: { positions: Float32Array; indices: Uint32Array }, color = 0x6ea8d9, matrix?: number[]): PreviewSource {
  return { views: () => ({ positions: () => mesh.positions, indices: () => mesh.indices }), color, matrix };
}

function draw(...sources: PreviewSource[]): Uint8ClampedArray {
  const r = new PreviewRender(sources);
  expect(r.step(Infinity)).toBe(true); // an infinite budget finishes in one call
  const image = r.image();
  if (image === null) {
    throw new Error("no image");
  }
  return image;
}

const coverage = (image: Uint8ClampedArray) => {
  let n = 0;
  for (let i = 3; i < image.length; i += 4) {
    n += image[i] > 0 ? 1 : 0;
  }
  return n / (image.length / 4);
};

describe("PreviewRender", () => {
  it("draws a framed, shaded mesh of the asked size on a transparent background", () => {
    const image = draw(source(box([100, -50, 7], [1, 1, 1])));
    expect(image.length).toBe(4 * PREVIEW_SIZE * PREVIEW_SIZE);
    // Framed whatever its position and scale: it fills most of the image.
    expect(coverage(image)).toBeGreaterThan(0.4);
    expect(coverage(image)).toBeLessThan(0.95);
    expect(image[3]).toBe(0); // a corner stays empty
    // Three visible faces shade differently.
    const shades = new Set<number>();
    for (let i = 0; i < image.length; i += 4) {
      if (image[i + 3] === 255) {
        shades.add(image[i]);
      }
    }
    expect(shades.size).toBeGreaterThanOrEqual(3);
  });

  it("frames by the live triangles: removed triangles and dead vertex slots are ignored", () => {
    const live = box([0, 0, 0], [1, 1, 1]);
    const positions = new Float32Array(27);
    positions.set(live.positions);
    positions.set([1e6, 1e6, 1e6], 24); // a dead slot's stale position
    const indices = new Uint32Array(live.indices.length + 3); // a removed triangle reads 0, 0, 0
    indices.set(live.indices);
    expect(draw(source({ positions, indices }))).toEqual(draw(source(live)));
  });

  it("gives different pictures for different shapes", () => {
    const a = draw(source(box([0, 0, 0], [1, 1, 1])));
    const b = draw(source(box([0, 0, 0], [3, 0.2, 0.4])));
    let diff = 0;
    for (let i = 0; i < a.length; i++) {
      diff += Math.abs(a[i] - b[i]);
    }
    expect(diff / a.length).toBeGreaterThan(10);
  });

  it("is bit-identical however it is sliced, and re-acquires its views each slice", () => {
    const mesh = box([0, 0, 0], [1, 2, 3]);
    let acquired = 0;
    const counted: PreviewSource = {
      color: 0x8fc46e,
      views: () => {
        acquired++;
        return { positions: () => mesh.positions, indices: () => mesh.indices };
      },
    };
    const sliced = new PreviewRender([counted]);
    let steps = 1;
    while (!sliced.step(0)) {
      steps++;
    }
    expect(sliced.image()).toEqual(draw(source(mesh, 0x8fc46e)));
    expect(acquired).toBeGreaterThanOrEqual(steps);
  });

  it("stops a slice on its budget at a large mesh", () => {
    const n = 200_000;
    const positions = new Float32Array(9 * n);
    const indices = new Uint32Array(3 * n);
    for (let t = 0; t < n; t++) {
      const x = (t % 500) / 500;
      const y = Math.floor(t / 500) / 400;
      positions.set([x, y, 0, x + 0.002, y, 0, x, y + 0.002, 0], 9 * t);
      indices.set([3 * t, 3 * t + 1, 3 * t + 2], 3 * t);
    }
    const r = new PreviewRender([source({ positions, indices })]);
    let steps = 1;
    while (!r.step(0)) {
      steps++;
    }
    expect(steps).toBeGreaterThan(4); // two passes, each cut into chunks
    expect(r.triangles).toBe(n);
  });

  it("composes several meshes by their placements and colours", () => {
    const tx = (x: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1];
    const unit = box([0, 0, 0], [1, 1, 1]);
    const image = draw(source(unit, 0xff0000, tx(-3)), source(unit, 0x0000ff, tx(3)));
    let red = 0;
    let blue = 0;
    for (let i = 0; i < image.length; i += 4) {
      if (image[i + 3] === 255) {
        red += image[i] > image[i + 2] ? 1 : 0;
        blue += image[i + 2] > image[i] ? 1 : 0;
      }
    }
    expect(red).toBeGreaterThan(100);
    expect(blue).toBeGreaterThan(100);
  });

  it("has no image without a drawable triangle: empty, flat to a point, or not finite", () => {
    const empty = new PreviewRender([source({ positions: new Float32Array(0), indices: new Uint32Array(0) })]);
    empty.step(Infinity);
    expect(empty.image()).toBeNull();
    const nan = new PreviewRender([
      source({ positions: new Float32Array([NaN, 0, 0, 1, 0, 0, 0, Infinity, 0]), indices: new Uint32Array([0, 1, 2]) }),
    ]);
    nan.step(Infinity);
    expect(nan.image()).toBeNull();
    const point = new PreviewRender([source({ positions: new Float32Array(9).fill(2), indices: new Uint32Array([0, 1, 2]) })]);
    point.step(Infinity);
    expect(point.image()).toBeNull();
  });
});

/** A host driven by hand: virtual time, explicit timers and idle callbacks. */
function fakeHost() {
  let time = 0;
  let timers: { at: number; cb: () => void; id: number }[] = [];
  let idles: ((budget: number) => void)[] = [];
  let next = 1;
  const state = { held: false };
  const host: PreviewHost = {
    held: () => state.held,
    now: () => time,
    setTimer: (cb, ms) => {
      timers.push({ at: time + ms, cb, id: next });
      return next++;
    },
    clearTimer: (id) => {
      timers = timers.filter((t) => t.id !== id);
    },
    idle: (cb) => {
      idles.push(cb);
    },
  };
  return {
    host,
    state,
    advance(ms: number) {
      time += ms;
      for (const t of timers.filter((t) => t.at <= time)) {
        timers = timers.filter((o) => o !== t);
        t.cb();
      }
    },
    runIdle() {
      const now = idles;
      idles = [];
      for (const cb of now) {
        cb(3);
      }
      return now.length;
    },
  };
}

/** A render that takes `slices` steps. */
function fakeJob(slices: number) {
  const job = {
    started: 0,
    finished: 0,
    start: () => {
      job.started++;
      let left = slices;
      return { step: () => --left <= 0 } as unknown as PreviewRender;
    },
    done: () => {
      job.finished++;
    },
  };
  return job;
}

describe("PreviewQueue", () => {
  it("starts nothing until the edits go quiet, however many arrive", () => {
    const f = fakeHost();
    const q = new PreviewQueue(f.host);
    const job = fakeJob(1);
    for (let i = 0; i < 100; i++) {
      q.invalidate("m", job, 400);
      f.advance(16); // a stroke's frames
      f.runIdle();
    }
    expect(job.started).toBe(0);
    f.advance(400);
    f.runIdle();
    expect(job.started).toBe(1);
    expect(job.finished).toBe(1);
  });

  it("starts and draws nothing while held, however long, then carries on", () => {
    const f = fakeHost();
    const q = new PreviewQueue(f.host);
    const job = fakeJob(2);
    f.state.held = true; // a press on the viewport: edits can be further apart than the quiet time
    q.invalidate("m", job, 400);
    for (let i = 0; i < 50; i++) {
      f.advance(100);
      f.runIdle();
    }
    expect(job.started).toBe(0);
    f.state.held = false;
    f.advance(100);
    f.runIdle(); // the first slice
    expect(job.started).toBe(1);
    f.state.held = true;
    for (let i = 0; i < 50; i++) {
      f.advance(100);
      f.runIdle();
    }
    expect(job.finished).toBe(0); // a render under way waits too
    f.state.held = false;
    f.advance(100);
    f.runIdle();
    expect(job.finished).toBe(1);
  });

  it("does one slice per idle callback, and restarts a render an edit overtook", () => {
    const f = fakeHost();
    const q = new PreviewQueue(f.host);
    const job = fakeJob(3);
    q.invalidate("m", job, 0);
    f.advance(0);
    f.runIdle();
    expect(job.started).toBe(1);
    expect(job.finished).toBe(0);
    q.invalidate("m", job, 100); // an edit mid-render: the old picture is stale
    f.runIdle();
    f.advance(100);
    f.runIdle();
    f.runIdle();
    f.runIdle();
    expect(job.started).toBe(2);
    expect(job.finished).toBe(1);
  });

  it("renders one mesh at a time, and a cancelled mesh never starts", () => {
    const f = fakeHost();
    const q = new PreviewQueue(f.host);
    const a = fakeJob(2);
    const b = fakeJob(1);
    const c = fakeJob(1);
    q.invalidate("a", a, 0);
    q.invalidate("b", b, 0);
    q.invalidate("c", c, 0);
    q.cancel("c");
    f.advance(0);
    f.runIdle();
    expect([a.started, b.started]).toEqual([1, 0]);
    f.runIdle();
    expect(a.finished).toBe(1);
    f.advance(0);
    f.runIdle();
    f.advance(0);
    f.runIdle();
    expect(b.finished).toBe(1);
    expect(c.started).toBe(0);
  });

  it("drops a render whose views throw (the mesh went away) and goes on", () => {
    const f = fakeHost();
    const q = new PreviewQueue(f.host);
    const bad = {
      start: () =>
        ({
          step: () => {
            throw new Error("gone");
          },
        }) as unknown as PreviewRender,
      done: () => {
        throw new Error("must not finish");
      },
    };
    const good = fakeJob(1);
    q.invalidate("bad", bad, 0);
    q.invalidate("good", good, 0);
    f.advance(0);
    f.runIdle();
    f.advance(0);
    f.runIdle();
    expect(good.finished).toBe(1);
  });
});
