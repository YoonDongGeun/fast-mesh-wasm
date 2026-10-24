// The stepped mesh adds (`meshAdds`, internal to the Worker engine): a
// chunked upload of the inputs, a stepped store build, then the BVH and
// normals in steps — the same mesh as `addMesh`, bit for bit, or the same
// error (issue #64).
import { describe, expect, it } from "vitest";
import { FastMeshError, Scene } from "../src/node.js";
import { meshAdds } from "../src/scene.js";

/** A wavy `n × n` terrain grid. */
function terrain(n: number): { positions: Float64Array; indices: Uint32Array } {
  const positions = new Float64Array(n * n * 3);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = (4 * i) / (n - 1);
      const y = (4 * j) / (n - 1);
      positions.set([x, y, 0.3 * Math.sin(1.7 * x) * Math.cos(1.3 * y)], (j * n + i) * 3);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let at = 0;
  for (let j = 0; j + 1 < n; j++) {
    for (let i = 0; i + 1 < n; i++) {
      const a = j * n + i;
      indices.set([a, a + 1, a + n + 1, a, a + n + 1, a + n], at);
      at += 6;
    }
  }
  return { positions, indices };
}

const T = terrain(12);

function errorOf(fn: () => unknown): { code: string; message: string } {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    const f = e as FastMeshError;
    return { code: f.code, message: f.message };
  }
  throw new Error("expected a throw");
}

const bytesOf = (v: ArrayBufferView) => Buffer.from(v.buffer, v.byteOffset, v.byteLength);

describe("stepped mesh adds over a chunked upload (issue #64)", () => {
  it("upload_steps_and_finish_match_addMesh", () => {
    const origins: ([number, number, number] | undefined)[] = [undefined, [1, 2, 3]];
    for (const origin of origins) {
      for (const budget of [1, 7, 100_000]) {
        // Chunk sizes deliberately not multiples of 3.
        for (const chunk of [1, 7, 100]) {
          const s = new Scene();
          try {
            const options = origin === undefined ? {} : { origin };
            const ref = s.addMesh(T.positions, T.indices, options);
            const rv = s.views(ref);
            const want = {
              positions: rv.positions().slice(),
              normals: rv.normals()?.slice() ?? null,
              indices: rv.indices().slice(),
              matrix: s.matrix4(ref).slice(),
            };
            const adds = meshAdds(s);
            const token = adds.beginUpload(T.positions.length / 3, T.indices.length / 3, options);
            for (let at = 0; at < T.positions.length; at += chunk) {
              adds.pushPositions(token, T.positions.subarray(at, at + chunk));
            }
            for (let at = 0; at < T.indices.length; at += chunk) {
              adds.pushIndices(token, T.indices.subarray(at, at + chunk));
            }
            while (!adds.step(token, budget)) {}
            const bufs = adds.buffers(token);
            expect(bytesOf(bufs.positions).equals(bytesOf(want.positions))).toBe(true);
            expect(bufs.normals === null).toBe(want.normals === null);
            if (bufs.normals !== null && want.normals !== null) {
              expect(bytesOf(bufs.normals).equals(bytesOf(want.normals))).toBe(true);
            }
            expect(bytesOf(bufs.indices).equals(bytesOf(want.indices))).toBe(true);
            const m = adds.finish(token);
            const v = s.views(m);
            expect(bytesOf(v.positions()).equals(bytesOf(want.positions))).toBe(true);
            expect(v.normals() === null).toBe(want.normals === null);
            if (v.normals() !== null && want.normals !== null) {
              expect(bytesOf(v.normals() as Float32Array).equals(bytesOf(want.normals))).toBe(true);
            }
            expect(bytesOf(v.indices()).equals(bytesOf(want.indices))).toBe(true);
            expect(bytesOf(s.matrix4(m)).equals(bytesOf(want.matrix))).toBe(true);
          } finally {
            s.free();
          }
        }
      }
    }
  });

  it("a_bad_mesh_fails_a_step_like_addMesh", () => {
    const badIndex = { positions: T.positions, indices: T.indices.slice() };
    badIndex.indices[100] = 999_999;
    const badPositions = { positions: T.positions.slice(), indices: T.indices };
    badPositions.positions[100] = Number.NaN;
    for (const bad of [badIndex, badPositions]) {
      const s = new Scene();
      try {
        const sync = errorOf(() => s.addMesh(bad.positions, bad.indices));
        const adds = meshAdds(s);
        const token = adds.beginUpload(bad.positions.length / 3, bad.indices.length / 3, {});
        adds.pushPositions(token, bad.positions);
        adds.pushIndices(token, bad.indices);
        const err = errorOf(() => {
          while (!adds.step(token, 7)) {}
        });
        expect(err).toEqual(sync);
        // The failed build is dropped.
        expect(errorOf(() => adds.step(token, 7)).message).toMatch("no mesh is being added");
      } finally {
        s.free();
      }
    }
  });

  it("buffers_before_the_store_is_built_throws", () => {
    const s = new Scene();
    try {
      const adds = meshAdds(s);
      const token = adds.beginUpload(T.positions.length / 3, T.indices.length / 3, {});
      expect(errorOf(() => adds.buffers(token)).message).toMatch("still being built");
      adds.pushPositions(token, T.positions);
      adds.pushIndices(token, T.indices);
      expect(adds.step(token, 1)).toBe(false);
      expect(errorOf(() => adds.buffers(token)).message).toMatch("still being built");
      while (!adds.step(token, 100_000)) {}
      expect(adds.buffers(token).positions.length).toBe(T.positions.length);
    } finally {
      s.free();
    }
  });

  it("finish_without_steps_builds_everything", () => {
    const s = new Scene();
    try {
      const ref = s.addMesh(T.positions, T.indices);
      const want = {
        positions: s.views(ref).positions().slice(),
        normals: s.views(ref).normals()?.slice() ?? null,
        indices: s.views(ref).indices().slice(),
      };
      const adds = meshAdds(s);
      const token = adds.beginUpload(T.positions.length / 3, T.indices.length / 3, {});
      adds.pushPositions(token, T.positions);
      adds.pushIndices(token, T.indices);
      const m = adds.finish(token);
      const v = s.views(m);
      expect(bytesOf(v.positions()).equals(bytesOf(want.positions))).toBe(true);
      expect(v.normals() === null).toBe(want.normals === null);
      if (v.normals() !== null && want.normals !== null) {
        expect(bytesOf(v.normals() as Float32Array).equals(bytesOf(want.normals))).toBe(true);
      }
      expect(bytesOf(v.indices()).equals(bytesOf(want.indices))).toBe(true);
      // And dropping an upload forgets it.
      const token2 = adds.beginUpload(1, 0, {});
      adds.drop(token2);
      expect(errorOf(() => adds.step(token2, 1)).message).toMatch("no mesh is being added");
    } finally {
      s.free();
    }
  });

  it("pushing_past_the_declared_counts_throws", () => {
    const s = new Scene();
    try {
      const adds = meshAdds(s);
      const token = adds.beginUpload(2, 1, {});
      adds.pushPositions(token, new Float64Array(6));
      expect(errorOf(() => adds.pushPositions(token, new Float64Array(1))).message).toMatch("declared");
      expect(errorOf(() => adds.pushIndices(token, new Uint32Array(4))).message).toMatch("declared");
      expect(errorOf(() => adds.step(token, 10)).message).toMatch("upload incomplete");
      adds.drop(token);
    } finally {
      s.free();
    }
  });
});
