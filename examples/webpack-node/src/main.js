import assert from "node:assert/strict";
import { computeBounds, computeVertexNormals } from "fast-mesh-wasm";

const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

assert.deepEqual(Array.from(computeVertexNormals(positions, indices)), [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
assert.deepEqual(computeBounds(positions).max, [1, 1, 0]);

console.log("fast-mesh-wasm works in a webpack server bundle");
