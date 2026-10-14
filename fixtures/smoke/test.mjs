import assert from "node:assert/strict";
import { computeBounds, computeVertexNormals, init } from "fast-mesh-wasm";

await init();

const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

assert.deepEqual(Array.from(computeVertexNormals(positions, indices)), [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);

assert.deepEqual(computeBounds(positions).max, [1, 1, 0]);
assert.deepEqual(computeVertexNormals(Array.from(positions), new Uint16Array(indices)).length, 12);

const runtime = typeof Bun !== "undefined" ? `bun ${Bun.version}` : `node ${process.version}`;
console.log(`fast-mesh-wasm works on ${runtime}`);
