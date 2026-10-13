# fast-mesh-wasm

[![npm](https://img.shields.io/npm/v/fast-mesh-wasm)](https://www.npmjs.com/package/fast-mesh-wasm)
[![CI](https://github.com/YoonDongGeun/fast-mesh-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/YoonDongGeun/fast-mesh-wasm/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Fast triangle mesh utilities for JavaScript and TypeScript, written in Rust and compiled to WebAssembly.

- Works in browsers (Vite, webpack, Next.js), Node.js and Bun
- Installs the same way with npm, pnpm and bun. No native build step, no install scripts, zero dependencies
- Full TypeScript types and inline docs, generated from the Rust source
- Uses the flat buffer layout of WebGL and three.js `BufferGeometry`, so data goes in and out without conversion

> **Status:** early development (0.x). The API may change between minor versions.

## Install

```sh
npm install fast-mesh-wasm
pnpm add fast-mesh-wasm
bun add fast-mesh-wasm
```

## Usage

```ts
import { init, computeVertexNormals, computeBounds } from "fast-mesh-wasm";

await init(); // loads the .wasm file; required in browsers, a no-op in Node.js and Bun

// A unit quad: 4 vertices, 2 triangles
const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

const normals = computeVertexNormals(positions, indices);
// Float32Array [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

using bounds = computeBounds(positions); // freed automatically at the end of the block
bounds.min; // Float32Array [0, 0, 0]
bounds.max; // Float32Array [1, 1, 0]
```

### With three.js

```ts
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geometry.setIndex(new THREE.BufferAttribute(indices, 1));

const normals = computeVertexNormals(
  geometry.attributes.position.array as Float32Array,
  geometry.index!.array as Uint32Array,
);
geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
```

## API

| Function | Returns | Description |
| --- | --- | --- |
| `init(input?)` | `Promise<void>` | Loads the WebAssembly module. Call once before anything else. Safe to call repeatedly. |
| `computeVertexNormals(positions, indices)` | `Float32Array` | Smooth, area-weighted per-vertex normals. Same length as `positions`. |
| `computeBounds(positions)` | `Bounds` | Axis-aligned bounding box with `min`, `max`, `center` and `size`. |

All functions throw an `Error` with a readable message when a buffer length is not a multiple of 3 or an index is out of range.

### Memory

Objects returned as classes (such as `Bounds`) live in WebAssembly memory, which the JavaScript garbage collector does not reclaim right away. Release them with `using` (TypeScript 5.2+, modern runtimes) or by calling `.free()`:

```ts
const bounds = computeBounds(positions);
try {
  console.log(bounds.size);
} finally {
  bounds.free();
}
```

Typed arrays such as the result of `computeVertexNormals` are ordinary JavaScript copies and need no cleanup.

## Supported environments

| Environment | Entry | Notes |
| --- | --- | --- |
| Browsers via a bundler | `dist/web.js` | Vite, webpack 5, Next.js and others copy the `.wasm` asset automatically |
| Node.js 22+ | `dist/node.js` | The module is ready on import |
| Bun | `dist/node.js` | Same as Node.js |

The right entry is picked automatically through the `exports` field of `package.json`. The package is ESM only.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the build pipeline and the release process. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
