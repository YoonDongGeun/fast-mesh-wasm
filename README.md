# fast-mesh-wasm

[![npm](https://img.shields.io/npm/v/fast-mesh-wasm)](https://www.npmjs.com/package/fast-mesh-wasm)
[![CI](https://github.com/YoonDongGeun/fast-mesh-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/YoonDongGeun/fast-mesh-wasm/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Fast triangle mesh utilities for JavaScript and TypeScript, written in Rust and compiled to WebAssembly.

- Works in browsers (Vite, webpack, Next.js), Node.js and Bun
- Installs the same way with npm, pnpm and bun. No native build step, no install scripts, zero dependencies
- Full TypeScript types and inline docs, generated from the Rust source
- A `Scene` for interactive work: placements, queries, distance-field colouring, sections, repair and undo/redo
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

const bounds = computeBounds(positions);
// { min: [0, 0, 0], max: [1, 1, 0], center: [0.5, 0.5, 0], size: [1, 1, 0] }
```

Inputs can be typed arrays or plain arrays: `Float32Array` positions and `Uint32Array` indices are used without copying, while `Uint16Array`, `Float64Array` and `number[]` are converted once.

### With three.js

```ts
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geometry.setIndex(new THREE.BufferAttribute(indices, 1));

const normals = computeVertexNormals(geometry.attributes.position.array, geometry.index!.array);
geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
```

### three.js adapter

`fast-mesh-wasm/three` binds `Scene` meshes to `THREE.Mesh` objects that follow the scene on every `scene.commitFrame()`. It needs `three` (an optional peer dependency, `>=0.160`) and, for TypeScript, `@types/three`:

```sh
npm install three
npm install -D @types/three
```

```ts
import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { Scene } from "fast-mesh-wasm";
import { attachTransformControls, bindMesh, bindSection } from "fast-mesh-wasm/three";

const scene = new Scene();
const a = scene.addMesh(positionsA, indicesA);
const b = scene.addMesh(positionsB, indicesB);

const threeScene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer();
renderer.localClippingEnabled = true; // needed for bindSection's clipping plane

const boundA = bindMesh(scene, a); // a THREE.Mesh in boundA.object
const boundB = bindMesh(scene, b, { color: 0x88aaff, opacity: 0.6 });
threeScene.add(boundA.object, boundB.object);

// Drag B with a TransformControls gizmo: one gesture is one scene drag and one undo step.
const controls = new TransformControls(camera, renderer.domElement);
controls.attach(boundB.object);
threeScene.add(controls.getHelper()); // three >= 0.169; on older three use threeScene.add(controls)
attachTransformControls(controls, scene, boundB);

// A section: contours and fill, and the bound meshes are clipped at its plane
// (the side the section normal points to stays).
const section = scene.addSection({ meshes: [a, b], plane: { origin: [0, 0, 0], normal: [0, 0, 1] } });
const boundSection = bindSection(scene, section, [boundA, boundB]);
threeScene.add(boundSection.group);

function frame() {
  scene.commitFrame(); // once per animation frame, before rendering: the adapters apply the changes here
  renderer.render(threeScene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

- Colouring (`scene.setColoring`) shows up as a vertex-colour attribute with no code in the app. Undo and redo move the objects through the same change stream.
- The engine's colour maps are sRGB-encoded bytes. The adapter converts them to three's linear working space as it copies them (a normalized `Uint16` colour attribute), so they look the same as in a renderer that passes the bytes through to a non-sRGB canvas.
- A material you pass to `bindMesh` is used as given and must not be shared between bound meshes (the adapter controls its `color`, `vertexColors`, `opacity`, `transparent` and `depthWrite`).
- The adapter only reads `scene.onChanges`; it never calls `takeChanges`.

A renderer-neutral version of the same loop, with raw WebGPU and no 3D library, is in [`examples/webgpu`](examples/webgpu).

## Scene API

For interactive work with several meshes, the `Scene` class keeps the meshes in WebAssembly memory and does the heavy parts in Rust: placements, a BVH per mesh, queries, signed distance fields with colour maps, planar sections, hole repair and undo/redo. Your renderer reads the results through zero-copy typed-array views, and you upload only the ranges that changed.

```ts
import { init, type MeshHandle, Scene, type Quat, type Vec3 } from "fast-mesh-wasm";

await init();
const scene = new Scene();

// Provided by your app (not part of this package): mesh data and the
// renderer's upload functions.
declare const upperPositions: Float64Array, upperIndices: Uint32Array;
declare const lowerPositions: Float64Array, lowerIndices: Uint32Array;
declare function uploadMatrix(mesh: MeshHandle, matrix: Float64Array): void;
declare function uploadColors(mesh: MeshHandle, startVertex: number, rgba: Uint8Array): void;

// Add meshes. Positions are in world space; the scene stores them relative to
// the mesh's bounding-box centre so single precision stays accurate.
const upper = scene.addMesh(upperPositions, upperIndices);
const lower = scene.addMesh(lowerPositions, lowerIndices);

// Colour the upper mesh by its signed distance to the lower one.
const gap = scene.addDistanceField({ name: "gap", on: upper, to: lower, maxDistance: 2 });
const colour = (field: typeof gap) =>
  scene.setColoring(upper, { field, map: "turbo", range: [-0.5, 1] });
colour(gap);

// Drag the lower mesh. The whole drag is one undo step. Each `move`
// recomputes the eager fields, so coalesce pointer events to one `move` per
// animation frame.
const drag = scene.drag(lower);
let pending: { rotation: Quat; translation: Vec3 } | undefined;
function onPointerMove(rotation: Quat, translation: Vec3) {
  pending = { rotation, translation }; // keep only the latest pose
}
function onPointerUp() {
  if (pending !== undefined) drag.move(pending);
  pending = undefined;
  drag.end();
}

// Once per animation frame: take what changed and upload only that.
function frame() {
  if (pending !== undefined) {
    drag.move(pending); // fields and colours follow
    pending = undefined;
  }
  for (const change of scene.commitFrame()) {
    const views = scene.views(change.mesh); // fresh views; do not keep them
    if (change.placementChanged) {
      uploadMatrix(change.mesh, scene.matrix4(change.mesh)); // Float64Array(16)
    }
    const colors = views.colors();
    if (colors !== null && change.colors !== "all") {
      for (const [start, count] of change.colors) {
        uploadColors(change.mesh, start, colors.subarray(4 * start, 4 * (start + count)));
      }
    } // "all" (a new mesh, or buffers that grew): re-upload the whole buffer
    // positions, normals and indices work the same way
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Toggle the colouring (display settings are not part of the undo history).
scene.setColoring(upper, null);
colour(gap);

// Undo and redo placements and edits.
if (scene.canUndo()) scene.undo();
scene.redo();

// Queries are in world space.
const hit = scene.raycast([0, 0, 50], [0, 0, -1]); // nearest hit over all meshes, or null
const closest = scene.meshDistance(upper, lower); // { distance, a, b, ... }

scene.free(); // or `using scene = new Scene()`
```

Errors are `FastMeshError`s with a `code` (`UNKNOWN_MESH`, `INVALID_ARGUMENT`, `STALE_BVH`, ...). After a trap inside the WebAssembly module every later call throws `INSTANCE_POISONED`. A view is only valid until the next scene call that may allocate or move buffers: a kept view may read other data or be empty, so re-acquire views after any scene call. `new Scene({ debug: true })` makes views warn when they are read after such a call. Mesh edits (`scene.edit`), `analyze`, `repair`, sections (`addSection`, `section`, `sectionGap`) and `findContacts` are documented in the types.

### Contact groups: fields to several meshes

A distance field can measure to several meshes at once. Give `to` a list, and each vertex holds the smallest signed distance over them: the nearest surface, or the deepest penetration. So a mesh touching several others shows one colour scale, whichever partner it touches.

```ts
// a, b, c, d: MeshHandles. Contacts a-b, b-c, a-c and a-d: one field per mesh, to its partners.
const fa = scene.addDistanceField({ name: "contact", on: a, to: [b, c, d], maxDistance: 1 });
const fb = scene.addDistanceField({ name: "contact", on: b, to: [a, c], maxDistance: 1 });
scene.fieldTargets(fa); // [b, c, d]
scene.fieldNearest(fa); // Uint32Array: per vertex, the index in [b, c, d] it came from (0xffffffff: none)
```

- **Exact.** The values are bit for bit the minimum of separate single-target fields, ties going to the earliest target, with the `parallel` feature on or off.
- **Only related fields recompute.** Moving `d` recomputes `a`'s field (`d` is one of its targets) and `d`'s own. Moving `a` recomputes every field that has `a`, and `b`'s field to `[a, c]` is left alone when only `d` moves. Within a field, only the vertices near the moved target, or whose value came from it, are recomputed.
- **The best value so far bounds the next target's search where that stays exact.** Once a vertex has a negative value (a penetration), a later target can only win by being deeper, so its search stops at that depth. In `"winding"` sign mode a positive value bounds the search too, with a winding-number test for a deeper inside. A positive value cannot bound a pseudonormal target, since a vertex inside it is negative however far its surface is. Either way a field to three targets is one field to keep, one colour pass and one value per vertex.
- **Removing a target mesh** removes it from the field; removing the last one removes the field (and its colouring). `fieldStatus(field).sign` sums up the targets (closed only if every target is closed); `fieldTargetStatus(field)` gives each one.
- **Rust.** `fast_mesh_scene::DistanceField::to` is now a `Vec<MeshId>`; `DistanceField::new(name, on, to, max_distance, sign)` builds the single-target form.

On a `WorkerScene` the same `to: [...]` works; read the indices with `await scene.fetchFieldNearest(field)` and the per-target status with `await scene.fieldTargetStatus(field)`. With the store, see `setContactGroup` below.

## Worker scene

`fast-mesh-wasm/worker` runs the same engine in a Web Worker, so adding a large mesh, building fields or recolouring never blocks the page. `createWorkerScene()` starts the Worker and resolves to a `WorkerScene` with the `Scene` methods. Your renderer keeps reading synchronous views: every result carries the bytes of its dirty ranges, which are copied into main-thread mirrors.

```ts
import * as THREE from "three";
import type { DragSession, PlacementInput } from "fast-mesh-wasm";
import { createWorkerScene } from "fast-mesh-wasm/worker";
import { bindMesh } from "fast-mesh-wasm/three"; // optional: the three.js adapter works on a WorkerScene too

// Provided by your app (not part of this package).
declare const upperPositions: Float64Array, upperIndices: Uint32Array;
declare const lowerPositions: Float64Array, lowerIndices: Uint32Array;
declare const renderer: THREE.WebGLRenderer, threeScene: THREE.Scene, camera: THREE.Camera;
declare const button: HTMLButtonElement, undoButton: HTMLButtonElement;

const scene = await createWorkerScene(); // no init() needed: the Worker loads the wasm

// Handles come back at once; the work runs in the Worker.
const upper = scene.addMesh(upperPositions, upperIndices);
const lower = scene.addMesh(lowerPositions, lowerIndices);
const gap = scene.addDistanceField({ name: "gap", on: upper, to: lower, maxDistance: 2 });
scene.setColoring(upper, { field: gap, map: "turbo", range: [-0.5, 1] });
const boundUpper = bindMesh(scene, upper);
threeScene.add(boundUpper.object);

// Drag the lower mesh: one drag per gesture, the latest pose kept until the next frame.
let drag: DragSession | undefined;
let pending: PlacementInput | undefined;
function onPointerDown() {
  drag = scene.drag(lower); // the whole drag is one undo step
}
function onPointerMove(pose: PlacementInput) {
  pending = pose; // keep only the latest pose
}
function onPointerUp() {
  if (pending !== undefined) drag?.move(pending);
  pending = undefined;
  drag?.end(); // returns at once
  drag = undefined;
}

// Once per animation frame: send this frame's requests, receive the results so far.
function frame() {
  if (pending !== undefined) drag?.move(pending); // at most one move per frame
  pending = undefined;
  scene.commitFrame(); // the adapters' listeners apply the changes here, as with Scene
  renderer.render(threeScene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Status of each piece of state: "idle" | "computing" | "ready" | "error".
scene.onStatus(() => {
  const coloring = scene.status(`display:${scene.idOf(upper)}`);
  button.classList.toggle("busy", coloring.status === "computing");
  undoButton.disabled = !scene.canUndo; // canUndo / canRedo are getters here
});

// Queries are async.
const hit = await scene.raycast([0, 0, 50], [0, 0, -1]);
const distance = await scene.meshDistance(upper, lower);
const values = await scene.fetchFieldValues(gap); // `fieldValues` on Scene
```

- **The frame loop.** `commitFrame()` sends everything queued since the last frame as one batch and returns the change records of the results received meanwhile (already applied to the mirrors, so `views`, `matrix4`, `placement`, `section` and `sectionPlane` read the new state). Results arrive between frames; nothing changes in the mirrors outside a message.
- **Requests run in order.** A newer colouring, section plane/meshes/fill or drag move drops the older one if it has not started yet, so fast toggling settles on the last state without doing the steps in between. Edits, repairs, `setPlacement`, adds, removals and undo/redo are never dropped. Each `setPlacement` is an undo step: for pointer-rate updates use a drag.
- **Drags are optimistic.** `scene.drag(mesh).move(pose)` moves the placement mirror at once, so the object follows the pointer; colours and sections catch up when the Worker's result arrives. In 9a a key's status covers only requests made under that key: a drag, `setPlacement` or edit puts its own key (`placement:<id>`, `edit:<id>`) into `"computing"`, not the `display:<id>` or `section:<id>:*` keys of the state that catches up after it. `end()` and `cancel()` return at once; `cancel()` restores the start pose and records nothing.
- **Wide-range colouring during a drag.** After each batch the Worker recomputes the fields the batch made due in small steps, and drops the recompute when a newer move, `setPlacement`, edit, repair, compact, undo/redo or removal of either mesh is already queued; other requests wait for it. A pose's colours are applied all at once, never partly. So while the pointer keeps moving and one recompute takes longer than a frame, newer moves drop the recompute, until 200 ms have passed since the last colours were applied: then the recompute under way finishes and is shown, and the newest pose follows. During a sustained drag the colours therefore update about every 200 ms plus one recompute, each update exactly one pose, and when the drag ends (`end()`) the last pose's colours arrive within about one recompute (measured: one at 1M, but about 1.5 at 10M when most of the mesh is in range): the end drops a recompute kept for an older pose. A pointer that only pauses, without ending the drag, may wait up to two recomputes. Edits, undo/redo, removals and cancels always drop a stale recompute. `fetchFieldValues()` and `fieldStatus()` first bring every due field up to date in one go, which cannot be dropped, so do not poll them during a drag. On two touching 1M-triangle meshes with `maxDistance` 8, a recompute takes from about 0.07 s to 0.7 s depending on how much of the mesh is in range, and about ten times that at 10M; the longest the Worker is blocked at a time is the start of a recompute, about 60 ms at 1M and 0.6 s at 10M (up to 0.14 s and 1.1 s were seen) (see [docs/benchmarks.md](docs/benchmarks.md)). These are single-threaded figures; on a cross-origin isolated page the threaded engine roughly halves the longest block and makes a recompute 1.4 to 4.5 times faster (see [Multithreading (COOP/COEP)](#multithreading-coopcoep)). A contact range (a small `maxDistance`) keeps up with the pointer.
- **Queries and anything that returns data** (`raycast`, `closestPoint`, `meshDistance`, `intersects`, `findContacts`, `analyze`, `edit`, `repair`, `fetchFieldValues`, `fieldStatus`, `sectionGap`) return promises. Once a frame loop runs they go out with the next `commitFrame()` (or after 100 ms if no frame comes); before the first `commitFrame()` they are sent on a microtask. For latency-critical picking, keep a light picking copy in a main-thread `Scene`.
- **Errors.** A failed state change shows on its key (`status(key).error`, a `FastMeshError` with the same `code` as on `Scene`); a failed promise rejects with it. After a trap inside the Worker every call fails with `INSTANCE_POISONED`, also when it happens on one of the multithreaded engine's pool threads (a panic or a trap in work split across threads); a few kinds of pool-thread failure still cannot be detected (see [Multithreading (COOP/COEP)](#multithreading-coopcoep)). `await scene.restart()` starts a new Worker and replays the scene from the main thread's mirrors: meshes (compacted: slot ids change, handles do not), placements, fields, colourings and sections. History is cleared, and edits or queries that had not finished fail and must be re-issued.
- **Not in the Worker scene:** user-opened steps (`beginStep`/`endStep`/`cancelStep`), Lazy and Manual modes (it is always eager) and `takeChanges` (use `onChanges` or the return value of `commitFrame()`). `terminate()` stops the Worker.
- **Large meshes.** `addMesh` copies its inputs on the main thread (they are then transferred to the Worker) and, without an `origin` option, scans the positions for the bounding-box centre. That takes about 9 ms for 1M triangles and about 63 ms for 10M (see [docs/benchmarks.md](docs/benchmarks.md)); passing `origin` skips the scan.

**Bundlers.** The Worker script is a separate file, `worker-entry.js`, loaded with `new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })`. Vite and other bundlers that understand this pattern pick it up and emit it with the `.wasm` it loads: [`examples/webgpu`](examples/webgpu) uses the default with Vite, in both `vite build` and `vite dev` (open it with `?worker`). If your setup does not handle the pattern, serve the package's `dist/` folder (the Worker script loads the files next to it) and pass the script's URL, or create the Worker yourself:

```ts
const scene = await createWorkerScene({ workerUrl: "/fast-mesh-wasm/worker-entry.js" });
// or
const scene = await createWorkerScene({ createWorker: () => new Worker(myUrl, { type: "module" }) });
```

`fast-mesh-wasm/worker` is for browsers and is not supported in Node.js (its tests run it on Node's `worker_threads` as a harness only).

## Multithreading (COOP/COEP)

On a cross-origin isolated page the Worker scene runs a multithreaded engine: its WebAssembly runs on a pool of Workers that share its memory, and the work that the Rust crates' `parallel` feature splits (BVH builds, sign data, distance and closest-point queries, near-triangle searches, see "Parallel" in [docs/benchmarks.md](docs/benchmarks.md)) is split across them. Results are bit-identical to the single-threaded engine. On two touching 1M-triangle meshes with a wide distance field, 8 threads made a drag's recompute about 1.6 to 4.5 times faster and halved the longest Worker block (headless Chrome; see "Threaded Worker" in [docs/benchmarks.md](docs/benchmarks.md)). Nothing changes in your code: the same `createWorkerScene()` picks the engine.

A page is cross-origin isolated (`crossOriginIsolated === true`) when its document is served with these two headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Send them on **every** response, including 304 (Not Modified) responses: Safari refuses to start a Worker whose script it revalidated with a 304 that lacks them.

**Vite** (dev and preview). Use a small plugin rather than `server.headers`/`preview.headers`, which Vite leaves off its 304 responses ([`examples/webgpu/vite.config.ts`](examples/webgpu/vite.config.ts) does this):

```ts
import { defineConfig, type Plugin } from "vite";

function crossOriginIsolation(): Plugin {
  const isolate = (_req: unknown, res: { setHeader(name: string, value: string): void }, next: () => void) => {
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    next();
  };
  return {
    name: "cross-origin-isolation",
    configureServer: (server) => void server.middlewares.use(isolate),
    configurePreviewServer: (server) => void server.middlewares.use(isolate),
  };
}

export default defineConfig({ plugins: [crossOriginIsolation()] });
```

**webpack-dev-server** (a middleware ahead of the others, so every response gets the headers):

```js
// webpack.config.js
module.exports = {
  devServer: {
    setupMiddlewares: (middlewares) => {
      middlewares.unshift((req, res, next) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      });
      return middlewares;
    },
  },
};
```

**nginx** (`add_header` already covers 200, 304 and the other success and redirect statuses; `always` adds the 4xx and 5xx ones too):

```nginx
add_header Cross-Origin-Opener-Policy "same-origin" always;
add_header Cross-Origin-Embedder-Policy "require-corp" always;
```

A `location` (or other block) with any `add_header` of its own, such as a `Cache-Control` for assets, inherits none of the server-level ones: repeat the two headers in every such block, or keep them in a file you `include` in each.

**Express** (before `express.static`):

```js
app.use((req, res, next) => {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
});
app.use(express.static("dist"));
```

**What the headers change for the rest of the page.** With COEP `require-corp`, the page and its Workers may only load cross-origin resources (images and textures, scripts, fonts, models, `fetch` from a CDN or API) that opt in: load them with CORS (`crossorigin` on `<img>`/`<script>`/`<link>`, `img.crossOrigin = "anonymous"` or three.js's `TextureLoader`, whose default is `anonymous`) from a server that sends `Access-Control-Allow-Origin`, or have that server send `Cross-Origin-Resource-Policy: cross-origin` (meant for public assets only: it lets any site embed them). Anything else is blocked, so check third-party embeds before turning isolation on. COOP `same-origin` also cuts the link between your page and cross-origin popups it opens (`window.opener`), which some sign-in flows rely on. Same-origin assets, including this package's files, need nothing. `Cross-Origin-Embedder-Policy: credentialless` also makes a page isolated and loads cross-origin resources without credentials instead of blocking those that do not opt in; Chrome and Firefox support it, Safari does not (the page is then not isolated there, and the scene runs single-threaded).

**The `threads` option.**

```ts
const scene = await createWorkerScene(); // default: navigator.hardwareConcurrency, at most 8
const fixed = await createWorkerScene({ threads: 4 }); // clamped to 1..64; a non-integer rejects with INVALID_ARGUMENT
const single = await createWorkerScene({ threads: 1 }); // the single-threaded engine; the threaded files are never fetched
console.log(scene.threads); // the threads the engine computes with: 1 when single-threaded (and until the Worker is ready)
```

**Fallback.** The scene always starts. On a page that is not cross-origin isolated it runs single-threaded and never fetches the threaded files. On an isolated page, if the threaded engine cannot start (its import or instantiation fails, the shared memory cannot be allocated, or the thread pool rejects or has not started within 10 s, for example because its nested Workers cannot load), the scene runs single-threaded. When the failure comes after the threaded engine was instantiated, its Worker is replaced by a fresh single-threaded one, and requests made meanwhile are kept and run there. A scene that fell back this way stays single-threaded across `restart()`, so a restart does not wait 10 s again. `scene.threads` tells which engine runs.

**Crashes on a pool thread.** With the single-threaded engine, a trap in the WebAssembly module poisons the scene (every call fails with `INSTANCE_POISONED`). With the multithreaded engine, a crash on a pool thread ends only that thread, and the engine's Worker keeps waiting for it. Each pool thread therefore reports its own crash to the scene on the main thread, through a `BroadcastChannel` with a name unique to that Worker. The scene then fails every pending call with `INSTANCE_POISONED` and terminates the Worker and its pool, as it does after a Worker error; `restart()` recovers with a new pool. This covers Rust panics (the panic hook reports the message before the abort) and WebAssembly traps on a pool thread. Panics and the `unreachable` trap are tested in Chrome, Firefox and WebKit. Other traps are expected to be detected through the same path as `unreachable`, but are not tested: out-of-bounds access, the abort after a failed allocation, and the JavaScript call stack running out. It costs nothing until a thread crashes: no polling, just one idle channel per threaded Worker. The following still cannot be detected, and leave the scene waiting (pending promises never settle, statuses stay `"computing"`), with `restart()` and `terminate()` still working:
- an overflow of a pool thread's stack in WebAssembly memory. WebAssembly has no guard page, so this is not a trap: it overwrites other memory, and is only noticed if that later traps;
- a nested Worker that the browser ends without an exception (for example when the tab's process runs out of memory);
- a pool thread that loops or deadlocks instead of crashing;
- a browser with no `BroadcastChannel` in Workers.

Applications that cannot accept these cases should pass `threads: 1`.

**The development warning.** In development builds (`import.meta.env.DEV`, as Vite sets it) or with the `debug` option, the scene logs once (per scene, also across restarts) why it runs single-threaded when more threads were asked for: the page is not isolated, or the threaded engine failed to start, with the cause. Production builds are silent.

**Bundlers.** No setting beyond the headers is needed: Vite (dev and build) and webpack 5 work as installed from npm, and so does a page without a bundler that serves the package's `dist/` folder as is (checked by `scripts/bundler-check.mjs`). Bundlers emit the threaded engine into your build output next to the single-threaded one: a chunk, its `.wasm` (about 988 KB, 372 KB gzipped, against 772 KB and 307 KB) and the thread pool's Worker helper. Pages that run single-threaded never download them.

**Without header control** (some static hosts), a service worker such as [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) can make a page cross-origin isolated by adding the headers itself. This package does not ship or test one.

**Browsers.** Current Chrome and Edge, Firefox and Safari support it. Checked: headless Chrome 155, Firefox 157 and Playwright's WebKit 27.2 on Windows (bit-identical results in all three). Known issue: Playwright's Windows WebKit build crashed its web process in 2 of about 20 development-server runs of the check that crashes a threaded Worker on purpose and restarts it; the cause is unknown, and real Safari on macOS has not been checked.

The threaded engine adds memory: its shared memory starts at about 1 MB (18 pages) and may grow to 4 GiB (wasm32's limit), and each pool Worker has its own stack. A 10M-triangle wide-drag scene used about 2.4 GB threaded against 2.1 GB single-threaded.

## React

Three subpaths connect a scene to UI code:

- `fast-mesh-wasm/store`: `SceneStore`, an observable store over a `Scene` or a `WorkerScene`. It has no dependencies, so it also works without React.
- `fast-mesh-wasm/react`: hooks over the store. Needs `react` (optional peer, `>=18`; tested with React 19).
- `fast-mesh-wasm/r3f`: `<FastMesh>` for React Three Fiber. Needs `@react-three/fiber` (optional peer, `>=9`) and `three`.

```tsx
import { createSceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene } from "fast-mesh-wasm/worker";
import { SceneProvider, useHistory, useMeshDisplay } from "fast-mesh-wasm/react";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { Canvas } from "@react-three/fiber";
import type { MeshHandle } from "fast-mesh-wasm";

// Provided by your app (not part of this package).
declare const upperPositions: Float64Array, upperIndices: Uint32Array;
declare const lowerPositions: Float64Array, lowerIndices: Uint32Array;

// One store per scene. With the synchronous Scene: `await init(); createSceneStore(new Scene())`.
const store = createSceneStore(await createWorkerScene());
const upper = store.addMesh(upperPositions, upperIndices);
const lower = store.addMesh(lowerPositions, lowerIndices);

function ColourButton({ mesh, to }: { mesh: MeshHandle; to: MeshHandle }) {
  const [state, setDisplay] = useMeshDisplay(mesh);
  const pressed = state.desired.coloring !== null; // what was requested
  const busy = state.status === "computing"; // not shown yet
  return (
    <button
      aria-pressed={pressed}
      aria-busy={busy}
      onClick={() =>
        setDisplay({ coloring: pressed ? null : { distanceTo: to, maxDistance: 2, map: "turbo", range: [0, 2] } })
      }
    >
      Distance colours
    </button>
  );
}

function UndoButton() {
  const { canUndo, undo } = useHistory();
  return <button disabled={!canUndo} onClick={undo}>Undo</button>;
}

export function App() {
  return (
    <SceneProvider store={store}>
      <Canvas>
        <ambientLight />
        <FastMesh mesh={upper} />
        <FastMesh mesh={lower} />
      </Canvas>
      <ColourButton mesh={upper} to={lower} />
      <UndoButton />
    </SceneProvider>
  );
}
```

**The store.** `store.getSnapshot()` returns a frozen `SceneSnapshot` (meshes, sections, history, selection and an overall status), replaced on every change, with unchanged parts shared between versions. `store.subscribe(listener)` notifies after each new snapshot: at once for an action, and once per task for the scene's events (on a microtask; `getSnapshot()` is up to date before then). An update re-derives only the meshes and sections it touches. What still grows with the scene is one copy of the snapshot's mesh map per change: a native `Map` copy, about 45 µs at 1500 meshes. Actions mirror the scene: `addMesh`, `removeMesh`, `setPlacement`, `drag`, `edit`, `repair`, `undo`, `redo`, `addSection`, `removeSection` and `restart` (Worker only). The declarative ones are new:

- `setDisplay(mesh, { coloring?, opacity?, visible? })`. A `coloring` colours the mesh by its distance to `distanceTo`: one mesh, or a list of them (the smallest signed distance over them; see [Contact groups](#contact-groups-fields-to-several-meshes)). The store owns the distance field behind it: one per coloured mesh, named `fast-mesh-store:<n>`, kept while the colouring is off, reused when it comes back with the same `distanceTo` (as a set: the order does not matter), `maxDistance` and `sign`, and replaced only when one of them changes. `opacity` (in [0, 1], default 1) and `visible` (default true) live only in the store; the engine never sees them. Display changes never enter the undo history.
- `setContactGroup(id, { meshes, pairs?, coloring } | null)` colours a group of meshes by contact. Each member gets `coloring` with its partners as `distanceTo`: the members it is paired with in `pairs` (either way round), or every other member with `pairs: "all"` (the default). `coloring` is a `DistanceColoring` without `distanceTo`, so any preset spreads into it. Moving a member recomputes only its own field and its partners'. Setting the group again updates the members; a member dropped from it, or all of them with `null`, is uncoloured and its store-owned field removed. A member the app has coloured itself since (with `setDisplay`) is left alone by those changes. A removed mesh leaves its groups: the others keep their remaining partners, and a member left with none is uncoloured. The colouring is checked when the group is set (`INVALID_ARGUMENT`). Groups are not in the snapshot: the members' `display` and `desired` colourings show them.

  ```ts
  store.setContactGroup("parts", {
    meshes: [a, b, c, d],
    pairs: [[a, b], [b, c], [a, c], [a, d]],
    coloring: { maxDistance: 1, map: "coolwarm", range: [-0.5, 0.5], center: 0, above: "base" },
  });
  ```

- `setSection(section, { plane?, meshes?, fill? })` changes a section.
- `select(mesh | null)` sets the selection. It is UI state only, and it clears when the selected mesh is removed.

**Contact heatmap.** `contactHeatmap({ depth, gap, tolerance?, exactDepth?, sign?, colors? })` builds a colouring (everything but `distanceTo`) that shows how two meshes touch. Spread it into `setDisplay`, colouring each mesh against the other:

```ts
import { contactHeatmap } from "fast-mesh-wasm/store";

const heat = contactHeatmap({ depth: 0.5, gap: 1, tolerance: 0.05 });
store.setDisplay(upper, { coloring: { distanceTo: lower, ...heat } });
store.setDisplay(lower, { coloring: { distanceTo: upper, ...heat } });
```

- The range is `[-depth, gap]`. Penetration runs from `deep` (`#7f0000`) at `-depth` to `shallow` (`#ff6b6b`) at the contact band, which is `contact` (`#ffffff`) across `[-tolerance, +tolerance]` (one stop at 0 when the tolerance is 0, the default). The gap runs from `near` (`#4caf50`) to `far` (`#1e3a8a`) at `gap`. Deeper than `depth` clamps to `deep`; farther than `gap` shows the base colour. `gap: 0` colours penetration only: the range is then `[-depth, tolerance]`, with no gap ramp, and anything beyond the contact band shows the base colour. `colors` overrides any of the five.
- `exactDepth: false` (the default) bounds the field at `max(depth, gap)`: the fast path for contact, but a vertex deeper than that is beyond the field and shows the base colour. `exactDepth: true` computes every vertex's distance (`maxDistance: Infinity`), so any penetration shows, at the cost of a full recompute on every update.
- `sign` (default `"auto"`) is passed through. On an open or inconsistently oriented target, the penetration side follows its surface normals (`fieldStatus` tells you which).
- Changing only `tolerance` or `colors` keeps `maxDistance` and `sign`, so the store reuses its field and only recolours.
- It throws `INVALID_ARGUMENT` unless `depth > 0`, `gap >= 0` (both finite) and `0 <= tolerance < min(depth, gap)` (`< depth` when `gap` is 0).

`store.fetchColoringValues(mesh)` resolves to a copy of the values of the field behind the mesh's colouring (one per vertex slot), or `null` when the mesh has no colouring: for readouts such as the deepest penetration, `-min(values)`. It uses `fetchFieldValues` on a Worker scene and copies `fieldValues` on a synchronous one. It rejects with `UNKNOWN_MESH` for a mesh not in the store.

Declarative changes update `desired` at once and reach the scene at the next `store.commitFrame()`, the last value per mesh or section in a frame winning (on both scenes, so rapid toggling never queues work). Without a frame loop they apply on a microtask. Call `store.commitFrame()` once per animation frame instead of `scene.commitFrame()`; `<FastMesh>` does it for you. The first `store.commitFrame()` switches the store to frame flushing: from then on, declarative changes wait for a frame. When the last `<FastMesh>` (or `useFrameDriver`) of a store unmounts, the store switches back to flushing on a microtask, so changes made with no canvas mounted still apply.

Meshes must be added through the store (`store.addMesh`). A mesh added straight to `store.scene` is not in the store: its hooks read it as removed, and `<FastMesh>` renders nothing for it.

`store.dispose()` removes the distance fields the store owns (and so their colourings) and stops listening to the scene; the rest of the scene is left as it is. On a Worker scene the removals go with the scene's next `commitFrame()`.

**Reading state in components.**

| Hook | Returns |
| --- | --- |
| `useScene()` | The store of the nearest `<SceneProvider store>` |
| `useMeshDisplay(mesh)` | `[state, setDisplay]`: the mesh's applied `coloring`, `opacity` and `visible`, plus `desired`, `status` and `error` |
| `useSection(section)` | `[state, setSection]`: the applied `plane`, `meshes` and `fill`, plus `desired`, `status` and `error` |
| `useHistory()` | `{ canUndo, canRedo, undo, redo }` |
| `useSceneStatus()` | `{ status, error? }` for the whole scene: `"error"` if anything failed, else `"computing"` while anything is pending, else `"idle"` |
| `useSelection()` | `[mesh \| null, select]` |
| `useStoreSelector(selector, isEqual?)` | Any slice of the snapshot; re-renders only when it changes |

- **Bind controlled inputs (sliders, toggles, checkboxes) to `state.desired.*`.** That is what the user last asked for, so the input follows them at once. The top-level `state.coloring`, `state.opacity` and `state.visible` (and a section's top-level `plane`, `meshes` and `fill`) are the applied values: what is on screen.
- **`status === "computing"` covers the pending window of colourings and sections**: the time between a request (the button is pressed) and the scene showing it, whether the change is waiting for the next frame or a colouring is being computed in the Worker. Opacity and visibility never read as pending: they live in the store and apply at the next flush. Show it as busy; the button never reads "on" while the mesh shows colours off outside this window. A mesh is also `"computing"` while something it depends on is pending: its placement or an edit, those of the mesh it is coloured by distance to, or an undo/redo. A section depends the same way on its meshes. Only a state's own requests can make it `"error"`; a failed edit rejects its own promise instead.
- **On the synchronous `Scene`** every status is `"ready"` right after the frame that applied it, or `"error"` if the call threw. `store.restart()` rejects with `INVALID_ARGUMENT` there.
- **A removed mesh or section** reads as `"idle"` with the default display (or no meshes) and its setter throws `UNKNOWN_MESH` or `UNKNOWN_SECTION`, so a component still mounted for it renders without throwing. Removing the mesh another mesh is coloured by distance to (alone) leaves that mesh uncoloured, its rule kept in `desired` and its status `"error"` (`UNKNOWN_MESH`) until its colouring changes; a colouring to several meshes keeps the others. A rule the engine rejected is not re-sent until it changes.
- **SSR.** On the server the hooks render an empty snapshot (`EMPTY_SNAPSHOT`) and never throw for missing data.

**`<FastMesh>`.** `<FastMesh mesh store? material? color? onBind? onClick?… />` binds a scene mesh (added through the store) with `bindMesh` on mount, follows the applied opacity and visibility, and disposes the binding on unmount or when the mesh is removed (it then renders nothing). Opacity and visibility apply inside the frame's `store.commitFrame()`, so they show in the same frame as a colouring flushed with them. R3F pointer events are passed to the mesh. Adding or removing the `store` prop remounts it (and rebinds the mesh). Give each `<FastMesh>` its own `material`, and dispose it yourself. Every `<FastMesh>` calls `useFrameDriver(store)`, which calls `store.commitFrame()` once per rendered frame however many are mounted (do not call `commitFrame()` yourself as well), and invalidates the canvas when the store or the scene changes, so `<Canvas frameloop="demand">` renders every change. Use `useFrameDriver()` on its own when you draw the scene without `<FastMesh>`.

- **`onBind(bound)`** delivers the `BoundMesh` once the mesh is bound, and `null` when that binding is disposed (unmount, removal, or a rebind after a `material` or `color` change, which then delivers the new one). Pass it to `bindSection` or `attachTransformControls` (`fast-mesh-wasm/three`), which take any `MeshBinding` (`Pick<BoundMesh, "mesh" | "object">`). `bindSection` clips the materials it sees when it is called, so bind the section again when `onBind` delivers a new binding. An inline callback is fine: a new callback never rebinds. [`examples/react`](examples/react) keeps the bindings in a map for its section and its gizmo.
- **One store across two `<Canvas>` roots.** The first mounted driver for a store does the committing; every canvas is invalidated on change. If the committing canvas does not render at all (`frameloop="never"`, say) while another one runs, the store stops committing. Drive such a store from one canvas only, or call `useFrameDriver` in the canvas that always renders.

[`examples/react`](examples/react) is a complete app on the Worker store (`?local` for the synchronous one): colouring toggles, opacity and visibility, a section, undo/redo, the scene status with Restart, dragging with three's TransformControls and STL drop.

**Showcase.** [`examples/showcase`](examples/showcase) is a small mesh tool on the same stack: load STL, OBJ and PLY files (parsed in a Worker), move meshes with a gizmo, analyse and repair them (weld, remove degenerate triangles, orient, remove small components, fill holes; each one undo step), and see how two meshes touch as a contact heatmap (`contactHeatmap`) with live readouts of the deepest penetration, the gap and the contacts.

## Functions

| Function | Returns | Description |
| --- | --- | --- |
| `init(input?)` | `Promise<void>` | Loads the WebAssembly module. Required once in browsers; resolves immediately in Node.js and Bun. Safe to call repeatedly. `input` optionally overrides where the `.wasm` file comes from (`InitInput`). |
| `computeVertexNormals(positions, indices)` | `Float32Array` | Smooth, area-weighted per-vertex normals. Same length as `positions`. |
| `computeBounds(positions)` | `Bounds` | Plain object `{ min, max, center, size }`, each a `[x, y, z]` array. |

All functions throw an `Error` with a readable message when a buffer length is not a multiple of 3 or an index is out of range, and when called before `init()` has finished in the browser.

## Supported environments

| Environment | Entry | Notes |
| --- | --- | --- |
| Browsers via a bundler | `dist/web.js` | Vite, webpack 5, Next.js and others copy the `.wasm` asset automatically |
| Node.js 22+ | `dist/node.js` | The module is ready on import, also inside webpack server bundles (e.g. Next.js) |
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
