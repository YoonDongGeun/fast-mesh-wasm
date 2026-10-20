---
"fast-mesh-wasm": minor
---

Add `fast-mesh-wasm/store`, `fast-mesh-wasm/react` and `fast-mesh-wasm/r3f`.

`fast-mesh-wasm/store`: `createSceneStore(scene)` wraps a `Scene` or a `WorkerScene` in a `SceneStore` with frozen, structurally shared snapshots (`getSnapshot` / `subscribe`): per mesh its applied `display`, the `desired` display and a `status` (`"idle"`, `"computing"`, `"ready"` or `"error"`), plus sections, history, selection and an overall status. Besides the scene's actions it adds declarative ones: `setDisplay(mesh, { coloring, opacity, visible })`, where `coloring` colours a mesh by its distance to another through a distance field the store owns (one per coloured mesh, reused when the colouring is toggled back on), `setSection(section, { plane, meshes, fill })` and `select(mesh)`. Declarative changes apply at the next `store.commitFrame()`, the last value per frame winning, so rapid toggling never queues work; a mesh or section reads `"computing"` until its change is shown, including while a placement, edit or undo it depends on is pending. `opacity`, `visible` and the selection live only in the store and never enter history. The store has no dependencies (no React, three.js or wasm glue).

`fast-mesh-wasm/react` (optional peer `react >=18`): `SceneProvider`, `useScene`, `useStoreSelector`, `useMeshDisplay`, `useSection`, `useHistory`, `useSceneStatus` and `useSelection`, built on `useSyncExternalStore` with memoised selectors, so components re-render only when their slice changes and render an empty snapshot on the server. Bind controlled inputs to `state.desired`; the top-level fields are what is shown.

`fast-mesh-wasm/r3f` (optional peer `@react-three/fiber >=9`, tested with 9.x; v8 is not supported): `<FastMesh mesh />` draws a scene mesh through `bindMesh`, follows its opacity and visibility, and disposes on unmount or removal; `useFrameDriver(store)` calls `store.commitFrame()` once per rendered frame however many `<FastMesh>` are mounted.

A new example, `examples/react`, runs the store on a Worker scene with colouring toggles, opacity, visibility, a section, undo/redo, Restart and TransformControls dragging.
