# fast-mesh-wasm + React + React Three Fiber

A sphere above a wavy plate, drawn with `<FastMesh>` (`fast-mesh-wasm/r3f`) and controlled through
React hooks (`fast-mesh-wasm/react`) over a `SceneStore` (`fast-mesh-wasm/store`). The scene runs in a
Web Worker by default; add `?local` to the URL to run it on the main thread with the synchronous `Scene`.
The components are the same in both modes.

## The pattern

```tsx
// main.tsx: one store for the app, seeded outside React (StrictMode runs initialisers twice).
const store = createSceneStore(await createWorkerScene()); // or createSceneStore(new Scene()) after init()
const sphere = store.addMesh(positions, indices);
root.render(<SceneProvider store={store}><App /></SceneProvider>);

// In the <Canvas>: <FastMesh> binds the mesh and drives the store's frames (store.commitFrame()).
<FastMesh mesh={sphere} material={material} onClick={...} />

// In the panel: controlled inputs bind to `desired`; `status === "computing"` is the pending window.
const [state, setDisplay] = useMeshDisplay(sphere);
<button aria-pressed={state.desired.coloring !== null} aria-busy={state.status === "computing"}
        onClick={() => setDisplay({ coloring: on ? null : { distanceTo: plate, maxDistance: 1.5, map: "turbo", range: [0, 1.5] } })} />
<input type="range" value={state.desired.opacity} onChange={(e) => setDisplay({ opacity: +e.target.value })} />
```

- `state.desired.*` is what was last requested: bind buttons and sliders to it, so they follow the
  user at once. The top-level `state.coloring`, `state.opacity` and `state.visible` are what is shown.
- `state.status === "computing"` covers the window between the two (a colouring being computed in the
  Worker, or a change waiting for the next frame). Show it as busy rather than flipping the button back.
- The store owns the distance field behind a colouring. Toggling it off and on reuses the field.

Files:

- `src/main.tsx`: creates the store (Worker or `?local`), seeds the meshes and the section, renders the app.
- `src/App.tsx`: the `<Canvas>`: `<FastMesh>` per mesh, the section (`bindSection`), an orbit camera and a
  TransformControls gizmo on the selected mesh (`attachTransformControls`), keyboard shortcuts and STL drop.
- `src/Panel.tsx`: the panels, with `useMeshDisplay`, `useSection`, `useHistory`, `useSceneStatus` and `useSelection`.
- `src/items.ts`, `src/meshes.ts`: the app's mesh list, the generators and the STL reader (from the WebGPU example).

`bindSection` and `attachTransformControls` (from `fast-mesh-wasm/three`) take `MeshBinding` objects.
Each `<FastMesh onBind>` delivers its binding; the app keeps them in a map by mesh and passes them on.
The section is bound again when a binding changes, since `bindSection` clips the materials it sees then.

## Run it

```sh
pnpm install
pnpm build                                        # builds the fast-mesh-wasm package
pnpm --filter fast-mesh-wasm-example-react dev    # then open the printed URL (add ?local for the main-thread scene)
```

Vite picks up the package's Worker script from its default `new URL("./worker-entry.js", import.meta.url)`,
so no Vite configuration is needed: `vite build` emits `dist/assets/worker-entry-*.js`.

## Controls

- Click a mesh: select it (a TransformControls gizmo appears). Drag the gizmo: move it (`T`) or rotate it (`R`).
  Each drag is one undo step. `Esc`: deselect.
- Drag empty space: orbit. Wheel: zoom.
- Ctrl+Z: undo. Ctrl+Shift+Z or Ctrl+Y: redo. Or the History buttons.
- Per mesh: colour by distance to the other mesh, opacity, visibility.
- Section: cut both meshes with a plane across z (the +z side stays) and move it with the slider.
- Drop a binary STL on the page: it replaces the selected mesh (or the first), fitted to its place and size.

## Manual test checklist

1. The page shows the sphere above the plate; the status reads "Worker scene: idle" (or "Local scene" with `?local`)
   and the console has no errors.
2. Press "Colour by distance to wavy plate": the button is pressed at once and "computing…" shows until the sphere
   is coloured. Press it again: the colours go.
3. Move the sphere's opacity slider: the sphere fades. Untick the plate's "Visible": it disappears.
4. Press "Cut at z": yellow contours and the −z half is clipped away. Move the offset slider: the cut follows.
5. Click the sphere, drag the gizmo's arrow: the sphere moves and Undo turns on. Ctrl+Z puts it back, Ctrl+Shift+Z
   moves it again. With colouring on, the colours follow the moved sphere.
6. Colour the plate by distance to the sphere, select the sphere and drop a binary STL: it replaces the sphere,
   and the plate's colouring now measures to the STL.
