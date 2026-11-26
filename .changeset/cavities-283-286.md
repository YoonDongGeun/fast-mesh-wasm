---
"fast-mesh-wasm": minor
---

Repairs keep the cavities of hollow solids (#286, #283). A cavity is a closed component wound inward inside the material of another; `MeshComponent.cavity`, from `Scene.analyze`, tells which ones are. `orient` turned cavities outward, so that a void became overlapping material, and `removeSmallComponents` and `autoRepair` deleted small ones, so that the void was filled. Now `orient` keeps a cavity facing inward (and turns an inside-out hollow solid as a whole), and `removeSmallComponents` and `autoRepair` keep the cavities of the components they keep. Pass `keepCavities: false` to `removeSmallComponents` or `repair.clean` to remove small cavities as before.
