//! `WasmScene` bindings for wraps (`Scene::wrap`, `Scene::wrap_plan`).

use fast_mesh_ops::wrap::{WrapOptions, WrapPlan};
use wasm_bindgen::prelude::*;

use super::{WasmScene, count};
use crate::error::Error;

#[wasm_bindgen]
impl WasmScene {
    /// Plans the voxel grid of `wrap_meshes` for the same `meshes` and
    /// `options` without running it. Returns `[nx, ny, nz, voxelSize, radius,
    /// voxels, bytes, maxVoxels, fits (0 or 1)]`.
    pub fn wrap_plan(&self, meshes: &[u32], options: &[f64]) -> Result<Vec<f64>, Error> {
        let ids = self.ids(meshes)?;
        let plan = self.scene.wrap_plan(&ids, &parse_options(options)?)?;
        Ok(encode_plan(&plan))
    }

    /// Adds a closed, manifold, outward-wound envelope without
    /// self-intersections around the meshes `meshes` (flat `[index,
    /// generation, ...]`) as a new mesh, as one undo step (undo removes it).
    /// `options` is `[gap, voxelSize (0: from the detail), detail, project (0
    /// or 1), maxVoxels]`, lengths in world units. `normals` turns the new
    /// mesh's smooth vertex normals on. Returns `[index, generation, vertices,
    /// triangles, projected, ...the plan as wrap_plan returns it]`.
    ///
    /// A grid over `maxVoxels` is an error that names its size and memory.
    pub fn wrap_meshes(
        &mut self,
        meshes: &[u32],
        options: &[f64],
        normals: bool,
    ) -> Result<Vec<f64>, Error> {
        let ids = self.ids(meshes)?;
        let options = parse_options(options)?;
        // Planned first, for an error with the numbers in it.
        let plan = self.scene.wrap_plan(&ids, &options)?;
        if !plan.fits {
            return Err(Error::invalid(&too_large(&plan)));
        }
        let r = self.scene.wrap(&ids, &options)?;
        let added = self.added(r.added, normals)?;
        let mut out: Vec<f64> = added.into_iter().map(f64::from).collect();
        out.extend([r.vertices, r.triangles, r.projected].map(count));
        out.extend(encode_plan(&r.plan));
        Ok(out)
    }
}

/// What a plan over its limit is refused with.
fn too_large(plan: &WrapPlan) -> String {
    let [nx, ny, nz] = plan.dims;
    format!(
        "wrap grid {nx} x {ny} x {nz} ({} voxels, about {} MB) is over the limit of {} voxels: \
         use a larger voxel size or less detail",
        plan.voxels,
        plan.bytes.div_ceil(1 << 20),
        plan.max_voxels
    )
}

fn encode_plan(plan: &WrapPlan) -> Vec<f64> {
    let [nx, ny, nz] = plan.dims.map(count);
    vec![
        nx,
        ny,
        nz,
        f64::from(plan.voxel_size),
        f64::from(plan.radius),
        plan.voxels as f64,
        plan.bytes as f64,
        count(plan.max_voxels),
        f64::from(u8::from(plan.fits)),
    ]
}

fn parse_options(flat: &[f64]) -> Result<WrapOptions, Error> {
    let &[gap, voxel_size, detail, project, max_voxels] = flat else {
        return Err(Error::invalid("wrap options must hold 5 values"));
    };
    let whole = |v: f64, what: &str| {
        if v.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&v) {
            Ok(v as u32)
        } else {
            Err(Error::invalid(what))
        }
    };
    Ok(WrapOptions {
        gap: gap as f32,
        voxel_size: (voxel_size != 0.0).then_some(voxel_size as f32),
        detail: whole(detail, "wrap detail must be a whole number")?,
        project: project != 0.0,
        max_voxels: whole(max_voxels, "wrap maxVoxels must be a whole number")? as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;
    use crate::render::{FLAG_ADDED, FLAG_REMOVED, FLAG_RESTORABLE};

    const CUBE: [f64; 24] = [
        0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
        1.,
    ];
    const CUBE_TRIS: [u32; 36] = [
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];

    fn options() -> [f64; 5] {
        [0.2, 0.0, 16.0, 0.0, 16_777_216.0]
    }

    #[test]
    fn wraps_plan_first_and_come_back_as_one_mesh() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        // Touches the first cube with a face.
        let moved: Vec<f64> = CUBE
            .iter()
            .enumerate()
            .map(|(i, &c)| if i % 3 == 0 { c + 1.0 } else { c })
            .collect();
        let b = s.add_mesh(&moved, &CUBE_TRIS, None, 0, false).unwrap();
        s.take_changes();
        let both = [a[0], a[1], b[0], b[1]];

        let plan = s.wrap_plan(&both, &options()).unwrap();
        // 2 / 16 voxels; half the gap is under one, so the radius is one.
        assert_eq!(plan[..5], [25.0, 17.0, 17.0, 0.125, 0.125]);
        assert_eq!(plan[5], 25.0 * 17.0 * 17.0);
        assert_eq!(plan[6], plan[5] * 17.0);
        assert_eq!(plan[7..], [16_777_216.0, 1.0]);

        let r = s.wrap_meshes(&both, &options(), true).unwrap();
        assert_eq!(r.len(), 5 + plan.len());
        assert_eq!(r[5..], plan);
        let (i, g) = (r[0] as u32, r[1] as u32);
        assert_eq!(f64::from(s.triangle_slots(i, g).unwrap()), r[3]);
        assert_eq!(r[4], 0.0);
        let mut projected = options();
        projected[3] = 1.0;
        assert!(s.wrap_meshes(&both, &projected, false).unwrap()[4] > 0.0);
        // One undo step removes each wrap (#271).
        assert!(s.undo().unwrap());
        assert!(s.mesh(i, g).is_ok());
        s.take_changes();
        assert!(s.undo().unwrap());
        assert!(s.mesh(i, g).is_err());
        assert!(s.is_restorable(i, g));
        let gone = s.take_changes();
        assert_eq!(gone[..4], [1, i, g, FLAG_REMOVED | FLAG_RESTORABLE]);
        assert!(!s.undo().unwrap());
        assert!(s.redo().unwrap());
        assert_eq!(f64::from(s.triangle_slots(i, g).unwrap()), r[3]);
        let back = s.take_changes();
        assert_eq!(back[..3], [1, i, g]);
        assert_ne!(back[3] & FLAG_ADDED, 0);
    }

    #[test]
    fn bad_input_and_big_grids_are_refused() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        let mut big = options();
        big[2] = 2000.0;
        let err = s.wrap_meshes(&a, &big, false).unwrap_err();
        assert_eq!(err.code, codes::INVALID_ARGUMENT);
        for part in ["2407 x 2407 x 2407", "13945313143 voxels", "16777216", "MB"] {
            assert!(err.text.contains(part), "{}: {part}", err.text);
        }
        assert_eq!(s.wrap_plan(&a, &big).unwrap()[8], 0.0);
        for (i, v) in [
            (0, -1.0),
            (0, f64::NAN),
            (1, -2.0),
            (2, 4.0),
            (2, 8.5),
            (4, 1.0),
        ] {
            let mut bad = options();
            bad[i] = v;
            assert_eq!(
                s.wrap_meshes(&a, &bad, false).unwrap_err().code,
                codes::INVALID_ARGUMENT,
                "{i} {v}"
            );
            assert!(s.wrap_plan(&a, &bad).is_err());
        }
        assert!(s.wrap_meshes(&a, &options()[..4], false).is_err());
        assert_eq!(
            s.wrap_meshes(&[], &options(), false).unwrap_err().code,
            codes::INVALID_ARGUMENT
        );
        assert_eq!(
            s.wrap_meshes(&[99, 0], &options(), false).unwrap_err().code,
            codes::UNKNOWN_MESH
        );
    }
}
