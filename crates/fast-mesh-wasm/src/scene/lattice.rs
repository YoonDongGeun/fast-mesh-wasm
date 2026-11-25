//! `WasmScene` bindings for lattice infill (`Scene::lattice_fill`,
//! `Scene::lattice_plan`, `Scene::lattice_preview`).
//!
//! `options` is `[kind (0 gyroid, 1 Schwarz P, 2 diamond), cellSize,
//! wallThickness, shellThickness, region (0 auto, 1 inside, 2 cavity),
//! voxelSize (0: from the lengths), maxVoxels, minFragment]`, lengths in
//! world units. A plan is `[region (1 or 2), nx, ny, nz, voxelSize,
//! wallThickness, voxels, triangles, bytes, maxVoxels, fits (0 or 1),
//! solidFraction, regionVolume, cropped (0 or 1)]`.

use fast_mesh_ops::lattice::{LatticeKind, LatticeOptions, LatticePlan, LatticeRegion};
use fast_mesh_store::StoreError;
use wasm_bindgen::prelude::*;

use super::{WasmScene, count};
use crate::error::Error;

/// Values of an encoded plan.
const PLAN: usize = 14;

#[wasm_bindgen]
impl WasmScene {
    /// Plans the voxel grid of `lattice_fill` for the same mesh and `options`
    /// without running it; returns the plan (see the module docs).
    pub fn lattice_plan(
        &self,
        index: u32,
        generation: u32,
        options: &[f64],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let plan = self.scene.lattice_plan(id, &parse_options(options)?)?;
        Ok(encode_plan(&plan).to_vec())
    }

    /// Fills the inside of a closed mesh, or the cavities it encloses, with a
    /// lattice, as one undo step. Returns `[vertices, triangles, fragments,
    /// solidVolume, uvsDropped (0 or 1), ...the plan]`.
    ///
    /// A grid over `maxVoxels` is an error that names its size and memory.
    pub fn lattice_fill(
        &mut self,
        index: u32,
        generation: u32,
        options: &[f64],
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let options = parse_options(options)?;
        // Planned first, for an error with the numbers in it.
        let plan = self.scene.lattice_plan(id, &options)?;
        if !plan.fits {
            return Err(Error::invalid(&too_large(&plan)));
        }
        let r = self.scene.lattice_fill(id, &options)?;
        let mut out = vec![
            count(r.vertices),
            count(r.triangles),
            count(r.fragments),
            r.solid_volume,
            f64::from(u8::from(r.uvs_dropped)),
        ];
        out.extend(encode_plan(&r.plan));
        Ok(out)
    }

    /// The walls `lattice_fill` would put into a mesh, on a grid of at most
    /// `max_voxels` voxels, in the mesh's local frame; the mesh stays as it
    /// is. Returns one `u32` buffer: `[vertices, triangles]`, the plan as
    /// `f64` bits (two words per value, the low one first), the positions as
    /// `f32` bits (3 per vertex) and the indices (3 per triangle).
    pub fn lattice_preview(
        &self,
        index: u32,
        generation: u32,
        options: &[f64],
        max_voxels: u32,
    ) -> Result<Vec<u32>, Error> {
        let id = self.id(index, generation)?;
        let options = parse_options(options)?;
        let p = self
            .scene
            .lattice_preview(id, &options, max_voxels as usize)?;
        let (positions, triangles) = (&p.walls.positions, &p.walls.triangles);
        let mut out: Vec<u32> = Vec::new();
        out.try_reserve_exact(2 + 2 * PLAN + 3 * positions.len() + 3 * triangles.len())
            .map_err(|_| StoreError::OutOfMemory)?;
        out.extend([positions.len() as u32, triangles.len() as u32]);
        for v in encode_plan(&p.plan) {
            let bits = v.to_bits();
            out.extend([bits as u32, (bits >> 32) as u32]);
        }
        out.extend(positions.iter().flatten().map(|c| c.to_bits()));
        out.extend_from_slice(triangles.as_flattened());
        Ok(out)
    }
}

/// What a plan over its limit is refused with.
fn too_large(plan: &LatticePlan) -> String {
    let [nx, ny, nz] = plan.dims;
    format!(
        "lattice grid {nx} x {ny} x {nz} ({} voxels, about {} MB) is over the limit of {} voxels: \
         use a larger voxel size, thicker walls or larger cells",
        plan.voxels,
        plan.bytes.div_ceil(1 << 20),
        plan.max_voxels
    )
}

fn encode_plan(p: &LatticePlan) -> [f64; PLAN] {
    let [nx, ny, nz] = p.dims.map(count);
    [
        match p.region {
            LatticeRegion::Auto => 0.0,
            LatticeRegion::Inside => 1.0,
            LatticeRegion::Cavity => 2.0,
        },
        nx,
        ny,
        nz,
        f64::from(p.voxel_size),
        f64::from(p.wall_thickness),
        p.voxels as f64,
        p.triangles as f64,
        p.bytes as f64,
        count(p.max_voxels),
        f64::from(u8::from(p.fits)),
        p.solid_fraction,
        p.region_volume,
        f64::from(u8::from(p.cropped)),
    ]
}

fn parse_options(flat: &[f64]) -> Result<LatticeOptions, Error> {
    let &[
        kind,
        cell_size,
        wall_thickness,
        shell_thickness,
        region,
        voxel_size,
        max_voxels,
        min_fragment,
    ] = flat
    else {
        return Err(Error::invalid("lattice options must hold 8 values"));
    };
    let max_voxels =
        if max_voxels.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&max_voxels) {
            max_voxels as usize
        } else {
            return Err(Error::invalid("lattice maxVoxels must be a whole number"));
        };
    Ok(LatticeOptions {
        kind: if kind == 0.0 {
            LatticeKind::Gyroid
        } else if kind == 1.0 {
            LatticeKind::SchwarzP
        } else if kind == 2.0 {
            LatticeKind::Diamond
        } else {
            return Err(Error::invalid("lattice kind must be 0, 1 or 2"));
        },
        cell_size: cell_size as f32,
        wall_thickness: wall_thickness as f32,
        shell_thickness: shell_thickness as f32,
        region: if region == 0.0 {
            LatticeRegion::Auto
        } else if region == 1.0 {
            LatticeRegion::Inside
        } else if region == 2.0 {
            LatticeRegion::Cavity
        } else {
            return Err(Error::invalid("lattice region must be 0, 1 or 2"));
        },
        voxel_size: (voxel_size != 0.0).then_some(voxel_size as f32),
        max_voxels,
        min_fragment: min_fragment as f32,
        // The scene puts the placement's in.
        scale: [1.0; 3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;

    const CUBE: [f64; 24] = [
        0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.,
        1.,
    ];
    const CUBE_TRIS: [u32; 36] = [
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];

    fn options() -> [f64; 8] {
        [1.0, 1.0, 0.25, 0.0, 0.0, 0.0625, 16_777_216.0, 0.0]
    }

    #[test]
    fn lattices_plan_first_and_fill_as_one_step() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        s.take_changes();
        let (i, g) = (a[0], a[1]);
        let plan = s.lattice_plan(i, g, &options()).unwrap();
        assert_eq!(plan.len(), PLAN);
        // 16 cells and two empty voxels per side.
        assert_eq!(plan[..6], [1.0, 21.0, 21.0, 21.0, 0.0625, 0.25]);
        assert_eq!(plan[6], 21.0f64.powi(3));
        assert_eq!(plan[8], 5.0 * plan[6] + 64.0 * plan[7]);
        assert_eq!(plan[9..11], [16_777_216.0, 1.0]);
        assert_eq!(plan[12..], [1.0, 0.0]);

        // The preview: the same walls, and no change.
        let preview = s.lattice_preview(i, g, &options(), 20_000).unwrap();
        let (v, t) = (preview[0] as usize, preview[1] as usize);
        assert_eq!(preview.len(), 2 + 2 * PLAN + 3 * v + 3 * t);
        let value = |k: usize| {
            let [lo, hi] = [preview[2 + 2 * k], preview[3 + 2 * k]].map(u64::from);
            f64::from_bits(lo | hi << 32)
        };
        assert_eq!(value(4), 0.0625);
        assert_eq!(value(9), 20_000.0);
        let positions = &preview[2 + 2 * PLAN..][..3 * v];
        assert!(
            positions
                .iter()
                .all(|&b| (-0.6..=0.6).contains(&f32::from_bits(b)))
        );
        assert!(
            preview[2 + 2 * PLAN + 3 * v..]
                .iter()
                .all(|&x| x < v as u32)
        );
        assert!(!s.undo().unwrap());

        let r = s.lattice_fill(i, g, &options()).unwrap();
        assert_eq!(r[5..], plan);
        assert_eq!([r[0], r[1]], [v as f64, t as f64]);
        assert_eq!(f64::from(s.triangle_count(i, g).unwrap()), r[1]);
        assert!(r[3] > 0.4 && r[3] < 0.7, "{}", r[3]);
        assert_eq!(r[4], 0.0);
        assert!(s.undo().unwrap());
        assert_eq!(s.triangle_count(i, g).unwrap(), 12);

        // A shell keeps the surface.
        let mut shelled = options();
        shelled[3] = 0.125;
        let r = s.lattice_fill(i, g, &shelled).unwrap();
        assert_eq!(f64::from(s.triangle_count(i, g).unwrap()), 12.0 + r[1]);
    }

    #[test]
    fn bad_input_and_big_grids_are_refused() {
        let mut s = WasmScene::new();
        let a = s.add_mesh(&CUBE, &CUBE_TRIS, None, 0, false).unwrap();
        let (i, g) = (a[0], a[1]);
        let mut big = options();
        (big[2], big[5]) = (0.02, 0.002);
        let err = s.lattice_fill(i, g, &big).unwrap_err();
        assert_eq!(err.code, codes::INVALID_ARGUMENT);
        for part in ["505 x 505 x 505", "128787625 voxels", "16777216", "MB"] {
            assert!(err.text.contains(part), "{}: {part}", err.text);
        }
        assert_eq!(s.lattice_plan(i, g, &big).unwrap()[10], 0.0);
        for (k, v) in [
            (0, 3.0),
            (0, f64::NAN),
            (1, 0.0),
            (2, -1.0),
            (2, 0.5),
            (3, -0.1),
            (4, 3.0),
            (4, 2.0),
            (5, -0.1),
            (5, 0.2),
            (6, 10.5),
            (6, 10.0),
            (7, -1.0),
        ] {
            let mut bad = options();
            bad[k] = v;
            let code = codes::INVALID_ARGUMENT;
            assert_eq!(
                s.lattice_fill(i, g, &bad).unwrap_err().code,
                code,
                "{k} {v}"
            );
            assert_eq!(
                s.lattice_plan(i, g, &bad).unwrap_err().code,
                code,
                "{k} {v}"
            );
        }
        assert!(s.lattice_fill(i, g, &options()[..7]).is_err());
        assert!(s.lattice_preview(i, g, &options(), 10).is_err());
        for r in [
            s.lattice_fill(99, 0, &options()).map(|_| ()),
            s.lattice_plan(99, 0, &options()).map(|_| ()),
            s.lattice_preview(99, 0, &options(), 1000).map(|_| ()),
        ] {
            assert_eq!(r.unwrap_err().code, codes::UNKNOWN_MESH);
        }
        assert_eq!(s.triangle_count(i, g).unwrap(), 12);
    }
}
