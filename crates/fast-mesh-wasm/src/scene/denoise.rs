//! `WasmScene` bindings for denoising and Delaunay flips
//! (`Scene::denoise`, `Scene::delaunay_flips`).
//!
//! Options are numbers, NaN for the default. `region` is a list of triangle
//! ids, `undefined` for the whole mesh. A report ends with the region's
//! stats before and after, each `triangles, min angle, mean normal
//! deviation` with the angles in degrees.

use fast_mesh_ops::{DelaunayOptions, DenoiseOptions, SurfaceStats};
use wasm_bindgen::prelude::*;

use super::WasmScene;
use crate::edit::count_opt;
use crate::error::Error;

#[wasm_bindgen]
impl WasmScene {
    /// Smooths noise out of a mesh while keeping edges sharper than
    /// `feature_angle` degrees, in `iterations` rounds of `strength` in
    /// `[0, 1]`, as one undo step. Returns `[moved, maxDisplacement,
    /// iterations, reverted, before × 3, after × 3]` (see the module docs).
    pub fn denoise(
        &mut self,
        index: u32,
        generation: u32,
        iterations: f64,
        strength: f64,
        feature_angle: f64,
        region: Option<Vec<u32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let d = DenoiseOptions::default();
        let options = DenoiseOptions {
            iterations: match count_opt(iterations, "iterations")? {
                // Too many for `u32` is too many for the op as well.
                Some(n) => u32::try_from(n).unwrap_or(u32::MAX),
                None => d.iterations,
            },
            strength: or_default(strength, d.strength),
            feature_angle_deg: or_default(feature_angle, d.feature_angle_deg),
        };
        let r = self.scene.denoise(id, &options, region.as_deref())?;
        let mut out = vec![
            r.moved as f64,
            r.max_displacement,
            f64::from(r.iterations),
            r.reverted as f64,
        ];
        out.extend(stats(&r.before));
        out.extend(stats(&r.after));
        Ok(out)
    }

    /// Flips a mesh's edges towards a Delaunay triangulation where that
    /// takes the surface at most `dihedral_limit` degrees from the original,
    /// as one undo step. Returns `[flips, passes, converged (0/1), before ×
    /// 3, after × 3]` (see the module docs).
    pub fn delaunay_flips(
        &mut self,
        index: u32,
        generation: u32,
        dihedral_limit: f64,
        region: Option<Vec<u32>>,
    ) -> Result<Vec<f64>, Error> {
        let id = self.id(index, generation)?;
        let options = DelaunayOptions {
            dihedral_limit_deg: or_default(
                dihedral_limit,
                DelaunayOptions::default().dihedral_limit_deg,
            ),
        };
        let r = self.scene.delaunay_flips(id, &options, region.as_deref())?;
        let mut out = vec![r.flips as f64, r.passes as f64, f64::from(r.converged)];
        out.extend(stats(&r.before));
        out.extend(stats(&r.after));
        Ok(out)
    }
}

/// `v` as `f32`, or `default` for NaN; the op checks its range.
fn or_default(v: f64, default: f32) -> f32 {
    if v.is_nan() { default } else { v as f32 }
}

fn stats(s: &SurfaceStats) -> [f64; 3] {
    [
        s.triangles as f64,
        s.min_angle_deg,
        s.mean_normal_deviation_deg,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;

    /// A 4 x 4 grid, sheared so that its diagonals are the long ones, with a
    /// ripple in `z`.
    fn grid(s: &mut WasmScene) -> (u32, u32) {
        let n = 4u32;
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let z = 0.05 * f64::from(7 * x + 13 * y).sin();
                positions.extend([f64::from(x) + 0.8 * f64::from(y), f64::from(y), z]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let id = s.add_mesh(&positions, &indices, None, 0, false).unwrap();
        (id[0], id[1])
    }

    const NAN: f64 = f64::NAN;

    #[test]
    fn reports_decode_and_each_is_one_undo_step() {
        let mut s = WasmScene::new();
        let (i, g) = grid(&mut s);
        let r = s.denoise(i, g, NAN, NAN, NAN, None).unwrap();
        // moved, displacement, iterations, reverted, then 2 x 3 stats.
        assert_eq!(r.len(), 10);
        assert!(r[0] > 0.0 && r[1] > 0.0);
        assert_eq!((r[2], r[3]), (5.0, 0.0));
        assert_eq!((r[4], r[7]), (32.0, 32.0));
        assert!(r[9] < r[6], "the normal deviation drops: {r:?}");
        let r = s.delaunay_flips(i, g, 20.0, None).unwrap();
        assert_eq!(r.len(), 9);
        assert!(r[0] > 0.0 && r[1] > 0.0);
        assert_eq!(r[2], 1.0);
        assert_eq!((r[3], r[6]), (32.0, 32.0));
        assert!(r[7] >= r[4], "the smallest angle rises: {r:?}");
        assert!(s.undo().unwrap());
        assert!(s.undo().unwrap());
        assert!(!s.undo().unwrap());
    }

    #[test]
    fn regions_options_and_errors() {
        let mut s = WasmScene::new();
        let (i, g) = grid(&mut s);
        let r = s
            .denoise(i, g, 1.0, 1.0, 60.0, Some(vec![0, 1, 2, 3, 8, 9, 10, 11]))
            .unwrap();
        assert_eq!((r[2], r[4]), (1.0, 8.0));
        let r = s.delaunay_flips(i, g, NAN, Some(vec![])).unwrap();
        assert_eq!((r[0], r[2], r[3]), (0.0, 1.0, 0.0));

        let invalid = |r: Result<Vec<f64>, Error>| {
            assert_eq!(r.unwrap_err().code, codes::INVALID_ARGUMENT);
        };
        invalid(s.denoise(i, g, 1.5, NAN, NAN, None));
        invalid(s.denoise(i, g, 1e12, NAN, NAN, None));
        invalid(s.denoise(i, g, NAN, 2.0, NAN, None));
        invalid(s.denoise(i, g, NAN, NAN, 0.0, None));
        invalid(s.denoise(i, g, NAN, NAN, NAN, Some(vec![999])));
        invalid(s.delaunay_flips(i, g, -1.0, None));
        invalid(s.delaunay_flips(i, g, f64::INFINITY, None));
        invalid(s.delaunay_flips(i, g, NAN, Some(vec![999])));
        assert_eq!(
            s.delaunay_flips(99, 0, NAN, None).unwrap_err().code,
            codes::UNKNOWN_MESH
        );
    }
}
