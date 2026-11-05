//! Rigid registration of one mesh to another: placements in and out of the
//! [`fast_mesh_ops::icp()`] solver.
//!
//! The solver runs in the fixed mesh's frame: its stored positions, with its
//! scale applied when it has one (on a baked copy), so that distances there
//! are world distances.

use fast_mesh_bvh::Bvh;
use fast_mesh_ops::{IcpOptions, IcpStatus, icp, sample_surface};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError};

use crate::baked::baked_mesh;
use crate::placement::orthonormalize;
use crate::{MeshId, Placement, Result, Scene, SceneError};

/// Options for [`Scene::register`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterOptions {
    /// The solver's options: iterations, tolerance, overlap and maximum
    /// distance, in world units.
    pub icp: IcpOptions,
    /// The most surface points of the moving mesh to use, evenly spaced over
    /// its triangles' centroids and its vertices
    /// ([`fast_mesh_ops::sample_surface`]); at least 1. The `rmse` is measured
    /// on these. Default 100,000.
    pub max_samples: usize,
    /// Whether a converged result becomes the moving mesh's placement, as one
    /// undo step. A result that did not converge is never applied. Default
    /// true.
    pub apply: bool,
}

impl Default for RegisterOptions {
    fn default() -> Self {
        RegisterOptions {
            icp: IcpOptions::default(),
            max_samples: 100_000,
            apply: true,
        }
    }
}

/// What [`Scene::register`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterReport {
    /// The moving mesh's aligned placement, with its scale unchanged: the
    /// converged result, or the best estimate when it did not converge.
    pub placement: Placement,
    /// The root mean square distance, in world units, from the used points to
    /// the fixed mesh at `placement`; infinite when no point has a
    /// correspondence.
    pub rmse: f64,
    /// Solver steps taken.
    pub iterations: usize,
    /// Whether the iteration settled with all six degrees of freedom
    /// constrained. That is not proof of the right alignment, since ICP is
    /// local: judge that from `rmse`.
    pub converged: bool,
    /// Why iteration stopped.
    pub status: IcpStatus,
    /// Whether `placement` was set on the moving mesh.
    pub applied: bool,
    /// Surface points of the moving mesh sampled.
    pub samples: usize,
    /// Sampled points used at `placement`: the rest were trimmed or out of
    /// range.
    pub inliers: usize,
    /// How well the worst-constrained motion is determined; see
    /// [`fast_mesh_ops::IcpReport::conditioning`].
    pub conditioning: f64,
}

/// `a` after `b`.
fn compose(a: &Rigid, b: &Rigid) -> Rigid {
    let (ra, rb) = (&a.rotation, &b.rotation);
    let t = b.translation;
    Rigid {
        rotation: std::array::from_fn(|i| {
            std::array::from_fn(|j| ra[i][0] * rb[0][j] + ra[i][1] * rb[1][j] + ra[i][2] * rb[2][j])
        }),
        translation: std::array::from_fn(|i| {
            ra[i][0] * t[0] + ra[i][1] * t[1] + ra[i][2] * t[2] + a.translation[i]
        }),
    }
}

/// The moving mesh's sampled surface points as `scale * (origin + local)`.
fn scaled_samples(mesh: &Mesh, placement: &Placement, max: usize) -> Result<Vec<[f64; 3]>> {
    let origin = mesh.origin();
    let (s, local) = (placement.scale, sample_surface(mesh, max)?);
    let mut out = Vec::new();
    out.try_reserve_exact(local.len())
        .map_err(|_| SceneError::Store(StoreError::OutOfMemory))?;
    out.extend(
        local
            .iter()
            .map(|p| -> [f64; 3] { std::array::from_fn(|i| s[i] * (origin[i] + p[i])) }),
    );
    Ok(out)
}

impl Scene {
    /// Rigidly aligns the mesh `moving` to the mesh `fixed` by point-to-plane
    /// ICP ([`fast_mesh_ops::icp()`]), using points on `moving`'s surface and
    /// `fixed`'s BVH.
    ///
    /// **The initial guess is the current placements.** ICP is local, so put
    /// the meshes roughly in place first, with [`Scene::set_placement`]; a
    /// good guess is off by a small part of the shape's features.
    ///
    /// **Applying.** With [`RegisterOptions::apply`] (the default) a converged
    /// result becomes `moving`'s placement as one undo step, or as part of the
    /// open step. A result that did not converge changes nothing, whatever
    /// the option, but its best estimate is returned; set it with
    /// [`Scene::set_placement`] if it will do.
    ///
    /// **Scales.** Only rotation and translation are solved. `moving`'s scale
    /// is applied to its points and kept as it is. `fixed`'s scale, uniform
    /// or not, is followed exactly, on a copy of `fixed` baked through it with
    /// its own BVH, so each such call costs a BVH build. Distances and options
    /// are in world units either way.
    ///
    /// **Partial overlap.** Set [`IcpOptions::overlap`] a little below the
    /// fraction of `moving` that lies over `fixed`, or nothing is rejected.
    ///
    /// Only `fixed`'s BVH is used, and only it is brought up to date.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] when the ids are one mesh or `max_samples`
    /// is zero; [`SceneError::StaleBvh`] for a stale Manual `fixed`;
    /// [`SceneError::Ops`] when a mesh has no triangles, an option is invalid,
    /// or a placement takes the points out of `f32` range in `fixed`'s
    /// frame. On error the scene is unchanged.
    pub fn register(
        &mut self,
        moving: MeshId,
        fixed: MeshId,
        options: &RegisterOptions,
    ) -> Result<RegisterReport> {
        self.drop_field_unit();
        self.entry(moving)?;
        self.entry(fixed)?;
        if moving == fixed {
            return Err(SceneError::InvalidInput("cannot register a mesh to itself"));
        }
        if options.max_samples == 0 {
            return Err(SceneError::InvalidInput("max_samples must be at least 1"));
        }
        self.prepare(Some(&[fixed]))?;
        let (m, f) = (self.entry(moving)?, self.entry(fixed)?);
        let points = scaled_samples(&m.mesh, &m.placement, options.max_samples)?;
        let moving_scale = m.placement.scale;
        // The moving vertices, once scaled, to the world.
        let world = Rigid {
            rotation: orthonormalize(m.placement.rotation),
            translation: m.placement.translation,
        };
        // Frame to world: `R (S (o + p)) + t` with frame coordinates `S p`.
        let (s, o) = (f.placement.scale, f.mesh.origin());
        let r = orthonormalize(f.placement.rotation);
        let scaled_origin: [f64; 3] = std::array::from_fn(|i| s[i] * o[i]);
        let frame = Rigid {
            rotation: r,
            translation: std::array::from_fn(|i| {
                r[i][0] * scaled_origin[0]
                    + r[i][1] * scaled_origin[1]
                    + r[i][2] * scaled_origin[2]
                    + f.placement.translation[i]
            }),
        };
        let initial = compose(&frame.inverse(), &world);
        let baked: Option<(Mesh, Bvh)> = if f.placement.is_scaled() {
            Some(baked_mesh(
                &f.mesh,
                |p| std::array::from_fn(|i| s[i] * f64::from(p[i])),
                f.bvh.options(),
            )?)
        } else {
            None
        };
        let (mesh, bvh) = baked
            .as_ref()
            .map_or((&f.mesh, &f.bvh), |(mesh, bvh)| (mesh, bvh));
        let solved = icp(&points, mesh, bvh, &initial, &options.icp)?;
        let out = compose(&frame, &solved.transform);
        let placement = Placement {
            rotation: out.rotation,
            translation: out.translation,
            scale: moving_scale,
        };
        let converged = solved.converged();
        let applied = options.apply && converged;
        if applied {
            self.set_placement(moving, placement)?;
        }
        Ok(RegisterReport {
            placement,
            rmse: solved.rmse,
            iterations: solved.iterations,
            converged,
            status: solved.status,
            applied,
            samples: solved.samples,
            inliers: solved.inliers,
            conditioning: solved.conditioning,
        })
    }
}
