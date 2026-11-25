//! Geodesic distances of scene meshes, in one call or in steps.
//!
//! The heat method's operators cost as much as a query, so the scene keeps
//! those of the mesh queried last and reuses them until the mesh changes.

use std::sync::{Arc, Mutex, PoisonError};

use fast_mesh_fields::{GeodesicHeatSolver, GeodesicPhase, GeodesicQuery};
use fast_mesh_store::Mesh;

use crate::baked::baked_copy;
use crate::scene::Entry;
use crate::{GeodesicOptions, MeshId, Result, Scene, SceneError};

/// What an earlier query of a mesh left for the next.
#[derive(Debug, Clone)]
struct Kept {
    solver: Arc<GeodesicHeatSolver>,
    /// The scaled copy the operators are of, and its scale: only for a mesh
    /// whose placement scales it differently along its axes.
    scaled: Option<([f64; 3], Arc<Mesh>)>,
}

/// The heat method's operators kept for a mesh, if any.
#[derive(Debug, Default)]
pub(crate) struct GeodesicCache(Mutex<Option<Kept>>);

impl GeodesicCache {
    /// Frees the operators: the mesh changed, or left the scene.
    pub(crate) fn clear(&mut self) {
        *self.0.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
    }

    fn get(&self) -> Option<Kept> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn set(&self, kept: Option<Kept>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = kept;
    }
}

/// A geodesic distance query of a scene mesh under way, from
/// [`Scene::begin_geodesic_distances`]. Dropping it cancels the query.
#[derive(Debug)]
pub struct GeodesicJob {
    id: MeshId,
    /// The mesh's content stamp at the start.
    content: u64,
    query: GeodesicQuery,
    /// The scaled copy the query runs on, under a non-uniform scale.
    scaled: Option<([f64; 3], Arc<Mesh>)>,
    /// What the distances on the mesh the query runs on are multiplied by.
    scale: f64,
    /// The operators are kept in the scene (or there are none to keep).
    kept: bool,
}

impl GeodesicJob {
    /// The mesh the query is of.
    #[must_use]
    pub fn mesh(&self) -> MeshId {
        self.id
    }

    /// What the next step does.
    #[must_use]
    pub fn phase(&self) -> GeodesicPhase {
        self.query.phase()
    }
}

impl Entry {
    /// The mesh's scaled copy for queries under its non-uniform scale.
    fn scaled_copy(&self, scale: [f64; 3]) -> Result<Arc<Mesh>> {
        let copy = baked_copy(&self.mesh, |p| {
            std::array::from_fn(|k| scale[k] * f64::from(p[k]))
        })?;
        Ok(Arc::new(copy))
    }
}

impl Scene {
    /// Computes geodesic surface distances from `sources` (vertex slot ids on
    /// `id`) in world length units.
    ///
    /// Writes `mesh.vertex_slots()` distances into `out` (with `f32::INFINITY`
    /// for dead slots, unreachable components, or distances exceeding
    /// `options.max_distance`). Does not modify the scene or its history.
    ///
    /// The heat method's operators are kept for later queries of the mesh,
    /// until it changes or another mesh is queried.
    pub fn geodesic_distances(
        &self,
        id: MeshId,
        sources: &[u32],
        options: &GeodesicOptions,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let job = self.begin_geodesic_distances(id, sources, options)?;
        *out = self.finish_geodesic_distances(job)?;
        Ok(())
    }

    /// Starts [`Scene::geodesic_distances`] in steps: nothing is computed
    /// yet. The distances are the ones of the mesh and its placement's scale
    /// as they are now.
    ///
    /// # Errors
    /// As for [`Scene::geodesic_distances`], for what is wrong with the
    /// arguments.
    pub fn begin_geodesic_distances(
        &self,
        id: MeshId,
        sources: &[u32],
        options: &GeodesicOptions,
    ) -> Result<GeodesicJob> {
        let entry = self.entry(id)?;
        let kept = entry.geodesics.get();
        let mut local = *options;
        let (scale, scaled) = match entry.placement.uniform_scale() {
            Some(s) => {
                if let Some(d) = options.max_distance
                    && d.is_finite()
                    && d >= 0.0
                {
                    let d = (f64::from(d) / s) as f32;
                    // (Past the range of f32 it cuts nothing off.)
                    local.max_distance = d.is_finite().then_some(d);
                }
                (s, None)
            }
            None => {
                let axes = entry.placement.scale;
                let copy = match kept.as_ref().and_then(|k| k.scaled.as_ref()) {
                    Some((of, copy)) if *of == axes => Arc::clone(copy),
                    _ => entry.scaled_copy(axes)?,
                };
                (1.0, Some((axes, copy)))
            }
        };
        let mesh = scaled.as_ref().map_or(&entry.mesh, |(_, m)| m.as_ref());
        let query = match kept {
            Some(k) => GeodesicQuery::with_solver(mesh, sources, &local, k.solver)?,
            None => GeodesicQuery::new(mesh, sources, &local)?,
        };
        Ok(GeodesicJob {
            id,
            content: entry.content,
            kept: false,
            query,
            scaled,
            scale,
        })
    }

    /// Runs a query over about `work` more elements of its mesh (at least
    /// one pass, unless it is done); returns whether the distances are
    /// ready. How it is cut into steps does not change them.
    ///
    /// The operators it builds are kept as soon as they are, so that a query
    /// after a cancelled one can start with them.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] when the mesh is gone;
    /// [`SceneError::InvalidInput`] when it has changed since the query
    /// started; as for [`Scene::geodesic_distances`] otherwise. The job is
    /// of no use after an error.
    pub fn step_geodesic_distances(&self, job: &mut GeodesicJob, work: usize) -> Result<bool> {
        let entry = self.entry(job.id)?;
        if entry.content != job.content {
            return Err(SceneError::InvalidInput(
                "the mesh changed while its geodesic distances were computed",
            ));
        }
        let mesh = job.scaled.as_ref().map_or(&entry.mesh, |(_, m)| m.as_ref());
        let done = job.query.step(mesh, work)?;
        if !job.kept
            && let Some(solver) = job.query.solver()
        {
            job.kept = true;
            for (id, other) in self.entries.iter() {
                if id != job.id {
                    other.geodesics.set(None);
                }
            }
            entry.geodesics.set(Some(Kept {
                solver: Arc::clone(solver),
                scaled: job.scaled.clone(),
            }));
        }
        Ok(done)
    }

    /// The distances of a query, running whatever is left of it.
    ///
    /// # Errors
    /// As for [`Scene::step_geodesic_distances`].
    pub fn finish_geodesic_distances(&self, mut job: GeodesicJob) -> Result<Vec<f32>> {
        while !self.step_geodesic_distances(&mut job, usize::MAX)? {}
        let mut out = job.query.into_distances().unwrap_or_default();
        if job.scale != 1.0 {
            for d in out.iter_mut().filter(|d| d.is_finite()) {
                *d = (f64::from(*d) * job.scale) as f32;
            }
        }
        Ok(out)
    }

    /// Frees the heat method's operators kept for later queries (see
    /// [`Scene::geodesic_distances`]).
    pub fn release_geodesics(&self) {
        for (_, entry) in self.entries.iter() {
            entry.geodesics.set(None);
        }
    }
}
