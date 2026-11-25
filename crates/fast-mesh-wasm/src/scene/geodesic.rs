//! `WasmScene` bindings for geodesic distances (`Scene::geodesic_distances`),
//! in one call or in steps.
//!
//! `method` is 0 (dijkstra) or 1 (heatMethod). `max_distance` is in world
//! units, NaN or infinite for no limit; `time_step_scale` is NaN for the
//! default.

use fast_mesh_scene::{GeodesicMethod, GeodesicOptions};
use wasm_bindgen::prelude::*;

use super::WasmScene;
use crate::error::Error;

fn options(method: u8, max_distance: f64, time_step_scale: f64) -> Result<GeodesicOptions, Error> {
    let defaults = GeodesicOptions::default();
    Ok(GeodesicOptions {
        method: match method {
            0 => GeodesicMethod::Dijkstra,
            1 => GeodesicMethod::HeatMethod,
            _ => {
                return Err(Error::invalid(
                    "method must be 0 (dijkstra) or 1 (heatMethod)",
                ));
            }
        },
        max_distance: if max_distance.is_nan() || max_distance == f64::INFINITY {
            None
        } else {
            Some(max_distance as f32)
        },
        time_step_scale: if time_step_scale.is_nan() {
            defaults.time_step_scale
        } else {
            time_step_scale
        },
    })
}

fn no_query() -> Error {
    Error::invalid("no geodesic query is under way with this id")
}

#[wasm_bindgen]
impl WasmScene {
    /// Geodesic distances from `sources` (vertex slot indices) in world
    /// units, one per vertex slot. The heat method's operators are kept for
    /// later queries of the mesh, until it changes or another mesh is
    /// queried.
    pub fn geodesic_distances(
        &self,
        index: u32,
        generation: u32,
        sources: &[u32],
        method: u8,
        max_distance: f64,
        time_step_scale: f64,
    ) -> Result<Vec<f32>, Error> {
        let id = self.id(index, generation)?;
        let options = options(method, max_distance, time_step_scale)?;
        let mut out = Vec::new();
        self.scene
            .geodesic_distances(id, sources, &options, &mut out)?;
        Ok(out)
    }

    /// Starts `geodesic_distances` in steps and returns the query's id.
    /// Nothing is computed yet.
    pub fn begin_geodesic_distances(
        &mut self,
        index: u32,
        generation: u32,
        sources: &[u32],
        method: u8,
        max_distance: f64,
        time_step_scale: f64,
    ) -> Result<u32, Error> {
        let id = self.id(index, generation)?;
        let options = options(method, max_distance, time_step_scale)?;
        let job = self.scene.begin_geodesic_distances(id, sources, &options)?;
        let query = self.next_geodesic;
        self.next_geodesic = self.next_geodesic.wrapping_add(1);
        self.geodesics.insert(query, job);
        Ok(query)
    }

    /// Runs a query over about `work` more elements of its mesh (a pass over
    /// it, a band of the diffusion or an iteration of the solve at least);
    /// `true` once its distances are ready. A query that fails is dropped:
    /// its mesh changed or went, or memory ran out.
    pub fn step_geodesic_distances(&mut self, query: u32, work: u32) -> Result<bool, Error> {
        let mut job = self.geodesics.remove(&query).ok_or_else(no_query)?;
        let done = self
            .scene
            .step_geodesic_distances(&mut job, work as usize)?;
        self.geodesics.insert(query, job);
        Ok(done)
    }

    /// The distances of a query, running whatever is left of it. The query
    /// is over, whatever comes of it.
    pub fn finish_geodesic_distances(&mut self, query: u32) -> Result<Vec<f32>, Error> {
        let job = self.geodesics.remove(&query).ok_or_else(no_query)?;
        Ok(self.scene.finish_geodesic_distances(job)?)
    }

    /// Cancels a query: what it computed is dropped, except the operators of
    /// its mesh when they were ready.
    pub fn drop_geodesic_distances(&mut self, query: u32) {
        self.geodesics.remove(&query);
    }

    /// Frees the operators kept for later heat-method queries.
    pub fn release_geodesics(&self) {
        self.scene.release_geodesics();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(scene: &mut WasmScene) -> [u32; 2] {
        let shape =
            fast_mesh_ops::primitives::generate(&fast_mesh_ops::primitives::Primitive::Icosphere(
                fast_mesh_ops::primitives::IcosphereOptions {
                    radius: 1.0,
                    subdivisions: 3,
                },
            ))
            .unwrap();
        let p: Vec<f64> = shape
            .positions
            .iter()
            .flatten()
            .map(|&c| c.into())
            .collect();
        let id = scene
            .add_mesh(&p, shape.triangles.as_flattened(), None, 0, true)
            .unwrap();
        [id[0], id[1]]
    }

    #[test]
    fn queries_in_steps() {
        let mut s = WasmScene::new();
        let m = sphere(&mut s);
        for method in [0, 1] {
            let whole = s
                .geodesic_distances(m[0], m[1], &[2, 9], method, 1.5, f64::NAN)
                .unwrap();
            assert!(whole.iter().any(|d| d.is_infinite()));
            s.release_geodesics();
            let a = s
                .begin_geodesic_distances(m[0], m[1], &[2, 9], method, 1.5, f64::NAN)
                .unwrap();
            let b = s
                .begin_geodesic_distances(m[0], m[1], &[2, 9], method, 1.5, f64::NAN)
                .unwrap();
            assert_ne!(a, b);
            let mut steps = 0;
            while !s.step_geodesic_distances(a, 1).unwrap() {
                steps += 1;
            }
            assert!(steps > 0);
            assert_eq!(s.finish_geodesic_distances(a).unwrap(), whole);
            assert!(s.finish_geodesic_distances(a).is_err());
            // Cancelled after a step.
            let _ = s.step_geodesic_distances(b, 1).unwrap();
            s.drop_geodesic_distances(b);
            assert!(s.step_geodesic_distances(b, 1).is_err());
            s.drop_geodesic_distances(b);
        }
        assert!(s.geodesics.is_empty());
    }

    #[test]
    fn queries_that_fail_are_dropped() {
        let mut s = WasmScene::new();
        let m = sphere(&mut s);
        assert!(
            s.begin_geodesic_distances(m[0], m[1], &[], 1, f64::NAN, f64::NAN)
                .is_err()
        );
        assert!(
            s.begin_geodesic_distances(m[0], m[1], &[0], 2, f64::NAN, f64::NAN)
                .is_err()
        );
        let q = s
            .begin_geodesic_distances(m[0], m[1], &[0], 1, f64::NAN, f64::NAN)
            .unwrap();
        s.scene.remove_mesh(s.id(m[0], m[1]).unwrap()).unwrap();
        assert!(s.step_geodesic_distances(q, 1).is_err());
        assert!(s.geodesics.is_empty());
    }
}
