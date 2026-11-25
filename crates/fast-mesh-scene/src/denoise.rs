//! Denoising and Delaunay flips of a scene mesh, each one undo step.

use fast_mesh_ops::{
    DelaunayOptions, DelaunayReport, DenoiseOptions, DenoiseReport, delaunay_flips, denoise,
};
use fast_mesh_store::TriangleId;

use crate::{MeshId, Result, Scene};

impl Scene {
    /// Smooths noise out of mesh `id` while keeping its sharp features, on
    /// the triangles in `region` (`None` for the whole mesh), as one undo
    /// step; see [`fast_mesh_ops::denoise`](mod@fast_mesh_ops::denoise).
    /// Vertices that use a triangle outside the region do not move. Angles
    /// and [`DenoiseReport::max_displacement`] are in the mesh's own space,
    /// before its placement's scale.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`](crate::SceneError::UnknownMesh) for a
    /// stale id; [`SceneError::Ops`](crate::SceneError::Ops) for an option
    /// out of range or a dead or out-of-range triangle in `region`;
    /// [`SceneError::Store`](crate::SceneError::Store) when out of memory.
    /// The scene is unchanged on error.
    pub fn denoise(
        &mut self,
        id: MeshId,
        options: &DenoiseOptions,
        region: Option<&[TriangleId]>,
    ) -> Result<DenoiseReport> {
        self.edit(id, |tx| denoise(tx, options, region))
    }

    /// Flips the edges of mesh `id` towards a Delaunay triangulation where
    /// its triangles are nearly coplanar, between two triangles of `region`
    /// (`None` for the whole mesh), as one undo step; see
    /// [`fast_mesh_ops::delaunay`]. No vertex moves. Angles are measured in
    /// the mesh's own space, before its placement's scale.
    ///
    /// # Errors
    /// As for [`Scene::denoise`].
    pub fn delaunay_flips(
        &mut self,
        id: MeshId,
        options: &DelaunayOptions,
        region: Option<&[TriangleId]>,
    ) -> Result<DelaunayReport> {
        self.edit(id, |tx| delaunay_flips(tx, options, region))
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::OpsError;
    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{MeshOptions, SceneError, SceneOptions};

    /// A flat `n x n` grid, sheared so that its diagonals are the long ones,
    /// with a ripple in `z`.
    fn sheared_grid(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                let z = 0.02 * ((7 * x + 13 * y) as f32).sin();
                positions.extend([x as f32 + 0.8 * y as f32, y as f32, z]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Mesh::from_buffers(&positions, &indices).unwrap()
    }

    fn parts(scene: &Scene, id: MeshId) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let m = scene.mesh(id).unwrap();
        (m.positions().to_vec(), m.triangles().to_vec())
    }

    fn scene_with(mesh: Mesh) -> (Scene, MeshId) {
        let mut scene = Scene::new(SceneOptions::default());
        let id = scene.add_mesh(mesh, MeshOptions::default()).unwrap();
        scene.take_changes();
        (scene, id)
    }

    #[test]
    fn each_is_one_undo_step() {
        let (mut scene, id) = scene_with(sheared_grid(8));
        let start = parts(&scene, id);

        let r = scene.denoise(id, &DenoiseOptions::default(), None).unwrap();
        assert!(r.moved > 0);
        assert!(r.after.mean_normal_deviation_deg < r.before.mean_normal_deviation_deg);
        let smooth = parts(&scene, id);
        assert_eq!(smooth.1, start.1, "denoise keeps the triangles");
        let changes = scene.take_changes();
        assert!(changes.iter().any(|c| c.id == id));

        let r = scene
            .delaunay_flips(id, &DelaunayOptions::default(), None)
            .unwrap();
        assert!(r.flips > 0 && r.converged);
        assert!(r.after.min_angle_deg > r.before.min_angle_deg);
        let flipped = parts(&scene, id);
        assert_eq!(flipped.0, smooth.0, "flips move no vertex");

        assert!(scene.undo().unwrap());
        assert_eq!(parts(&scene, id), smooth);
        assert!(scene.undo().unwrap());
        assert_eq!(parts(&scene, id), start);
        assert!(!scene.can_undo());
        assert!(scene.redo().unwrap());
        assert!(scene.redo().unwrap());
        assert_eq!(parts(&scene, id), flipped);

        // Nothing left to flip: nothing recorded.
        scene
            .delaunay_flips(id, &DelaunayOptions::default(), None)
            .unwrap();
        assert!(scene.undo().unwrap());
        assert_eq!(parts(&scene, id), smooth);
    }

    #[test]
    fn a_region_limits_both_and_errors_change_nothing() {
        let (mut scene, id) = scene_with(sheared_grid(8));
        let start = parts(&scene, id);
        // The first three rows of cells.
        let region: Vec<TriangleId> = (0..3 * 16).collect();
        let r = scene
            .denoise(id, &DenoiseOptions::default(), Some(&region))
            .unwrap();
        assert_eq!(r.before.triangles, region.len());
        let after = parts(&scene, id);
        // Vertex rows 3 and up use a triangle outside the region.
        assert!(r.moved > 0 && after.0[3 * 9..] == start.0[3 * 9..]);
        let r = scene
            .delaunay_flips(id, &DelaunayOptions::default(), Some(&region))
            .unwrap();
        assert!(r.flips > 0);
        let after = parts(&scene, id);
        assert_eq!(after.1[region.len()..], start.1[region.len()..]);

        let dead = [10_000];
        let bad = Err(SceneError::Ops(OpsError::InvalidInput(
            "triangle id is dead or out of range",
        )));
        assert_eq!(
            scene.denoise(id, &DenoiseOptions::default(), Some(&dead)),
            bad
        );
        assert_eq!(
            scene
                .delaunay_flips(id, &DelaunayOptions::default(), Some(&dead))
                .map(|_| DenoiseReport::default()),
            bad
        );
        assert_eq!(parts(&scene, id), after);
        scene.remove_mesh(id).unwrap();
        assert_eq!(
            scene.denoise(id, &DenoiseOptions::default(), None),
            Err(SceneError::UnknownMesh)
        );
    }
}
