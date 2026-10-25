//! Joining two meshes: append one's geometry into the other and bridge a
//! loop on each, as one undo step, then remove the emptied mesh.

use fast_mesh_ops::append::{AppendSource, append_mesh};
use fast_mesh_ops::bridge::{BridgeOptions, bridge_loops};
use fast_mesh_ops::{LoopId, OpsError, boundary_loops};
use fast_mesh_store::{Mesh, StoreError, VertexId};

use crate::placement::relative_rigid;
use crate::{MeshId, Result, Scene, SceneError};

/// What to join: a loop on the target and a loop on the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JoinOptions {
    /// The loop on the target mesh, from its [`boundary_loops`].
    pub loop_a: LoopId,
    /// The loop on the source mesh, from its [`boundary_loops`].
    pub loop_b: LoopId,
    /// Maximum vertices per loop; `None` means the bridge default (128).
    pub max_edges: Option<usize>,
}

/// What [`Scene::join`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JoinReport {
    /// Source vertices copied into the target.
    pub vertices_added: usize,
    /// Source triangles copied into the target.
    pub triangles_added: usize,
    /// Bridge triangles between the two loops.
    pub bridge_triangles_added: usize,
}

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

/// The source mesh's alive geometry in the target's local frame, with the
/// source loop's vertices in loop order.
struct Source {
    source: AppendSource,
    /// Source position index per source-mesh vertex slot (`u32::MAX` for a
    /// dead slot).
    index_of: Vec<u32>,
    /// The source loop's vertices, in loop order.
    loop_vertices: Vec<VertexId>,
}

/// Copies the source mesh's alive positions (mapped into the target's local
/// frame) and triangles into an [`AppendSource`], resolving the source loop
/// on the way. Reads only; the scene is untouched.
fn prepare_source(
    target: &Mesh,
    target_placement: &crate::Placement,
    source: &Mesh,
    source_placement: &crate::Placement,
    loop_b: LoopId,
) -> Result<Source> {
    let rigid = relative_rigid(
        target_placement,
        target.origin(),
        source_placement,
        source.origin(),
    );
    let loops = boundary_loops(source)?;
    let loop_vertices = loops
        .iter()
        .find(|l| l.id == loop_b)
        .map(|l| l.vertices.clone())
        .ok_or(OpsError::StaleLoops)?;
    let mut index_of: Vec<u32> = Vec::new();
    index_of
        .try_reserve(source.vertex_slots())
        .map_err(|_| oom())?;
    index_of.resize(source.vertex_slots(), u32::MAX);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    positions
        .try_reserve(source.vertex_count())
        .map_err(|_| oom())?;
    for (v, slot) in index_of.iter_mut().enumerate() {
        let Some(p) = source.position(v as VertexId) else {
            continue;
        };
        let w = rigid.apply(p);
        let q = [w[0] as f32, w[1] as f32, w[2] as f32];
        if !q.iter().all(|c| c.is_finite()) {
            return Err(SceneError::InvalidInput("joined positions must be finite"));
        }
        let idx = u32::try_from(positions.len()).map_err(|_| oom())?;
        *slot = idx;
        positions.push(q);
    }
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    triangles
        .try_reserve(source.triangle_count())
        .map_err(|_| oom())?;
    for t in 0..source.triangle_slots() {
        let Some(c) = source.triangle(t as u32) else {
            continue;
        };
        let mapped = c.map(|v| index_of[v as usize]);
        debug_assert!(
            mapped.iter().all(|&m| m != u32::MAX),
            "live triangle on a dead vertex"
        );
        triangles.push(mapped);
    }
    Ok(Source {
        source: AppendSource {
            positions,
            triangles,
        },
        index_of,
        loop_vertices,
    })
}

impl Scene {
    /// Joins the source mesh into the target: copies the source's alive
    /// positions (mapped into the target's local frame, so the world shape
    /// is kept) and triangles into the target, bridges `loop_a` to the
    /// copied `loop_b`, and removes the source mesh. The copy and the bridge
    /// run in one edit, so they undo together; the removal is permanent,
    /// like [`Scene::remove_mesh`].
    ///
    /// Only positions and triangles cross over: the source's channels are
    /// dropped, and the target's channels take their defaults on the new
    /// vertices.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] when the two ids are one mesh;
    /// [`SceneError::Ops`] with [`OpsError::StaleLoops`] for a loop that is
    /// not there, or [`OpsError::InvalidInput`] when the loops cannot bridge
    /// (unclean, over `max_edges`, winding the same way). A failed join
    /// changes nothing: both meshes are left as they were. If removing the
    /// source fails (out of memory), the joined geometry stays recorded and
    /// the source mesh with it.
    pub fn join(&mut self, a: MeshId, b: MeshId, options: &JoinOptions) -> Result<JoinReport> {
        if a == b {
            return Err(SceneError::InvalidInput("cannot join a mesh to itself"));
        }
        let prepared = {
            let target = self.entry(a)?;
            let source = self.entry(b)?;
            prepare_source(
                &target.mesh,
                &target.placement,
                &source.mesh,
                &source.placement,
                options.loop_b,
            )?
        };
        let Source {
            source,
            index_of,
            loop_vertices,
        } = prepared;
        let report = self.edit(a, |tx| -> std::result::Result<JoinReport, OpsError> {
            let appended = append_mesh(tx, &source)?;
            // The copied loop is the one loop whose vertices are exactly the
            // copied source loop's: appended vertices are new ids, referenced
            // only by the copied triangles, so no other loop can hold them.
            let mapped = |v: VertexId| appended.vertex_map[index_of[v as usize] as usize];
            let loops = boundary_loops(tx.mesh())?;
            let copied = loops
                .iter()
                .find(|l| {
                    l.vertices.len() == loop_vertices.len()
                        && loop_vertices
                            .iter()
                            .all(|&v| l.vertices.contains(&mapped(v)))
                })
                .map(|l| l.id)
                .ok_or(OpsError::StaleLoops)?;
            let bridged = bridge_loops(
                tx,
                &BridgeOptions {
                    loops: (options.loop_a, copied),
                    max_edges: options.max_edges,
                },
            )?;
            Ok(JoinReport {
                vertices_added: appended.vertices_added,
                triangles_added: appended.triangles_added,
                bridge_triangles_added: bridged.triangles_added,
            })
        })?;
        self.remove_mesh(b)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::{LoopId, OpsError};
    use fast_mesh_store::Mesh;

    use crate::{MeshOptions, Placement, Scene, SceneError, SceneOptions};

    fn quad_a() -> Mesh {
        Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 2, 1, 0, 3, 2],
        )
        .unwrap()
    }

    fn quad_b() -> Mesh {
        Mesh::from_buffers(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap()
    }

    fn loop_of(mesh: &Mesh, min_vertex: u32) -> LoopId {
        fast_mesh_ops::boundary_loops(mesh)
            .unwrap()
            .iter()
            .find(|l| l.id.min_vertex == min_vertex)
            .unwrap()
            .id
    }

    fn scene_of(a: Mesh, b: Mesh) -> Scene {
        let mut scene = Scene::new(SceneOptions::default());
        scene.add_mesh(a, MeshOptions::default()).unwrap();
        scene.add_mesh(b, MeshOptions::default()).unwrap();
        scene
    }

    #[test]
    fn join_merges_source_into_target_and_removes_it() {
        let a_mesh = quad_a();
        let b_mesh = quad_b();
        let loop_a = loop_of(&a_mesh, 0);
        let loop_b = loop_of(&b_mesh, 0);
        let mut scene = scene_of(a_mesh, b_mesh);
        let ids: Vec<_> = scene.ids().collect();
        let (a, b) = (ids[0], ids[1]);

        let report = scene
            .join(
                a,
                b,
                &crate::JoinOptions {
                    loop_a,
                    loop_b,
                    max_edges: None,
                },
            )
            .unwrap();
        assert_eq!(report.vertices_added, 4);
        assert_eq!(report.triangles_added, 2);
        assert_eq!(report.bridge_triangles_added, 8);
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 2 + 2 + 8);
        assert_eq!(scene.mesh(b).unwrap_err(), SceneError::UnknownMesh);

        // One undo step restores the target; the source stays removed.
        assert!(scene.undo().unwrap());
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 2);
        assert_eq!(scene.mesh(b).unwrap_err(), SceneError::UnknownMesh);
        assert!(scene.redo().unwrap());
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 2 + 2 + 8);
    }

    fn join_options(mesh_a: &Mesh, mesh_b: &Mesh) -> crate::JoinOptions {
        crate::JoinOptions {
            loop_a: loop_of(mesh_a, 0),
            loop_b: loop_of(mesh_b, 0),
            max_edges: None,
        }
    }

    #[test]
    fn join_rejects_same_mesh() {
        let a_mesh = quad_a();
        let b_mesh = quad_b();
        let options = join_options(&a_mesh, &b_mesh);
        let mut scene = scene_of(a_mesh, b_mesh);
        let ids: Vec<_> = scene.ids().collect();
        let (a, b) = (ids[0], ids[1]);

        assert_eq!(
            scene.join(a, a, &options),
            Err(SceneError::InvalidInput("cannot join a mesh to itself"))
        );
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 2);
        assert_eq!(scene.mesh(b).unwrap().triangle_count(), 2);
        assert!(!scene.can_undo());
    }

    #[test]
    fn join_rejects_stale_loops_and_changes_nothing() {
        let a_mesh = quad_a();
        let b_mesh = quad_b();
        let fake = LoopId {
            min_vertex: 99,
            len: 4,
            hash: 7,
        };
        let mut scene = scene_of(a_mesh, b_mesh);
        let ids: Vec<_> = scene.ids().collect();
        let (a, b) = (ids[0], ids[1]);
        let options = join_options(scene.mesh(a).unwrap(), scene.mesh(b).unwrap());

        // Unknown source loop: rejected before anything is touched.
        let bad_b = crate::JoinOptions {
            loop_b: fake,
            ..options
        };
        assert_eq!(
            scene.join(a, b, &bad_b),
            Err(SceneError::Ops(OpsError::StaleLoops))
        );
        // Unknown target loop: the bridge finds nothing; the edit rolls back.
        let bad_a = crate::JoinOptions {
            loop_a: fake,
            ..options
        };
        assert_eq!(
            scene.join(a, b, &bad_a),
            Err(SceneError::Ops(OpsError::StaleLoops))
        );
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 2);
        assert_eq!(scene.mesh(b).unwrap().triangle_count(), 2);
        assert!(!scene.can_undo());
    }

    #[test]
    fn join_rejects_same_winding_and_changes_nothing() {
        let a_mesh = quad_a();
        // Quad B facing -z like A: both rims wind the same way, so no
        // outward tube exists.
        let b_mesh = Mesh::from_buffers(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            &[0, 2, 1, 0, 3, 2],
        )
        .unwrap();
        let options = join_options(&a_mesh, &b_mesh);
        let mut scene = scene_of(a_mesh, b_mesh);
        let ids: Vec<_> = scene.ids().collect();
        let (a, b) = (ids[0], ids[1]);

        assert!(matches!(
            scene.join(a, b, &options),
            Err(SceneError::Ops(OpsError::InvalidInput(_)))
        ));
        assert_eq!(scene.mesh(a).unwrap().triangle_count(), 2);
        assert_eq!(scene.mesh(b).unwrap().triangle_count(), 2);
        assert!(!scene.can_undo());
    }

    #[test]
    fn join_maps_source_into_target_frame() {
        let a_mesh = quad_a();
        // Quad B in its own local frame at z=0 facing +z, placed at z=1.
        let b_mesh = Mesh::from_buffers(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        let options = join_options(&a_mesh, &b_mesh);
        let mut scene = Scene::new(SceneOptions::default());
        let a = scene.add_mesh(a_mesh, MeshOptions::default()).unwrap();
        let b = scene
            .add_mesh(
                b_mesh,
                MeshOptions {
                    placement: Placement {
                        translation: [0.0, 0.0, 1.0],
                        ..Placement::IDENTITY
                    },
                    ..MeshOptions::default()
                },
            )
            .unwrap();

        let report = scene.join(a, b, &options).unwrap();
        assert_eq!((report.vertices_added, report.triangles_added), (4, 2));
        assert_eq!(report.bridge_triangles_added, 8);
        // The copied vertices sit at z=1 in the target's local frame: the
        // world shape is kept.
        let mesh = scene.mesh(a).unwrap();
        let mut low = 0;
        let mut high = 0;
        for v in 0..mesh.vertex_slots() {
            let Some(p) = mesh.position(v as u32) else {
                continue;
            };
            if p[2] < 0.5 {
                low += 1;
            } else {
                assert!((p[2] - 1.0).abs() < 1e-6);
                high += 1;
            }
        }
        assert_eq!((low, high), (4, 4));
    }
}
