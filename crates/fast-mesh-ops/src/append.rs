//! Appending copied geometry to a mesh.
//!
//! [`append_mesh`] copies a position/triangle soup into a transaction and
//! reports the new ids. Positions arrive in the target mesh's local frame;
//! any world-frame transform is the caller's job. Only positions and
//! triangles cross over: source channels take defaults, derived channels
//! (like normals) are recomputed by the scene. Triangle indices are bounds
//! checked; anything else about the soup is the caller's responsibility.

use fast_mesh_store::{Transaction, VertexId};

use crate::{OpsError, Result, reserve};

/// Geometry to append, indexed from zero.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AppendSource {
    /// Vertex positions in the target mesh's local frame.
    pub positions: Vec<[f32; 3]>,
    /// Triangles indexed into [`AppendSource::positions`].
    pub triangles: Vec<[u32; 3]>,
}

/// Outcome of [`append_mesh`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppendReport {
    /// New vertex id per source position, in source order.
    pub vertex_map: Vec<VertexId>,
    /// Vertices added.
    pub vertices_added: usize,
    /// Triangles added.
    pub triangles_added: usize,
}

/// Copies `source` into `tx`, returning the new vertex ids in source order.
///
/// Triangle indices past the positions are rejected before anything is
/// written. Either every vertex and triangle lands or nothing does.
pub fn append_mesh(tx: &mut Transaction, source: &AppendSource) -> Result<AppendReport> {
    for t in &source.triangles {
        if t.iter().any(|&v| (v as usize) >= source.positions.len()) {
            return Err(OpsError::InvalidInput(
                "append triangle indexes no position",
            ));
        }
    }
    // Reserved before the first write; past it only infallible pushes into
    // reserved capacity remain, so any allocation failure leaves an empty
    // patch.
    let mut mapped: Vec<[VertexId; 3]> = Vec::new();
    reserve(&mut mapped, source.triangles.len())?;
    let ids = tx.add_vertices(&source.positions)?;
    for t in &source.triangles {
        mapped.push([ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]]);
    }
    tx.add_triangles(&mapped)?;
    Ok(AppendReport {
        vertices_added: ids.len(),
        triangles_added: mapped.len(),
        vertex_map: ids,
    })
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::Transaction;

    use super::*;
    use crate::test_util::{edit, mesh_of, parts, sweep};

    fn quad_source() -> AppendSource {
        AppendSource {
            positions: vec![
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        }
    }

    #[test]
    fn append_copies_geometry_with_id_map() {
        let mut mesh = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[[0, 1, 2]]);
        let (before_pos, before_tris) = parts(&mesh);
        let (report, _) = edit(&mut mesh, |tx| append_mesh(tx, &quad_source()));
        assert_eq!(report.vertices_added, 4);
        assert_eq!(report.triangles_added, 2);
        assert_eq!(report.vertex_map.len(), 4);
        let (pos_after, tris_after) = parts(&mesh);
        assert_eq!(tris_after.len(), before_tris.len() + 2);
        // Originals untouched; appended vertices carry exact positions.
        assert_eq!(&pos_after[..before_pos.len()], &before_pos[..]);
        for (k, &id) in report.vertex_map.iter().enumerate() {
            let p = &pos_after[3 * id as usize..3 * id as usize + 3];
            assert_eq!(p, &quad_source().positions[k][..]);
        }
        for t in &tris_after[before_tris.len()..] {
            assert!(t.iter().all(|v| report.vertex_map.contains(v)));
        }
    }

    #[test]
    fn append_empty_source_is_noop() {
        let mut mesh = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[[0, 1, 2]]);
        let before = parts(&mesh);
        let empty = AppendSource {
            positions: Vec::new(),
            triangles: Vec::new(),
        };
        let (report, patch_empty) = edit(&mut mesh, |tx| append_mesh(tx, &empty));
        assert_eq!((report.vertices_added, report.triangles_added), (0, 0));
        assert!(report.vertex_map.is_empty());
        assert!(patch_empty);
        assert_eq!(parts(&mesh), before);
    }

    #[test]
    fn append_sweep_keeps_atomicity() {
        let mesh = mesh_of(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[[0, 1, 2]]);
        let source = quad_source();
        let n = sweep(mesh, |tx: &mut Transaction| {
            append_mesh(tx, &source)?;
            Ok(())
        });
        assert!(n > 0);
    }
}
