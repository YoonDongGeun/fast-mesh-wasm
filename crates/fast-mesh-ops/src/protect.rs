//! Per-vertex protection around a position-moving operation.
//!
//! [`with_protection`] runs any operation in a transaction, then scales back
//! what it did to the listed vertices. It knows nothing of the operation's
//! insides, so brushes keep one code path whether or not anything is protected.

use fast_mesh_store::{Transaction, VertexId};

use crate::{OpsError, Result, reserve};

/// Runs `op` in `tx`, then scales each listed vertex's move by its weight:
/// `0` leaves the vertex at the bits it had before `op`, `1` keeps the whole
/// move, and values in between land on the straight line between the two.
///
/// Vertices that end where they started are not written. `op` must leave the
/// listed vertices alive. What `op` reports describes its own move, before
/// the scaling: a neighbor it computed against a protected vertex's moved
/// position keeps that result.
///
/// # Errors
/// - [`OpsError::InvalidInput`] when the lists differ in length, a weight is
///   not in `[0, 1]`, or a listed vertex is not alive; `op` has not run then.
/// - Whatever `op` returns, and [`OpsError::Store`] when memory runs out. The
///   caller's transaction should then be rolled back.
pub fn with_protection<R>(
    tx: &mut Transaction,
    vertices: &[VertexId],
    weights: &[f32],
    op: impl FnOnce(&mut Transaction) -> Result<R>,
) -> Result<R> {
    if vertices.len() != weights.len() {
        return Err(OpsError::InvalidInput(
            "protection needs one weight per vertex",
        ));
    }
    if weights.iter().any(|w| !(0.0..=1.0).contains(w)) {
        return Err(OpsError::InvalidInput(
            "protection weights must be between 0 and 1",
        ));
    }
    let mesh = tx.mesh();
    let positions = mesh.positions();
    let mut before: Vec<[f32; 3]> = Vec::new();
    reserve(&mut before, vertices.len())?;
    for &v in vertices {
        if !mesh.is_vertex_alive(v) {
            return Err(OpsError::InvalidInput("a protected vertex is not alive"));
        }
        before.push(positions[v as usize]);
    }
    // Reserved before `op` writes anything: running out of memory here leaves
    // the transaction untouched, like every operation.
    let mut ids: Vec<VertexId> = Vec::new();
    reserve(&mut ids, vertices.len())?;
    let mut back: Vec<[f32; 3]> = Vec::new();
    reserve(&mut back, vertices.len())?;
    let report = op(tx)?;
    let positions = tx.mesh().positions();
    for ((&v, &w), from) in vertices.iter().zip(weights).zip(&before) {
        let to = positions[v as usize];
        if w == 1.0 || same_bits(&to, from) {
            continue;
        }
        let at = if w == 0.0 {
            *from
        } else {
            [
                from[0] + w * (to[0] - from[0]),
                from[1] + w * (to[1] - from[1]),
                from[2] + w * (to[2] - from[2]),
            ]
        };
        ids.push(v);
        back.push(at);
    }
    if !ids.is_empty() {
        tx.set_positions(&ids, &back)?;
    }
    Ok(report)
}

fn same_bits(a: &[f32; 3], b: &[f32; 3]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deform::{self, Brush, Falloff, GrabOptions, InflateOptions, SmoothOptions};
    use crate::test_util::{mesh_of, sweep};
    use fast_mesh_store::Mesh;

    /// A flat grid of `n x n` quads in the xy plane, over [0, 1]².
    fn grid(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32 / n as f32, y as f32 / n as f32, 0.0]);
            }
        }
        let at = |x: u32, y: u32| y * (n + 1) + x;
        let mut triangles = Vec::new();
        for y in 0..n {
            for x in 0..n {
                triangles.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
                triangles.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        mesh_of(&positions, &triangles)
    }

    fn brush() -> Brush {
        Brush {
            center: [0.5, 0.5, 0.0],
            radius: 0.4,
            falloff: Falloff::Smooth,
            strength: 0.1,
        }
    }

    /// The left half of the grid: the vertices with `x < 0.5`.
    fn left_half(mesh: &Mesh) -> Vec<VertexId> {
        mesh.positions()
            .iter()
            .enumerate()
            .filter(|(_, p)| p[0] < 0.5)
            .map(|(v, _)| v as VertexId)
            .collect()
    }

    /// Runs `op` under a full hold on the left half; asserts the held
    /// vertices keep their bits and some others move.
    fn holds(op: impl FnOnce(&mut Transaction) -> Result<()>) {
        let mut mesh = grid(20);
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let before = mesh.positions().to_vec();
        let mut tx = mesh.begin_edit();
        with_protection(&mut tx, &held, &weights, op).unwrap();
        tx.commit().unwrap();
        for &v in &held {
            let (a, b) = (before[v as usize], mesh.positions()[v as usize]);
            assert!(same_bits(&a, &b), "held vertex {v} moved");
        }
        let moved = before
            .iter()
            .zip(mesh.positions())
            .filter(|(a, b)| !same_bits(a, b))
            .count();
        assert!(moved > 0, "nothing moved on the free side");
    }

    #[test]
    fn inflate_moves_only_free_vertices() {
        holds(|tx| {
            deform::inflate(tx, &InflateOptions { brush: brush() })?;
            Ok(())
        });
    }

    #[test]
    fn grab_moves_only_free_vertices() {
        holds(|tx| {
            let options = GrabOptions {
                brush: brush(),
                direction: [0.0, 0.0, 0.2],
            };
            deform::grab(tx, &options)?;
            Ok(())
        });
    }

    #[test]
    fn smooth_moves_only_free_vertices() {
        let mut mesh = grid(20);
        // A bump to melt: a flat grid would not move under smoothing.
        let mut tx = mesh.begin_edit();
        deform::grab(
            &mut tx,
            &GrabOptions {
                brush: brush(),
                direction: [0.0, 0.0, 0.3],
            },
        )
        .unwrap();
        tx.commit().unwrap();
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let before = mesh.positions().to_vec();
        let mut tx = mesh.begin_edit();
        with_protection(&mut tx, &held, &weights, |tx| {
            let options = SmoothOptions {
                brush: Brush {
                    strength: 0.5,
                    ..brush()
                },
                cycles: 2,
                ..SmoothOptions::default()
            };
            deform::smooth(tx, &options)
        })
        .unwrap();
        tx.commit().unwrap();
        for &v in &held {
            assert!(same_bits(
                &before[v as usize],
                &mesh.positions()[v as usize]
            ));
        }
        assert!(
            before
                .iter()
                .zip(mesh.positions())
                .any(|(a, b)| !same_bits(a, b))
        );
    }

    #[test]
    fn a_weight_scales_the_move() {
        let run = |weight: Option<f32>| {
            let mut mesh = grid(10);
            let v = 5 * 11 + 5; // The center vertex.
            let (ids, weights) = match weight {
                Some(w) => (vec![v as VertexId], vec![w]),
                None => (vec![], vec![]),
            };
            let mut tx = mesh.begin_edit();
            with_protection(&mut tx, &ids, &weights, |tx| {
                deform::inflate(tx, &InflateOptions { brush: brush() })
            })
            .unwrap();
            tx.commit().unwrap();
            mesh.positions()[v][2]
        };
        let whole = run(None);
        assert!(whole != 0.0);
        assert_eq!(run(Some(1.0)).to_bits(), whole.to_bits());
        assert!((run(Some(0.25)) - 0.25 * whole).abs() < 1e-6);
        assert_eq!(run(Some(0.0)).to_bits(), 0f32.to_bits());
    }

    #[test]
    fn undo_restores_every_vertex() {
        let mut mesh = grid(10);
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let before = mesh.positions().to_vec();
        let mut tx = mesh.begin_edit();
        with_protection(&mut tx, &held, &weights, |tx| {
            deform::inflate(tx, &InflateOptions { brush: brush() })
        })
        .unwrap();
        let patch = tx.commit().unwrap();
        assert!(mesh.positions() != before.as_slice());
        patch.revert(&mut mesh).unwrap();
        assert_eq!(mesh.positions(), before.as_slice());
    }

    #[test]
    fn running_out_of_memory_writes_nothing() {
        let mesh = grid(4);
        let held = left_half(&mesh);
        let weights = vec![0.0; held.len()];
        let failures = sweep(mesh, |tx| {
            with_protection(tx, &held, &weights, |tx| {
                deform::inflate(tx, &InflateOptions { brush: brush() })?;
                Ok(())
            })
        });
        assert!(failures >= 3);
    }

    #[test]
    fn rejects_bad_input_before_running() {
        let mut mesh = grid(2);
        let mut tx = mesh.begin_edit();
        let mut ran = false;
        let mut try_with = |ids: &[VertexId], weights: &[f32]| {
            with_protection(&mut tx, ids, weights, |_| {
                ran = true;
                Ok(())
            })
            .unwrap_err()
        };
        assert!(matches!(try_with(&[0], &[]), OpsError::InvalidInput(_)));
        assert!(matches!(try_with(&[0], &[1.5]), OpsError::InvalidInput(_)));
        assert!(matches!(
            try_with(&[0], &[f32::NAN]),
            OpsError::InvalidInput(_)
        ));
        assert!(matches!(
            try_with(&[9999], &[0.0]),
            OpsError::InvalidInput(_)
        ));
        assert!(!ran);
    }
}
