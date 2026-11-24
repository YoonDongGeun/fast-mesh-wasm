//! Tests that need the crate's test hooks; the properties of the result are
//! in `tests/lattice.rs`.

use fast_mesh_store::Mesh;

use super::*;
use crate::test_util::{
    assert_parallel_identical, box_mesh, combine, edit, flip, mesh_of, parts, sweep, uv_sphere,
};

/// A ball with a concentric cavity.
fn hollow_ball() -> Mesh {
    let inner = uv_sphere(24, 32, 0.7);
    let all: Vec<u32> = (0..inner.triangle_slots() as u32).collect();
    combine(&[&uv_sphere(24, 32, 1.0), &flip(&inner, &all)])
}

fn options() -> LatticeOptionsBuilder {
    LatticeOptions::builder().cell_size(0.6).wall_thickness(0.1)
}

/// The vertices and triangles as bits, and the report.
fn run(mut mesh: Mesh, options: &LatticeOptions) -> (Vec<u32>, Vec<[u32; 3]>, LatticeReport) {
    let (report, _) = edit(&mut mesh, |tx| lattice_fill(tx, options));
    let (positions, triangles) = parts(&mesh);
    let bits = positions.iter().map(|p| p.to_bits()).collect();
    (bits, triangles, report)
}

#[test]
fn lattices_are_identical_with_and_without_parallel() {
    let shelled = options().shell_thickness(0.1);
    for (what, mesh, opts) in [
        ("open", uv_sphere(24, 32, 1.0), options()),
        ("shell", box_mesh([-1.0; 3], [1.0; 3]), shelled),
        ("cavity", hollow_ball(), options()),
    ] {
        let opts = opts.build();
        let (positions, triangles) = parts(&mesh);
        let copy = || mesh_of(&positions, &triangles);
        let (bits, _, report) = assert_parallel_identical(what, || run(copy(), &opts));
        assert!(bits.len() > 3000 && report.triangles > 1000, "{what}");
        let preview =
            assert_parallel_identical(what, || lattice_preview(&copy(), &opts, 40_000).unwrap());
        assert!(!preview.walls.triangles.is_empty(), "{what}");
    }
}

#[test]
fn running_out_of_memory_writes_nothing() {
    let coarse = LatticeOptions::builder()
        .cell_size(1.0)
        .wall_thickness(0.25)
        .voxel_size(Some(0.125));
    for (mesh, opts) in [
        (box_mesh([0.0; 3], [1.0; 3]), coarse.clone()),
        (
            box_mesh([0.0; 3], [1.5; 3]),
            coarse.clone().shell_thickness(0.25),
        ),
        (hollow_ball(), coarse.clone()),
    ] {
        let opts = opts.build();
        let points = sweep(mesh, |tx| lattice_fill(tx, &opts).map(|_| ()));
        assert!(points > 30, "{points} reservations");
    }
    // A preview reports it too.
    let mesh = box_mesh([0.0; 3], [1.0; 3]);
    let opts = coarse.build();
    let expected = lattice_preview(&mesh, &opts, 1000).unwrap();
    let mut failed = 0;
    loop {
        crate::fail_allocations_after(Some(failed));
        let got = lattice_preview(&mesh, &opts, 1000);
        crate::fail_allocations_after(None);
        match got {
            Err(e) => assert_eq!(e, OpsError::Store(StoreError::OutOfMemory)),
            Ok(got) => {
                assert_eq!(got, expected);
                break;
            }
        }
        failed += 1;
    }
    assert!(failed > 20, "{failed} reservations");
}
