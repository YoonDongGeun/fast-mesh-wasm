#![no_main]
//! Fuzzes [`Bvh::build`](fast_mesh_bvh::Bvh::build) plus
//! [`query_aabb`](fast_mesh_bvh::Bvh::query_aabb) on arbitrary soup, built
//! through either buffer constructor (the `f32` and `f64` parsers). Every
//! reported triangle must be a live slot of the mesh.

#[path = "common.rs"]
mod common;

use arbitrary::{Arbitrary, Unstructured};
use common::FuzzBuffers;
use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
struct BvhInput {
    mesh: FuzzBuffers,
    via_f64: bool,
    fast: bool,
    max_leaf_size: u8,
    queries: Vec<[[f32; 3]; 2]>,
}

impl<'a> Arbitrary<'a> for BvhInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let mesh = FuzzBuffers::arbitrary(u)?;
        let (via_f64, fast, max_leaf_size) =
            (bool::arbitrary(u)?, bool::arbitrary(u)?, u8::arbitrary(u)?);
        let corner = |u: &mut Unstructured<'a>| -> arbitrary::Result<[f32; 3]> {
            Ok([
                f32::from(u16::arbitrary(u)? % 321) * 0.05 - 8.0,
                f32::from(u16::arbitrary(u)? % 321) * 0.05 - 8.0,
                f32::from(u16::arbitrary(u)? % 321) * 0.05 - 8.0,
            ])
        };
        let mut queries = Vec::new();
        for _ in 0..(u8::arbitrary(u)? % 9) {
            queries.push([corner(u)?, corner(u)?]);
        }
        Ok(BvhInput {
            mesh,
            via_f64,
            fast,
            max_leaf_size,
            queries,
        })
    }
}

fuzz_target!(|input: BvhInput| {
    let mesh = if input.via_f64 {
        input.mesh.mesh_f64()
    } else {
        input.mesh.mesh()
    };
    let options = BuildOptions {
        quality: if input.fast {
            Quality::Fast
        } else {
            Quality::Sah
        },
        max_leaf_size: u32::from(input.max_leaf_size % 16 + 1),
        ..BuildOptions::default()
    };
    let Ok(bvh) = Bvh::build(&mesh, &options) else {
        return;
    };
    let mut out = Vec::new();
    for [a, b] in &input.queries {
        // Both orders: the query must handle an inverted box without panicking.
        for (min, max) in [(a, b), (b, a)] {
            out.clear();
            bvh.query_aabb(&mesh, *min, *max, &mut out);
            for &t in &out {
                assert!(t < mesh.triangle_slots() as u32, "query hit a dead slot");
                assert!(mesh.is_triangle_alive(t), "query hit a dead triangle");
            }
        }
    }
});
