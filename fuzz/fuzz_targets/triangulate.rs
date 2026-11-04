#![no_main]
//! Fuzzes [`triangulate`](fast_mesh_query::triangulate::triangulate) on
//! arbitrary polygons with up to two holes. The kernel promises to never panic
//! and to number every triangle with real points; the target asserts the
//! numbering.

#[path = "common.rs"]
mod common;

use arbitrary::{Arbitrary, Unstructured};
use fast_mesh_query::triangulate::triangulate;
use libfuzzer_sys::fuzz_target;

/// An outer ring of at most 48 points plus up to two holes of at most 24
/// points each, finite and quantized like the mesh coordinates.
#[derive(Debug)]
struct FuzzPolygon {
    outer: Vec<[f64; 2]>,
    holes: Vec<Vec<[f64; 2]>>,
}

fn point(u: &mut Unstructured<'_>) -> arbitrary::Result<[f64; 2]> {
    let coord = |u: &mut Unstructured<'_>| -> arbitrary::Result<f64> {
        Ok(f64::from(u16::arbitrary(u)? % 321) * 0.05 - 8.0)
    };
    Ok([coord(u)?, coord(u)?])
}

impl<'a> Arbitrary<'a> for FuzzPolygon {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let ring = |u: &mut Unstructured<'a>, max: u8| -> arbitrary::Result<Vec<[f64; 2]>> {
            let n = usize::from(u8::arbitrary(u)? % (max + 1));
            (0..n).map(|_| point(u)).collect()
        };
        let outer = ring(u, 48)?;
        let mut holes = Vec::new();
        for _ in 0..(u8::arbitrary(u)? % 3) {
            holes.push(ring(u, 24)?);
        }
        Ok(FuzzPolygon { outer, holes })
    }
}

fuzz_target!(|input: FuzzPolygon| {
    let hole_refs: Vec<&[[f64; 2]]> = input.holes.iter().map(Vec::as_slice).collect();
    let mut out = Vec::new();
    let _ = triangulate(&input.outer, &hole_refs, &mut out);
    let total = input.outer.len() + input.holes.iter().map(Vec::len).sum::<usize>();
    for tri in &out {
        for &v in tri {
            assert!(
                (v as usize) < total,
                "triangle numbers a point that does not exist"
            );
        }
    }
});
