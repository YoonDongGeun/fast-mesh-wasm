#![no_main]
//! Fuzzes [`weld`](fast_mesh_ops::weld) on arbitrary soup (quantized
//! coordinates, so coincident vertices occur) at small tolerances.

#[path = "common.rs"]
mod common;

use arbitrary::Arbitrary;
use common::FuzzBuffers;
use fast_mesh_ops::weld;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct WeldInput {
    mesh: FuzzBuffers,
    tolerance: u8,
}

fuzz_target!(|input: WeldInput| {
    let mut mesh = input.mesh.mesh();
    // 0.0 to 2.55 in steps of 0.01.
    let tolerance = f32::from(input.tolerance) * 0.01;
    let mut tx = mesh.begin_edit();
    let _ = weld(&mut tx, tolerance);
    let _ = tx.commit();
});
