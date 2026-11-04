#![no_main]
//! Fuzzes [`fill_holes`](fast_mesh_ops::fill_holes) on arbitrary meshes: random
//! soup with ragged boundary loops plus grids with punched-out cells (clean
//! square holes). Options are capped so every input stays fast.

#[path = "common.rs"]
mod common;

use arbitrary::Arbitrary;
use common::{FuzzBuffers, FuzzGrid};
use fast_mesh_ops::{fill_holes, FillOptions};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FillInput {
    soup: FuzzBuffers,
    grid: FuzzGrid,
    use_grid: bool,
    fair: bool,
    allow_unclean: bool,
    dp_limit: u8,
    max_refine: u8,
    max_edges: Option<u8>,
}

fuzz_target!(|input: FillInput| {
    let mut mesh = if input.use_grid {
        input.grid.buffers.mesh()
    } else {
        input.soup.mesh()
    };
    let options = FillOptions {
        max_edges: input.max_edges.map(usize::from),
        allow_unclean: input.allow_unclean,
        fair: input.fair,
        // Below 3 every loop goes to the projection; the cap keeps the
        // `O(n³)` minimum-weight search fast.
        dp_limit: usize::from(input.dp_limit) % 25,
        max_refine_vertices: Some(usize::from(input.max_refine) % 65),
        ..FillOptions::default()
    };
    let mut tx = mesh.begin_edit();
    let _ = fill_holes(&mut tx, &options);
    let _ = tx.commit();
});
