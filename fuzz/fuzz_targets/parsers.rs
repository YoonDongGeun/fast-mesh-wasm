#![no_main]
//! Fuzzes the file parsers of the `fast-mesh` command line on arbitrary bytes
//! (the first byte picks the format). Whatever parses must satisfy the
//! documented result invariants and survive a write and read back.

use fast_mesh_cli::io::{parse, write_mesh, Format};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    let Some((&selector, bytes)) = input.split_first() else {
        return;
    };
    let format = match selector % 4 {
        0 => Format::Stl,
        1 => Format::Obj,
        2 => Format::Ply,
        _ => Format::Glb,
    };
    let Ok(data) = parse(format, bytes) else {
        return;
    };
    assert_eq!(data.positions.len() % 3, 0);
    assert_eq!(data.indices.len() % 3, 0);
    assert!(!data.indices.is_empty());
    assert!(data.positions.iter().all(|x| x.is_finite()));
    let vertices = data.positions.len() / 3;
    assert!(data.indices.iter().all(|&i| (i as usize) < vertices));

    // A coordinate may not fit a binary format's 32-bit floats: a refusal,
    // and OBJ, which is text, always takes it.
    let mut out = Vec::new();
    write_mesh(&data, Format::Obj, false, &mut out).expect("a parsed mesh is writable as OBJ");
    let back = parse(Format::Obj, &out).expect("written OBJ parses");
    assert_eq!(back.indices.len(), data.indices.len());
});
