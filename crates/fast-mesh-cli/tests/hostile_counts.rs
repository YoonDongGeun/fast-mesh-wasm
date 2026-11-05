//! A header claiming billions of elements must fail before any allocation of
//! that size. A counting allocator is the evidence: it records the largest
//! single request while the parser runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fast_mesh_cli::io::{Format, parse};

struct Tracking;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System`; the counter has no effect on it.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Tracking = Tracking;

static SERIAL: Mutex<()> = Mutex::new(());

const LIMIT: usize = 1 << 20;

/// Parses `bytes`, asserting a refusal and no single allocation over 1 MiB.
fn refused_cheaply(format: Format, bytes: &[u8]) -> String {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    LARGEST.store(0, Ordering::Relaxed);
    let result = parse(format, bytes);
    let largest = LARGEST.load(Ordering::Relaxed);
    let message = match result {
        Ok(_) => panic!("parsed a hostile file"),
        Err(e) => e.to_string(),
    };
    assert!(largest < LIMIT, "{largest} byte allocation: {message}");
    message
}

#[test]
fn the_tracker_sees_large_allocations() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    LARGEST.store(0, Ordering::Relaxed);
    let big = std::hint::black_box(vec![1u8; 2 * LIMIT]);
    assert!(LARGEST.load(Ordering::Relaxed) >= 2 * LIMIT);
    drop(big);
}

#[test]
fn stl_triangle_count() {
    let mut b = vec![0u8; 84];
    for count in [u32::MAX, 3_000_000_000, 1 << 26] {
        b[80..84].copy_from_slice(&count.to_le_bytes());
        assert!(refused_cheaply(Format::Stl, &b).contains("bytes"));
    }
}

fn ply(format: &str, vertices: u64, faces: u64) -> Vec<u8> {
    format!(
        "ply\nformat {format} 1.0\nelement vertex {vertices}\nproperty float x\nproperty float y\nproperty float z\nelement face {faces}\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n"
    )
    .into_bytes()
}

#[test]
fn ply_element_counts() {
    for format in ["ascii", "binary_little_endian"] {
        for (v, f) in [
            (4_000_000_000, 1),
            (3, 4_000_000_000),
            (u32::MAX as u64 + 1, 1),
            (1 << 40, 1 << 40),
            (u64::MAX, 1),
            (u64::MAX / 2, u64::MAX / 2),
        ] {
            let message = refused_cheaply(Format::Ply, &ply(format, v, f));
            assert!(message.contains("PLY"), "{message}");
        }
    }
}

#[test]
fn ply_element_count_over_u64_is_a_bad_line() {
    let text = "ply\nformat ascii 1.0\nelement vertex 99999999999999999999999\nproperty float x\nend_header\n";
    refused_cheaply(Format::Ply, text.as_bytes());
}

#[test]
fn ply_propertyless_element_cannot_spin() {
    // Billions of zero-byte elements must not loop: this returns at once.
    let text = "ply\nformat binary_little_endian 1.0\nelement nothing 18446744073709551615\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n";
    refused_cheaply(Format::Ply, text.as_bytes());
}

#[test]
fn ply_list_count() {
    // The list claims 4 billion entries in a small file.
    let mut b = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uint int vertex_indices\nend_header\n".to_vec();
    b.extend_from_slice(&4_000_000_000u32.to_le_bytes());
    b.extend_from_slice(&[0; 16]);
    assert!(refused_cheaply(Format::Ply, &b).contains("PLY"));
}

fn glb(json: &str) -> Vec<u8> {
    let mut j = json.as_bytes().to_vec();
    while !j.len().is_multiple_of(4) {
        j.push(b' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&((20 + j.len() + 8 + 36) as u32).to_le_bytes());
    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&j);
    out.extend_from_slice(&36u32.to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&[0u8; 36]);
    out
}

#[test]
fn glb_accessor_counts() {
    for count in ["4000000000", "4294967295", "9007199254740991", "1e300"] {
        let json = format!(
            r#"{{"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":{count},"type":"VEC3"}}],
            "bufferViews":[{{"buffer":0,"byteLength":36}}],"buffers":[{{"byteLength":36}}]}}"#
        );
        refused_cheaply(Format::Glb, &glb(&json));
    }
}

#[test]
fn glb_index_accessor_count() {
    let json = r#"{"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
                   {"bufferView":0,"componentType":5125,"count":4000000000,"type":"SCALAR"}],
      "bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36}]}"#;
    refused_cheaply(Format::Glb, &glb(json));
}

#[test]
fn glb_accessor_without_a_view_cannot_conjure_zeros() {
    let json = r#"{"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
      "accessors":[{"componentType":5126,"count":4000000000,"type":"VEC3"}]}"#;
    refused_cheaply(Format::Glb, &glb(json));
}

#[test]
fn glb_chunk_length() {
    let mut b = glb(r#"{"asset":{}}"#);
    b[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    refused_cheaply(Format::Glb, &b);
}

/// Nothing is wrong with any one count here: two thousand nodes reuse a mesh of
/// a thousand primitives over the same vertices, 2 billion once expanded.
#[test]
fn glb_instancing_bomb() {
    let mut nodes = String::from("[");
    for _ in 0..2_000 {
        nodes.push_str(r#"{"mesh":0},"#);
    }
    nodes.push_str(r#"{"mesh":0}]"#);
    let mut primitives = String::from("[");
    for _ in 0..1000 {
        primitives.push_str(r#"{"attributes":{"POSITION":0}},"#);
    }
    primitives.push_str(r#"{"attributes":{"POSITION":0}}]"#);
    let json = format!(
        r#"{{"nodes":{nodes},"meshes":[{{"primitives":{primitives}}}],
        "accessors":[{{"bufferView":0,"componentType":5126,"count":999,"type":"VEC3"}}],
        "bufferViews":[{{"buffer":0,"byteLength":11988}}],"buffers":[{{"byteLength":11988}}]}}"#
    );
    let mut j = json.into_bytes();
    while !j.len().is_multiple_of(4) {
        j.push(b' ');
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"glTF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&((20 + j.len() + 8 + 11988) as u32).to_le_bytes());
    bytes.extend_from_slice(&(j.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"JSON");
    bytes.extend_from_slice(&j);
    bytes.extend_from_slice(&11988u32.to_le_bytes());
    bytes.extend_from_slice(b"BIN\0");
    bytes.extend_from_slice(&[0u8; 11988]);
    assert!(refused_cheaply(Format::Glb, &bytes).contains("too many"));
}
