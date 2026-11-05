//! The parsers against hand-written files: what they must accept (the
//! behaviour of the TypeScript parsers they port) and what they must refuse.

use fast_mesh_cli::io::{Format, MeshData, parse, write_mesh};

fn triangles(data: &MeshData) -> Vec<[[f64; 3]; 3]> {
    data.indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            [0, 1, 2].map(|c| {
                let v = t[c] as usize;
                [
                    data.positions[3 * v],
                    data.positions[3 * v + 1],
                    data.positions[3 * v + 2],
                ]
            })
        })
        .collect()
}

fn err(format: Format, bytes: &[u8]) -> String {
    match parse(format, bytes) {
        Ok(d) => panic!("parsed: {} triangles", d.indices.len() / 3),
        Err(e) => e.to_string(),
    }
}

fn binary_stl(tris: &[[f32; 9]], head: &[u8]) -> Vec<u8> {
    let mut b = vec![0u8; 80];
    b[..head.len()].copy_from_slice(head);
    b.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in tris {
        b.extend_from_slice(&[0u8; 12]);
        for c in t {
            b.extend_from_slice(&c.to_le_bytes());
        }
        b.extend_from_slice(&[0u8; 2]);
    }
    b
}

const TRI: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

// ---- STL ----

#[test]
fn stl_binary_with_a_solid_header_stays_binary() {
    let d = parse(Format::Stl, &binary_stl(&[TRI], b"solid but binary")).unwrap();
    assert_eq!(d.indices, [0, 1, 2]);
    assert_eq!(d.positions.len(), 9);
}

#[test]
fn stl_merges_bit_identical_corners_and_signed_zero() {
    let second = [-0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let d = parse(Format::Stl, &binary_stl(&[TRI, second], b"")).unwrap();
    assert_eq!(d.positions.len() / 3, 4);
    assert_eq!(d.indices, [0, 1, 2, 0, 2, 3]);
}

#[test]
fn stl_ascii_tolerates_keyword_names() {
    let text = "solid vertex\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1e0 0 0\nvertex 0 +1.0 .0\nendloop\nendfacet\nendsolid vertex\n";
    let d = parse(Format::Stl, text.as_bytes()).unwrap();
    assert_eq!(
        triangles(&d),
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    );
}

#[test]
fn stl_refusals() {
    let tri = "facet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\n";
    assert!(err(Format::Stl, b"not a mesh").contains("not an STL"));
    assert!(err(Format::Stl, b"").contains("not an STL"));
    // Size does not match the count, and no "solid".
    let mut short = binary_stl(&[TRI], b"");
    short.pop();
    assert!(err(Format::Stl, &short).contains("bytes"));
    let mut long = binary_stl(&[TRI], b"");
    long.push(0);
    assert!(err(Format::Stl, &long).contains("bytes"));
    // A non-finite corner.
    let nan = [f32::NAN, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert!(err(Format::Stl, &binary_stl(&[nan], b"")).contains("finite"));
    // ASCII: no faces, bad numbers, not 3 per facet, cut off.
    assert!(err(Format::Stl, b"solid x\nendsolid x\n").contains("no faces"));
    for bad in ["nan", "inf", "1e999", "abc"] {
        let text = format!(
            "solid x\nfacet normal 0 0 1\nouter loop\nvertex {bad} 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n"
        );
        err(Format::Stl, text.as_bytes());
    }
    let two = "solid x\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nendloop\nendsolid x\n";
    assert!(err(Format::Stl, two.as_bytes()).contains("3 per facet"));
    let cut = format!("solid x\n{tri}");
    assert!(err(Format::Stl, cut.as_bytes()).contains("endsolid"));
    err(Format::Stl, b"solid x\nouter loop\nvertex 0 0");
}

// ---- OBJ ----

#[test]
fn obj_faces_and_corners() {
    let text = b"# comment \xff\no name\nv 0 0 0\nv 1 0 0 0.5 0.5 0.5\nv 1 1 0\nv\t0 1 0\nvn 0 0 1\nvt 0 0\nf 1/1/1 2//1 3/1 4\nf -4 -3 -2\ns off\n";
    let d = parse(Format::Obj, text).unwrap();
    assert_eq!(d.positions.len() / 3, 4);
    // The quad is a fan from its first corner.
    assert_eq!(d.indices, [0, 1, 2, 0, 2, 3, 0, 1, 2]);
}

#[test]
fn obj_allows_faces_before_their_vertices() {
    let d = parse(Format::Obj, b"f 1 2 3\nv 0 0 0\nv 1 0 0\nv 0 1 0\n").unwrap();
    assert_eq!(d.indices, [0, 1, 2]);
}

#[test]
fn obj_crlf() {
    let d = parse(Format::Obj, b"v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n").unwrap();
    assert_eq!(d.indices, [0, 1, 2]);
}

#[test]
fn obj_refusals() {
    assert!(err(Format::Obj, b"").contains("no faces"));
    assert!(err(Format::Obj, b"v 0 0 0\n").contains("no faces"));
    let line3 = err(Format::Obj, b"v 0 0 0\nv 1 0 0\nv 0 x 0\nf 1 2 3\n");
    assert!(line3.contains("line 3"), "{line3}");
    err(Format::Obj, b"v 0 0\nf 1 1 1\n");
    err(Format::Obj, b"v nan 0 0\nf 1 1 1\n");
    err(Format::Obj, b"v inf 0 0\nf 1 1 1\n");
    assert!(err(Format::Obj, b"v 0 0 0\nf 1 1\n").contains("3 corners"));
    assert!(err(Format::Obj, b"v 0 0 0\nf 0 1 1\n").contains("index"));
    assert!(err(Format::Obj, b"v 0 0 0\nf a 1 1\n").contains("index"));
    assert!(err(Format::Obj, b"v 0 0 0\nf 1 1 2\n").contains("past"));
    assert!(err(Format::Obj, b"v 0 0 0\nf -2 1 1\n").contains("before"));
    err(Format::Obj, b"v 0 0 0\nf 99999999999999 1 1\n");
    err(Format::Obj, b"v 0 0 0\nf 99999999999999999999999999 1 1\n");
}

// ---- PLY ----

#[test]
fn ply_ascii_with_extra_properties() {
    let text = "ply\nformat ascii 1.0\ncomment end_header not yet\nelement vertex 4\nproperty float nx\nproperty float x\nproperty uchar red\nproperty float y\nproperty float z\nproperty list uchar int tags\nelement face 2\nproperty uchar flag\nproperty list uchar uint vertex_indices\nelement extra 1\nproperty int q\nend_header\n9 0 1 0 0 0\n9 1 1 0 0 1 7\n9 1 1 1 0 0\n9 0 1 1 0 0\n5 4 0 1 2 3\n5 3 0 1 2\n42\n";
    let d = parse(Format::Ply, text.as_bytes()).unwrap();
    assert_eq!(
        d.positions,
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(d.indices, [0, 1, 2, 0, 2, 3, 0, 1, 2]);
}

#[test]
fn ply_vertex_index_alias_and_crlf_header() {
    let text = "ply\r\nformat ascii 1.0\r\nelement vertex 3\r\nproperty double x\r\nproperty double y\r\nproperty double z\r\nelement face 1\r\nproperty list uchar int vertex_index\r\nend_header\r\n0 0 0\r\n1 0 0\r\n0 1 0\r\n3 0 1 2\r\n";
    assert_eq!(
        parse(Format::Ply, text.as_bytes()).unwrap().indices,
        [0, 1, 2]
    );
}

fn binary_ply(index_type: &str, index: &[u8]) -> Vec<u8> {
    let head = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty double x\nproperty float y\nproperty short z\nelement face 1\nproperty list uchar {index_type} vertex_indices\nend_header\n"
    );
    let mut b = head.into_bytes();
    for (x, y, z) in [(0.5f64, 0.0f32, 0i16), (1.5, 0.0, 0), (0.5, 1.0, 7)] {
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b.extend_from_slice(&z.to_le_bytes());
    }
    b.push(3);
    b.extend_from_slice(index);
    b
}

#[test]
fn ply_binary_scalar_types() {
    let d = parse(Format::Ply, &binary_ply("uchar", &[0, 1, 2])).unwrap();
    assert_eq!(d.indices, [0, 1, 2]);
    assert_eq!(d.positions, [0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.5, 1.0, 7.0]);
    let wide: Vec<u8> = [0u32, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        parse(Format::Ply, &binary_ply("uint", &wide))
            .unwrap()
            .indices,
        [0, 1, 2]
    );
}

#[test]
fn ply_refusals() {
    let tail = "element vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n";
    let ascii = |body: &str| format!("ply\nformat ascii 1.0\n{tail}{body}").into_bytes();
    assert!(err(Format::Ply, b"").contains("not a PLY"));
    assert!(err(Format::Ply, b"plywood\n").contains("not a PLY"));
    assert!(err(Format::Ply, b"ply\nformat ascii 1.0\n").contains("end_header"));
    assert!(err(Format::Ply, format!("ply\n{tail}0 0 0\n").as_bytes()).contains("format"));
    let be = format!("ply\nformat binary_big_endian 1.0\n{tail}");
    assert!(err(Format::Ply, be.as_bytes()).contains("big-endian"));
    err(Format::Ply, b"ply\nformat weird 1.0\nend_header\n");
    assert!(err(Format::Ply, b"ply\nformat ascii 1.0\nend_header\n").contains("vertex"));
    err(
        Format::Ply,
        b"ply\nformat ascii 1.0\nproperty float x\nend_header\n",
    );
    err(
        Format::Ply,
        b"ply\nformat ascii 1.0\nelement vertex -1\nend_header\n",
    );
    err(
        Format::Ply,
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty wobble x\nend_header\n",
    );
    let no_list = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty int q\nend_header\n0 0 0 0\n";
    assert!(err(Format::Ply, no_list.as_bytes()).contains("index list"));
    // A float list count makes no sense.
    let float_count = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list float int vertex_indices\nend_header\n0 0 0\n3 0 0 0\n";
    err(Format::Ply, float_count.as_bytes());

    err(Format::Ply, &ascii("0 0 0\n1 0 0\n0 1 0\n"));
    assert!(err(Format::Ply, &ascii("0 0 0\n1 0 0\n0 1 0\n3 0 1")).contains("ends early"));
    assert!(err(Format::Ply, &ascii("0 0 0\n1 x 0\n0 1 0\n3 0 1 2\n")).contains("number"));
    err(Format::Ply, &ascii("0 0 0\n1 nan 0\n0 1 0\n3 0 1 2\n"));
    assert!(err(Format::Ply, &ascii("0 0 0\n1 0 0\n0 1 0\n3 0 1 3\n")).contains("out of range"));
    assert!(err(Format::Ply, &ascii("0 0 0\n1 0 0\n0 1 0\n3 0 1 -1\n")).contains("out of range"));
    assert!(err(Format::Ply, &ascii("0 0 0\n1 0 0\n0 1 0\n3 0 1 1.5\n")).contains("out of range"));
    err(Format::Ply, &ascii("0 0 0\n1 0 0\n0 1 0\n-3 0 1 2\n"));
    let mut truncated = binary_ply("uchar", &[0, 1, 2]);
    truncated.pop();
    assert!(err(Format::Ply, &truncated).contains("ends early"));
}

// ---- GLB ----

/// A GLB around `json` with `bin` as its body.
fn glb(json: &str, bin: &[u8]) -> Vec<u8> {
    let mut j = json.as_bytes().to_vec();
    while !j.len().is_multiple_of(4) {
        j.push(b' ');
    }
    let mut b = bin.to_vec();
    while !b.len().is_multiple_of(4) {
        b.push(0);
    }
    let total = 12 + 8 + j.len() + 8 + b.len();
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&j);
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&b);
    out
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Four vertices at (0,0,0) (1,0,0) (1,1,0) (0,1,0), `indices_json` appended
/// to the accessors, `primitive_extra` to the primitive; `nodes` is the full
/// nodes array.
fn square(indices_json: &str, index_bytes: &[u8], nodes: &str, primitive_extra: &str) -> Vec<u8> {
    let mut bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    bin.extend_from_slice(index_bytes);
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"nodes":{nodes},
        "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}{primitive_extra}}}]}}],
        "accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"}}{indices_json}],
        "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":{}}}],
        "buffers":[{{"byteLength":{}}}]}}"#,
        index_bytes.len(),
        bin.len()
    );
    glb(&json, &bin)
}

const NODE: &str = r#"[{"mesh":0}]"#;

#[test]
fn glb_index_widths() {
    let extra = r#","indices":1"#;
    for (ty, bytes) in [
        (5121, vec![0u8, 1, 2]),
        (
            5123,
            [0u16, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        (
            5125,
            [0u32, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ] {
        let acc = format!(r#",{{"bufferView":1,"componentType":{ty},"count":3,"type":"SCALAR"}}"#);
        let d = parse(Format::Glb, &square(&acc, &bytes, NODE, extra)).unwrap();
        assert_eq!(d.indices, [0, 1, 2], "{ty}");
    }
}

#[test]
fn glb_strip_fan_and_unindexed() {
    let acc = r#",{"bufferView":1,"componentType":5121,"count":4,"type":"SCALAR"}"#;
    let idx = [0u8, 1, 3, 2];
    let strip = square(acc, &idx, NODE, r#","indices":1,"mode":5"#);
    assert_eq!(
        parse(Format::Glb, &strip).unwrap().indices,
        [0, 1, 3, 1, 2, 3]
    );
    let idx = [0u8, 1, 2, 3];
    let fan = square(acc, &idx, NODE, r#","indices":1,"mode":6"#);
    assert_eq!(
        parse(Format::Glb, &fan).unwrap().indices,
        [0, 1, 2, 0, 2, 3]
    );
    // Four corners do not make whole triangles.
    let flat = square("", &[], NODE, "");
    assert!(err(Format::Glb, &flat).contains("multiple of 3"));
    let points = square("", &[], NODE, r#","mode":0"#);
    assert!(err(Format::Glb, &points).contains("mode"));
    let bad_index = square(
        r#",{"bufferView":1,"componentType":5121,"count":3,"type":"SCALAR"}"#,
        &[0, 1, 9],
        NODE,
        r#","indices":1"#,
    );
    assert!(err(Format::Glb, &bad_index).contains("past"));
}

const SCALARS: &str = r#",{"bufferView":1,"componentType":5121,"count":3,"type":"SCALAR"}"#;
const TRI_IDX: [u8; 3] = [0, 1, 2];
const IDX: &str = r#","indices":1"#;

#[test]
fn glb_bakes_the_node_hierarchy() {
    // Parent translates by (10,0,0); child scales by 2 and rotates a quarter
    // turn about z, so (1,0,0) -> (0,2,0) -> (10,2,0).
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let nodes = format!(
        r#"[{{"children":[1],"translation":[10,0,0]}},{{"mesh":0,"scale":[2,2,2],"rotation":[0,0,{h},{h}]}}]"#
    );
    let d = parse(Format::Glb, &square(SCALARS, &TRI_IDX, &nodes, IDX)).unwrap();
    let t = triangles(&d)[0];
    let close = |a: [f64; 3], b: [f64; 3]| {
        assert!((0..3).all(|k| (a[k] - b[k]).abs() < 1e-9), "{a:?} vs {b:?}");
    };
    close(t[0], [10.0, 0.0, 0.0]);
    close(t[1], [10.0, 2.0, 0.0]);
    close(t[2], [8.0, 2.0, 0.0]);

    let nodes = r#"[{"mesh":0,"matrix":[1,0,0,0, 0,1,0,0, 0,0,1,0, 5,6,7,1]}]"#;
    let d = parse(Format::Glb, &square(SCALARS, &TRI_IDX, nodes, IDX)).unwrap();
    assert_eq!(triangles(&d)[0][0], [5.0, 6.0, 7.0]);
}

#[test]
fn glb_mirroring_node_keeps_the_surface_outward() {
    // A negative determinant turns the winding: the triangle must come out
    // with its corners swapped, or volumes would change sign.
    let nodes = r#"[{"mesh":0,"scale":[-1,1,1]}]"#;
    let d = parse(Format::Glb, &square(SCALARS, &TRI_IDX, nodes, IDX)).unwrap();
    assert_eq!(d.indices, [0, 2, 1]);
    assert_eq!(triangles(&d)[0][2], [-1.0, 0.0, 0.0]);
}

#[test]
fn glb_merges_primitives_and_instances() {
    // No `scenes`: every node without a parent is a root.
    let nodes = r#"[{"mesh":0},{"mesh":0,"translation":[5,0,0]}]"#;
    let d = parse(Format::Glb, &square(SCALARS, &TRI_IDX, nodes, IDX)).unwrap();
    assert_eq!(d.positions.len() / 3, 8);
    assert_eq!(d.indices, [0, 1, 2, 4, 5, 6]);
    assert_eq!(triangles(&d)[1][0], [5.0, 0.0, 0.0]);
}

#[test]
fn glb_uses_the_default_scene() {
    let bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let json = r#"{"scene":1,"scenes":[{"nodes":[0]},{"nodes":[1]}],
      "nodes":[{"mesh":0},{"mesh":0,"translation":[5,0,0]}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],
      "bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36}]}"#;
    let d = parse(Format::Glb, &glb(json, &bin)).unwrap();
    assert_eq!(d.positions[..3], [5.0, 0.0, 0.0]);
    let missing = json.replace(r#""scene":1"#, r#""scene":7"#);
    assert!(err(Format::Glb, &glb(&missing, &bin)).contains("scene"));
}

#[test]
fn glb_stride() {
    // Interleaved: 12 bytes of position, 4 of something else.
    let mut bin = Vec::new();
    for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        bin.extend_from_slice(&f32s(&p));
        bin.extend_from_slice(&[0xAA; 4]);
    }
    let json = r#"{"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],
      "bufferViews":[{"buffer":0,"byteLength":48,"byteStride":16}],"buffers":[{"byteLength":48}]}"#;
    let d = parse(Format::Glb, &glb(json, &bin)).unwrap();
    assert_eq!(d.positions, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}

fn glb_with(accessor: &str, view: &str, primitive: &str, node: &str) -> Vec<u8> {
    let bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let json = format!(
        r#"{{"nodes":[{{"mesh":0{node}}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}{primitive}}}]}}],
        "accessors":[{{"componentType":5126,"type":"VEC3"{accessor}}}],
        "bufferViews":[{{"buffer":0{view}}}],"buffers":[{{"byteLength":36}}]}}"#
    );
    glb(&json, &bin)
}

const GOOD_ACC: &str = r#","bufferView":0,"count":3"#;
const GOOD_VIEW: &str = r#","byteLength":36"#;

#[test]
fn glb_baseline_for_the_refusals_parses() {
    assert_eq!(
        parse(Format::Glb, &glb_with(GOOD_ACC, GOOD_VIEW, "", ""))
            .unwrap()
            .indices,
        [0, 1, 2]
    );
}

#[test]
fn glb_refusals() {
    let e = |a: &str, v: &str, p: &str, n: &str| err(Format::Glb, &glb_with(a, v, p, n));
    let has = |message: String, needle: &str| {
        assert!(message.contains(needle), "{needle} in {message}");
    };
    // Compression and extensions that would change the geometry.
    has(
        e(
            GOOD_ACC,
            GOOD_VIEW,
            r#","extensions":{"KHR_draco_mesh_compression":{}}"#,
            "",
        ),
        "Draco",
    );
    has(
        e(
            GOOD_ACC,
            r#","byteLength":36,"extensions":{"EXT_meshopt_compression":{}}"#,
            "",
            "",
        ),
        "meshopt",
    );
    has(
        e(
            GOOD_ACC,
            r#","byteLength":36,"extensions":{"KHR_meshopt_compression":{}}"#,
            "",
            "",
        ),
        "meshopt",
    );
    has(
        e(
            GOOD_ACC,
            GOOD_VIEW,
            "",
            r#","extensions":{"EXT_mesh_gpu_instancing":{}}"#,
        ),
        "instancing",
    );
    // Accessors.
    has(e(r#","count":3"#, GOOD_VIEW, "", ""), "buffer view");
    has(e(r#","bufferView":0,"count":4"#, GOOD_VIEW, "", ""), "past");
    has(
        e(
            r#","bufferView":0,"count":3,"byteOffset":4"#,
            GOOD_VIEW,
            "",
            "",
        ),
        "past",
    );
    e(r#","bufferView":0,"count":-3"#, GOOD_VIEW, "", "");
    e(r#","bufferView":0,"count":1.5"#, GOOD_VIEW, "", "");
    has(
        e(
            r#","bufferView":0,"count":3,"sparse":{}"#,
            GOOD_VIEW,
            "",
            "",
        ),
        "sparse",
    );
    e(r#","bufferView":7,"count":3"#, GOOD_VIEW, "", "");
    // Views.
    has(e(GOOD_ACC, r#","byteLength":999"#, "", ""), "past");
    has(
        e(GOOD_ACC, r#","byteLength":36,"byteStride":4"#, "", ""),
        "stride",
    );
    has(e(GOOD_ACC, "", "", ""), "byteLength");
    // Nodes.
    has(e(GOOD_ACC, GOOD_VIEW, "", r#","matrix":[1,2,3]"#), "matrix");
    has(
        e(GOOD_ACC, GOOD_VIEW, "", r#","rotation":[0,0,0,0]"#),
        "rotation",
    );
    has(
        e(
            GOOD_ACC,
            GOOD_VIEW,
            "",
            r#","scale":[1e308,1e308,1e308],"translation":[1e308,0,0]"#,
        ),
        "finite",
    );
    has(e(GOOD_ACC, GOOD_VIEW, "", r#","children":[9]"#), "node");
    // The primitive.
    has(e(GOOD_ACC, GOOD_VIEW, r#","indices":5"#, ""), "accessor");
}

#[test]
fn glb_refuses_non_finite_positions_and_bad_position_types() {
    let bin = f32s(&[f32::NAN, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let json = r#"{"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],
      "bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36}]}"#;
    assert!(err(Format::Glb, &glb(json, &bin)).contains("finite"));
    err(Format::Glb, &glb(&json.replace("5126", "5123"), &bin));
    err(Format::Glb, &glb(&json.replace("VEC3", "VEC2"), &bin));
}

#[test]
fn glb_refuses_buffers_it_cannot_reach() {
    let bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    for buffers in [
        r#"[{"byteLength":36,"uri":"model.bin"}]"#,
        r#"[{"byteLength":36,"uri":"data:application/octet-stream;base64,AAAA"}]"#,
    ] {
        let json = format!(
            r#"{{"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}}],
            "bufferViews":[{{"buffer":0,"byteLength":36}}],"buffers":{buffers}}}"#
        );
        let message = err(Format::Glb, &glb(&json, &bin));
        assert!(message.contains("URI"), "{message}");
    }
}

#[test]
fn glb_refuses_cycles_and_shared_children() {
    let doc = |nodes: &str, scene: &str| {
        let json = format!(
            r#"{{"scenes":[{{"nodes":{scene}}}],"nodes":{nodes},"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}}],
            "bufferViews":[{{"buffer":0,"byteLength":36}}],"buffers":[{{"byteLength":36}}]}}"#
        );
        glb(&json, &f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    };
    for (nodes, scene) in [
        (r#"[{"children":[0],"mesh":0}]"#, "[0]"),
        (r#"[{"children":[1,1]},{"mesh":0}]"#, "[0]"),
        (r#"[{"mesh":0}]"#, "[0,0]"),
    ] {
        let message = err(Format::Glb, &doc(nodes, scene));
        assert!(message.contains("more than once"), "{message}");
    }
    assert!(err(Format::Glb, &doc(r#"[{"mesh":0}]"#, "[3]")).contains("node"));
}

#[test]
fn glb_container_refusals() {
    assert!(err(Format::Glb, b"").contains("not a GLB"));
    assert!(err(Format::Glb, b"{\"asset\":{}}").contains("not a GLB"));
    let good = glb_with(GOOD_ACC, GOOD_VIEW, "", "");
    let mut v1 = good.clone();
    v1[4] = 1;
    assert!(err(Format::Glb, &v1).contains("version"));
    assert!(err(Format::Glb, &good[..good.len() - 8]).contains("past"));
    let mut lying = good.clone();
    lying[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(err(Format::Glb, &lying).contains("past"));
    // No JSON chunk.
    let mut none = Vec::new();
    none.extend_from_slice(b"glTF");
    none.extend_from_slice(&2u32.to_le_bytes());
    none.extend_from_slice(&12u32.to_le_bytes());
    assert!(err(Format::Glb, &none).contains("JSON"));
    // No triangle meshes at all.
    assert!(err(Format::Glb, &glb(r#"{"asset":{}}"#, b"")).contains("no triangle meshes"));
    assert!(err(Format::Glb, &glb("[1,2]", b"")).contains("object"));
    assert!(err(Format::Glb, &glb("{", b"")).contains("JSON"));
}

#[test]
fn glb_refuses_json_nesting_too_deep_to_recurse() {
    let json = "[".repeat(2_000_000);
    assert!(err(Format::Glb, &glb(&json, b"")).contains("deep"));
}

#[test]
fn glb_long_node_chain_does_not_overflow_the_stack() {
    let n = 300_000;
    let mut nodes = String::from("[");
    for i in 0..n {
        nodes.push_str(&format!("{{\"children\":[{}]}},", i + 1));
    }
    nodes.push_str(r#"{"mesh":0}]"#);
    let json = format!(
        r#"{{"scenes":[{{"nodes":[0]}}],"nodes":{nodes},"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],
        "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}}],
        "bufferViews":[{{"buffer":0,"byteLength":36}}],"buffers":[{{"byteLength":36}}]}}"#
    );
    let bytes = glb(&json, &f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    assert_eq!(parse(Format::Glb, &bytes).unwrap().indices, [0, 1, 2]);
}

// ---- Writers ----

fn encoded(data: &MeshData, format: Format, ascii: bool) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    write_mesh(data, format, ascii, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

#[test]
fn writers_refuse_what_their_format_cannot_hold() {
    let huge = MeshData {
        positions: vec![1e300, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        indices: vec![0, 1, 2],
    };
    for f in [Format::Stl, Format::Ply, Format::Glb] {
        let message = encoded(&huge, f, false).unwrap_err();
        assert!(message.contains("32-bit"), "{message}");
    }
    // The text formats carry it.
    let text = encoded(&huge, Format::Obj, false).unwrap();
    assert_eq!(parse(Format::Obj, &text).unwrap().positions[0], 1e300);

    let empty = MeshData {
        positions: vec![0.0; 9],
        indices: vec![],
    };
    for f in [Format::Stl, Format::Obj, Format::Ply, Format::Glb] {
        let message = encoded(&empty, f, false).unwrap_err();
        assert!(message.contains("no triangles"), "{message}");
    }
    let bad = MeshData {
        positions: vec![0.0; 3],
        indices: vec![0, 0, 7],
    };
    assert!(encoded(&bad, Format::Obj, false).is_err());
}

#[test]
fn glb_output_is_well_formed() {
    let tri = MeshData {
        positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        indices: vec![0, 1, 2],
    };
    let bytes = encoded(&tri, Format::Glb, false).unwrap();
    assert_eq!(&bytes[..4], b"glTF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        bytes.len()
    );
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert!(json_len.is_multiple_of(4));
    assert_eq!(&bytes[16..20], b"JSON");
    let bin_at = 20 + json_len;
    assert_eq!(
        u32::from_le_bytes(bytes[bin_at..bin_at + 4].try_into().unwrap()),
        36 + 12
    );
    assert_eq!(&bytes[bin_at + 4..bin_at + 8], b"BIN\0");
    let json = std::str::from_utf8(&bytes[20..bin_at]).unwrap();
    for needle in [
        r#""version":"2.0""#,
        r#""min":[0,0,0]"#,
        r#""max":[1,1,0]"#,
        r#""componentType":5125"#,
    ] {
        assert!(json.contains(needle), "{needle} in {json}");
    }
}

// ---- Robustness: every prefix and a sweep of byte flips must return. ----

fn samples() -> Vec<(Format, Vec<u8>)> {
    let data = MeshData {
        positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        indices: vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3],
    };
    let mut out = Vec::new();
    for (f, ascii) in [
        (Format::Stl, false),
        (Format::Stl, true),
        (Format::Obj, false),
        (Format::Ply, false),
        (Format::Ply, true),
        (Format::Glb, false),
    ] {
        out.push((f, encoded(&data, f, ascii).unwrap()));
    }
    out
}

fn check_invariants(data: &MeshData) {
    assert_eq!(data.positions.len() % 3, 0);
    assert_eq!(data.indices.len() % 3, 0);
    assert!(!data.indices.is_empty());
    assert!(data.positions.iter().all(|x| x.is_finite()));
    let vertices = data.positions.len() / 3;
    assert!(data.indices.iter().all(|&i| (i as usize) < vertices));
}

#[test]
fn every_truncation_returns_without_panicking() {
    for (format, bytes) in samples() {
        for cut in 0..bytes.len() {
            if let Ok(d) = parse(format, &bytes[..cut]) {
                check_invariants(&d);
            }
        }
    }
}

/// Formats with explicit counts or terminators notice any cut (OBJ alone has
/// neither, so a cut at a line end is a smaller valid file).
#[test]
fn truncated_files_are_refused() {
    for (format, bytes) in samples() {
        if format == Format::Obj {
            continue;
        }
        // Whitespace at the very end of a text file carries no data, and
        // neither does the free text after `endsolid`.
        let keep = match bytes.windows(8).position(|w| w == b"endsolid") {
            Some(at) => at + 8,
            None => bytes.trim_ascii_end().len(),
        };
        for cut in 0..keep {
            assert!(
                parse(format, &bytes[..cut]).is_err(),
                "{format:?} cut at {cut} of {}",
                bytes.len()
            );
        }
    }
}

#[test]
fn byte_flips_return_without_panicking() {
    let mut state = 0x9E37_79B9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for (format, bytes) in samples() {
        for _ in 0..3_000 {
            let mut m = bytes.clone();
            for _ in 0..(1 + next() % 4) {
                let at = next() as usize % m.len();
                m[at] = match next() % 4 {
                    0 => 0,
                    1 => 0xFF,
                    2 => b'9',
                    _ => next() as u8,
                };
            }
            if let Ok(d) = parse(format, &m) {
                check_invariants(&d);
            }
        }
    }
}
