//! End-to-end tests: they execute the built `fast-mesh` binary on files
//! written here by hand, so expected values never come from the code under
//! test (a unit cube has area 6 and volume 1, and so on).

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CUBE_V: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];
/// Outward-wound.
const CUBE_T: [[usize; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [1, 2, 6],
    [1, 6, 5],
    [2, 3, 7],
    [2, 7, 6],
    [3, 0, 4],
    [3, 4, 7],
];

type Soup = Vec<[[f64; 3]; 3]>;

/// A cube of side `size` with its low corner at `at`, as a triangle soup.
fn cube(at: [f64; 3], size: f64) -> Soup {
    CUBE_T
        .iter()
        .map(|t| t.map(|v| [0, 1, 2].map(|k| at[k] + size * CUBE_V[v][k])))
        .collect()
}

fn unit_cube() -> Soup {
    cube([0.0; 3], 1.0)
}

fn ascii_stl(soup: &Soup) -> Vec<u8> {
    let mut s = String::from("solid test\n");
    for t in soup {
        s.push_str(" facet normal 0 0 0\n  outer loop\n");
        for v in t {
            s.push_str(&format!("   vertex {} {} {}\n", v[0], v[1], v[2]));
        }
        s.push_str("  endloop\n endfacet\n");
    }
    s.push_str("endsolid test\n");
    s.into_bytes()
}

fn binary_stl(soup: &Soup) -> Vec<u8> {
    let mut b = vec![0u8; 80];
    b.extend_from_slice(&(soup.len() as u32).to_le_bytes());
    for t in soup {
        b.extend_from_slice(&[0u8; 12]);
        for v in t {
            for c in v {
                b.extend_from_slice(&(*c as f32).to_le_bytes());
            }
        }
        b.extend_from_slice(&[0u8; 2]);
    }
    b
}

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cli-{name}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Dir(path)
    }
    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    /// Names in the directory, sorted: catches a leftover temporary file.
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

fn run<S: AsRef<OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fast-mesh"))
        .args(args)
        .output()
        .unwrap()
}

/// Builds an argument list out of paths and strings alike.
macro_rules! args {
    ($($a:expr),* $(,)?) => { [$(OsString::from($a)),*] };
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exited with a code")
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).unwrap()
}

/// The value of `"key":` in the (flat enough) JSON report; a test-side reader
/// so that the expected values do not go through the code under test.
fn field<'a>(json: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":");
    let at = json
        .find(&needle)
        .unwrap_or_else(|| panic!("no {key} in {json}"))
        + needle.len();
    let rest = json[at..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim()
}

fn number(json: &str, key: &str) -> f64 {
    field(json, key).parse().unwrap()
}

fn measure_json(path: &Path) -> String {
    let o = run(&args!["measure", path, "--format", "json"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    stdout(&o)
}

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-5, "{a} vs {b}");
}

#[test]
fn measure_unit_cube_json() {
    let dir = Dir::new("measure");
    for (name, bytes) in [
        ("ascii.stl", ascii_stl(&unit_cube())),
        ("binary.stl", binary_stl(&unit_cube())),
    ] {
        let path = dir.file(name, &bytes);
        let o = run(&args!["measure", &path, "--format", "json"]);
        assert_eq!(code(&o), 0);
        assert!(o.stderr.is_empty(), "{}", stderr(&o));
        let json = stdout(&o);
        assert!(json.trim_start().starts_with('{') && json.trim_end().ends_with('}'));
        assert_eq!(number(&json, "vertices"), 8.0);
        assert_eq!(number(&json, "triangles"), 12.0);
        assert_close(number(&json, "area"), 6.0);
        assert_close(number(&json, "volume"), 1.0);
        assert_close(number(&json, "signed_volume"), 1.0);
        assert_eq!(field(&json, "closed"), "true");
        assert_eq!(field(&json, "volume_reason"), "null");
        assert_eq!(number(&json, "boundary_loops"), 0.0);
    }
}

#[test]
fn measure_open_mesh_has_no_volume() {
    let dir = Dir::new("measure-open");
    let mut soup = unit_cube();
    soup.pop();
    let json = measure_json(&dir.file("open.stl", &ascii_stl(&soup)));
    assert_close(number(&json, "area"), 5.5);
    assert_eq!(field(&json, "closed"), "false");
    assert_eq!(field(&json, "volume"), "null");
    assert_eq!(field(&json, "volume_reason"), "\"open\"");
    assert_eq!(number(&json, "boundary_loops"), 1.0);
}

#[test]
fn measure_text_output_names_the_values() {
    let dir = Dir::new("measure-text");
    let path = dir.file("c.stl", &ascii_stl(&unit_cube()));
    let o = run(&args!["measure", &path]);
    assert_eq!(code(&o), 0);
    let text = stdout(&o);
    assert!(text.contains("area: 6"), "{text}");
    assert!(text.contains("closed: true"), "{text}");
}

#[test]
fn repair_fills_a_known_hole() {
    let dir = Dir::new("repair");
    let mut soup = unit_cube();
    soup.pop();
    let input = dir.file("in.stl", &binary_stl(&soup));
    let output = dir.path("out.stl");
    let o = run(&args![
        "repair", &input, "-o", &output, "--auto", "--format", "json"
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let json = stdout(&o);
    assert_eq!(number(&json, "filled"), 1.0);
    assert_eq!(number(&json, "failed"), 0.0);
    let after = measure_json(&output);
    assert_eq!(field(&after, "closed"), "true");
    assert_close(number(&after, "volume"), 1.0);
    assert_close(number(&after, "area"), 6.0);
}

#[test]
fn repair_leaves_a_sound_mesh_unchanged() {
    let dir = Dir::new("repair-sound");
    // Coordinates not round in binary, away from zero: the mesh's own origin
    // must not disturb a single output float.
    let input = dir.file("in.stl", &binary_stl(&cube([100.1, -20.2, 0.3], 0.7)));
    let output = dir.path("out.stl");
    let o = run(&args!["repair", &input, "-o", &output, "--auto"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(
        triangles_of_binary_stl(&fs::read(&output).unwrap()),
        triangles_of_binary_stl(&fs::read(&input).unwrap())
    );
}

#[test]
fn repair_options_reach_the_pipeline() {
    let dir = Dir::new("repair-options");
    // A cube with one corner of one triangle off by 2e-3, plus a separate
    // tetrahedron of 4 triangles (smaller than the cube's 12).
    let mut soup = unit_cube();
    let mut jittered = false;
    for t in soup.iter_mut() {
        for v in t.iter_mut() {
            if *v == [1.0, 1.0, 1.0] && !jittered {
                v[0] += 2e-3;
                jittered = true;
            }
        }
    }
    let at = 10.0;
    let (a, b, c, d) = (
        [at, 0.0, 0.0],
        [at + 1.0, 0.0, 0.0],
        [at, 1.0, 0.0],
        [at, 0.0, 1.0],
    );
    soup.extend([[a, c, b], [a, b, d], [b, c, d], [c, a, d]]);
    let input = dir.file("in.stl", &ascii_stl(&soup));
    let output = dir.path("out.obj");
    let repair = |extra: &[&str]| {
        let mut list = args![
            "repair", &input, "-o", &output, "--auto", "--format", "json"
        ]
        .to_vec();
        list.extend(extra.iter().map(OsString::from));
        run(&list)
    };

    // Too small a weld tolerance and no fill: the seam stays open.
    let o = repair(&[
        "--weld-tolerance",
        "1e-5",
        "--min-triangles",
        "1",
        "--max-edges",
        "0",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let json = stdout(&o);
    assert_eq!(number(&json, "merged_vertices"), 0.0);
    assert_eq!(number(&json, "removed_components"), 0.0);
    assert_eq!(field(&measure_json(&output), "closed"), "false");

    // A wide enough tolerance welds it; a bound of 5 removes the tetrahedron.
    let o = repair(&[
        "--weld-tolerance",
        "0.01",
        "--min-triangles",
        "5",
        "--no-fair",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let json = stdout(&o);
    assert_eq!(number(&json, "merged_vertices"), 1.0);
    assert_eq!(number(&json, "removed_components"), 1.0);
    let after = measure_json(&output);
    assert_eq!(number(&after, "triangles"), 12.0);
    assert_eq!(field(&after, "closed"), "true");
}

#[test]
fn repair_max_edges_any_and_equals_syntax() {
    let dir = Dir::new("repair-any");
    let mut soup = unit_cube();
    soup.pop();
    let input = dir.file("in.stl", &ascii_stl(&soup));
    let output = dir.path("out.stl");
    let o = run(&args![
        "repair",
        &input,
        "--output",
        &output,
        "--auto",
        "--max-edges=any",
        "--format=json"
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(number(&stdout(&o), "filled"), 1.0);
}

#[test]
fn repair_warns_on_stderr_about_holes_left_open() {
    let dir = Dir::new("repair-warn");
    let mut soup = unit_cube();
    soup.pop();
    let input = dir.file("in.stl", &ascii_stl(&soup));
    let output = dir.path("out.stl");
    let o = run(&args![
        "repair",
        &input,
        "-o",
        &output,
        "--auto",
        "--max-edges",
        "2",
        "--format",
        "json"
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stderr(&o).contains("open"), "{}", stderr(&o));
    let json = stdout(&o);
    assert_eq!(number(&json, "filled"), 0.0);
    assert_eq!(number(&json, "skipped"), 1.0);
}

#[test]
fn repair_rejects_bad_options_as_usage_errors() {
    let dir = Dir::new("repair-usage");
    let input = dir.file("in.stl", &ascii_stl(&unit_cube()));
    let output = dir.path("out.stl");
    let with = |extra: &[&str]| {
        let mut list = args!["repair", &input, "-o", &output].to_vec();
        list.extend(extra.iter().map(OsString::from));
        run(&list)
    };
    assert_eq!(code(&with(&[])), 2, "--auto is required");
    for bad in [
        ["--auto", "--weld-tolerance", "-1"],
        ["--auto", "--weld-tolerance", "nan"],
        ["--auto", "--weld-tolerance", "1e99"],
        ["--auto", "--min-triangles", "many"],
        ["--auto", "--max-edges", "-3"],
    ] {
        let o = with(&bad);
        assert_eq!(code(&o), 2, "{bad:?}");
        assert!(o.stdout.is_empty());
        assert!(!stderr(&o).is_empty());
    }
    assert!(!output.exists());
}

/// Expected values by hand: two unit cubes, one moved by (½, ½, ½), overlap in
/// a cube of side ½. Each cube has 0.75 of its surface inside the other, so the
/// union's area is 12 - 2 × 0.75 and the difference trades A's 0.75 for B's.
fn check_booleans(offset: f64) {
    let dir = Dir::new(&format!("boolean-{offset}"));
    let at = [offset; 3];
    let moved = [offset + 0.5; 3];
    let a = dir.file("a.stl", &binary_stl(&cube(at, 1.0)));
    let b = dir.file("b.stl", &ascii_stl(&cube(moved, 1.0)));
    for (op, volume, area) in [
        ("union", 1.875, 10.5),
        ("intersection", 0.125, 1.5),
        ("difference", 0.875, 6.0),
    ] {
        let output = dir.path(&format!("{op}.stl"));
        let o = run(&args![
            "boolean", &a, &b, "--op", op, "-o", &output, "--format", "json"
        ]);
        assert_eq!(code(&o), 0, "{op}: {}", stderr(&o));
        assert_eq!(field(&stdout(&o), "closed"), "true", "{op}");
        let json = measure_json(&output);
        assert_eq!(field(&json, "closed"), "true", "{op}");
        assert!(
            (number(&json, "volume") - volume).abs() < 1e-3,
            "{op}: {json}"
        );
        assert!((number(&json, "area") - area).abs() < 1e-3, "{op}: {json}");
    }
}

#[test]
fn boolean_of_overlapping_cubes() {
    check_booleans(0.0);
}

/// The two inputs get different local origins here: the tool has to be mapped
/// into the first one's frame.
#[test]
fn boolean_far_from_the_origin() {
    check_booleans(1000.0);
}

#[test]
fn boolean_refuses_an_open_input_without_output() {
    let dir = Dir::new("boolean-open");
    let mut open = unit_cube();
    open.pop();
    let a = dir.file("a.stl", &ascii_stl(&open));
    let b = dir.file("b.stl", &ascii_stl(&cube([0.5; 3], 1.0)));
    let output = dir.path("out.stl");
    let o = run(&args!["boolean", &a, &b, "--op", "union", "-o", &output]);
    assert_eq!(code(&o), 1);
    assert!(!stderr(&o).is_empty());
    assert_eq!(dir.names(), ["a.stl", "b.stl"]);
}

#[test]
fn boolean_with_an_empty_result_writes_nothing() {
    let dir = Dir::new("boolean-empty");
    let a = dir.file("a.stl", &ascii_stl(&unit_cube()));
    let b = dir.file("b.stl", &ascii_stl(&cube([5.0; 3], 1.0)));
    let output = dir.path("out.stl");
    let o = run(&args![
        "boolean",
        &a,
        &b,
        "--op",
        "intersection",
        "-o",
        &output
    ]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("no triangles"), "{}", stderr(&o));
    assert_eq!(dir.names(), ["a.stl", "b.stl"]);
}

#[test]
fn boolean_usage() {
    let dir = Dir::new("boolean-usage");
    let a = dir.file("a.stl", &ascii_stl(&unit_cube()));
    let b = dir.file("b.stl", &ascii_stl(&cube([0.5; 3], 1.0)));
    let output = dir.path("out.stl");
    let o = run(&args!["boolean", &a, &b, "--op", "xor", "-o", &output]);
    assert_eq!(code(&o), 2);
    let o = run(&args!["boolean", &a, "--op", "union", "-o", &output]);
    assert_eq!(code(&o), 2, "one input");
    let o = run(&args!["boolean", &a, &b, "-o", &output]);
    assert_eq!(code(&o), 2, "no --op");
}

/// Triangles with the smallest corner first (winding kept), sorted: equal
/// geometry whatever the vertex numbering. Read by the test from the binary
/// STL the CLI writes.
fn triangles_of_binary_stl(bytes: &[u8]) -> Vec<[[f32; 3]; 3]> {
    let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 84 + 50 * n);
    let mut out = Vec::new();
    for t in 0..n {
        let at = 84 + 50 * t + 12;
        let f =
            |k: usize| f32::from_le_bytes(bytes[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
        let c = [[f(0), f(1), f(2)], [f(3), f(4), f(5)], [f(6), f(7), f(8)]];
        let first = (0..3)
            .min_by(|&i, &j| c[i].partial_cmp(&c[j]).unwrap())
            .unwrap();
        out.push([c[first], c[(first + 1) % 3], c[(first + 2) % 3]]);
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

#[test]
fn format_round_trips_preserve_geometry() {
    let dir = Dir::new("round-trip");
    // Coordinates that are not round in binary, and a second cube further out.
    let mut soup = cube([0.1, -0.2, 0.3], 0.7);
    soup.extend(cube([100.25, 7.5, -3.125], 2.0));
    let source = dir.file("source.stl", &binary_stl(&soup));
    let expected = triangles_of_binary_stl(&fs::read(&source).unwrap());

    for (name, flags) in [
        ("a.stl", &[][..]),
        ("b.stl", &["--ascii"][..]),
        ("c.obj", &[][..]),
        ("d.ply", &[][..]),
        ("e.ply", &["--ascii"][..]),
        ("f.glb", &[][..]),
    ] {
        let there = dir.path(name);
        let mut list = args!["convert", &source, "-o", &there].to_vec();
        list.extend(flags.iter().map(OsString::from));
        let o = run(&list);
        assert_eq!(code(&o), 0, "{name}: {}", stderr(&o));

        let back = dir.path(&format!("{name}.back.stl"));
        let o = run(&args!["convert", &there, "-o", &back]);
        assert_eq!(code(&o), 0, "{name}: {}", stderr(&o));
        assert_eq!(
            triangles_of_binary_stl(&fs::read(&back).unwrap()),
            expected,
            "{name}"
        );
        let json = measure_json(&there);
        assert_eq!(number(&json, "vertices"), 16.0, "{name}");
        assert_eq!(number(&json, "triangles"), 24.0, "{name}");
    }
}

#[test]
fn convert_reports_json() {
    let dir = Dir::new("convert-json");
    let source = dir.file("s.stl", &ascii_stl(&unit_cube()));
    let out = dir.path("o.glb");
    let o = run(&args!["convert", &source, "-o", &out, "--format", "json"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(o.stderr.is_empty());
    let json = stdout(&o);
    assert_eq!(number(&json, "triangles"), 12.0);
    assert_eq!(number(&json, "vertices"), 8.0);
}

#[test]
fn ascii_flag_writes_text_and_is_refused_where_meaningless() {
    let dir = Dir::new("ascii");
    let source = dir.file("s.stl", &binary_stl(&unit_cube()));
    for (name, head) in [("a.stl", "solid"), ("b.ply", "ply\nformat ascii")] {
        let out = dir.path(name);
        let o = run(&args!["convert", &source, "-o", &out, "--ascii"]);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        assert!(fs::read_to_string(&out).unwrap().starts_with(head));
    }
    for name in ["c.obj", "d.glb"] {
        let out = dir.path(name);
        let o = run(&args!["convert", &source, "-o", &out, "--ascii"]);
        assert_eq!(code(&o), 2, "{name}");
        assert!(!out.exists());
    }
}

#[test]
fn convert_drops_unused_vertices() {
    let dir = Dir::new("compact");
    let obj = "v 9 9 9\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 2 3 4\n";
    let input = dir.file("in.obj", obj.as_bytes());
    let out = dir.path("out.obj");
    let o = run(&args!["convert", &input, "-o", &out]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = fs::read_to_string(&out).unwrap();
    assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 3);
    assert!(!text.contains('9'), "{text}");
    assert!(text.contains("f 1 2 3"), "{text}");
}

#[test]
fn exit_codes() {
    let dir = Dir::new("exit-codes");
    let good = dir.file("good.stl", &ascii_stl(&unit_cube()));
    let bad = dir.file("bad.stl", b"this is not a mesh file at all");
    let trunc = dir.file("trunc.stl", &binary_stl(&unit_cube())[..200]);
    let out = dir.path("out.stl");
    let missing = dir.path("missing.stl");
    let good = good.to_str().unwrap();
    let bad = bad.to_str().unwrap();
    let trunc = trunc.to_str().unwrap();
    let out = out.to_str().unwrap();
    let missing = missing.to_str().unwrap();

    let usage: &[&[&str]] = &[
        &[],
        &["frobnicate"],
        &["measure"],
        &["measure", good, "--nope"],
        &["measure", good, "--format"],
        &["measure", good, "--format", "xml"],
        &["measure", good, "--format", "json", "--format", "json"],
        &["measure", good, "--ascii"],
        &["measure", good, "extra"],
        &["measure", "mesh.xyz"],
        &["measure", "mesh"],
        &["measure", "mesh.gltf"],
        &["convert", good],
        &["convert", good, "-o", "out.xyz"],
        &["convert", good, "-o", "out.gltf"],
        &["convert", good, "-o"],
    ];
    for args in usage {
        let o = run(args);
        assert_eq!(code(&o), 2, "{args:?}");
        assert!(o.stdout.is_empty(), "{args:?}");
        assert!(!o.stderr.is_empty(), "{args:?}");
    }

    let runtime: &[&[&str]] = &[
        &["measure", missing],
        &["measure", bad],
        &["measure", trunc, "--format", "json"],
        &["convert", bad, "-o", out],
    ];
    for args in runtime {
        let o = run(args);
        assert_eq!(code(&o), 1, "{args:?}");
        assert!(
            o.stdout.is_empty(),
            "stdout stays machine-readable: {args:?}"
        );
        assert!(!o.stderr.is_empty(), "{args:?}");
    }
    assert!(!Path::new(out).exists());

    for args in [
        &["--help"][..],
        &["-h"][..],
        &["help"][..],
        &["measure", "--help"][..],
        &["--version"][..],
    ] {
        let o = run(args);
        assert_eq!(code(&o), 0, "{args:?}");
        assert!(!o.stdout.is_empty(), "{args:?}");
    }
}

#[test]
fn help_documents_exit_codes() {
    let text = stdout(&run(&["--help"]));
    for word in ["repair", "measure", "boolean", "convert", "Exit codes"] {
        assert!(text.contains(word), "{word}");
    }
}

#[test]
fn refuses_to_overwrite_an_input_without_force() {
    let dir = Dir::new("overwrite");
    let mut soup = unit_cube();
    soup.pop();
    let original = binary_stl(&soup);
    let input = dir.file("in.stl", &original);
    let other = dir.file("other.stl", &ascii_stl(&cube([0.5; 3], 1.0)));
    let closed = dir.file("closed.stl", &ascii_stl(&unit_cube()));

    // The same file spelled another way counts too.
    let sub = dir.path("sub");
    fs::create_dir(&sub).unwrap();
    let spelled = sub.join("..").join("in.stl");
    for out in [&input, &spelled] {
        let o = run(&args!["repair", &input, "-o", out, "--auto"]);
        assert_eq!(code(&o), 2, "{}", stderr(&o));
        assert!(stderr(&o).contains("--force"), "{}", stderr(&o));
        assert_eq!(fs::read(&input).unwrap(), original);
    }
    for out in [&closed, &other] {
        let o = run(&args![
            "boolean", &closed, &other, "--op", "union", "-o", out
        ]);
        assert_eq!(code(&o), 2, "either boolean input");
    }

    let o = run(&args!["repair", &input, "-o", &input, "--auto", "--force"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(field(&measure_json(&input), "closed"), "true");
}

#[test]
fn a_failed_run_leaves_an_existing_output_alone() {
    let dir = Dir::new("no-partial");
    let bad = dir.file(
        "bad.ply",
        b"ply\nformat ascii 1.0\nelement vertex 3\nend_header\n",
    );
    let out = dir.file("out.stl", b"previous contents");
    let o = run(&args!["convert", &bad, "-o", &out]);
    assert_eq!(code(&o), 1);
    assert_eq!(fs::read(&out).unwrap(), b"previous contents");
    assert_eq!(dir.names(), ["bad.ply", "out.stl"]);
}

#[test]
fn an_unwritable_output_is_a_runtime_failure() {
    let dir = Dir::new("no-dir");
    let good = dir.file("in.stl", &ascii_stl(&unit_cube()));
    let out = dir.path("does/not/exist/out.stl");
    let o = run(&args!["convert", &good, "-o", &out]);
    assert_eq!(code(&o), 1);
    assert_eq!(dir.names(), ["in.stl"]);
}

#[test]
fn compressed_glb_fails_clearly() {
    // Only as much of a GLB as the parser needs to see the extension.
    let json = br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0},
        "extensions":{"KHR_draco_mesh_compression":{"bufferView":0,"attributes":{"POSITION":0}}}}]}]}"#;
    let mut padded = json.to_vec();
    while !padded.len().is_multiple_of(4) {
        padded.push(b' ');
    }
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + padded.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(padded.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&padded);
    let dir = Dir::new("draco");
    let input = dir.file("in.glb", &glb);
    let out = dir.path("out.stl");
    let o = run(&args!["convert", &input, "-o", &out]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("Draco"), "{}", stderr(&o));
    assert!(!out.exists());
}
