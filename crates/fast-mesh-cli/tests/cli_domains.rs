//! End-to-end integration tests for the hierarchical domain CLI subcommands
//! alongside legacy flat commands.

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

fn cube(at: [f64; 3], size: f64) -> Soup {
    CUBE_T
        .iter()
        .map(|t| t.map(|v| [0, 1, 2].map(|k| at[k] + size * CUBE_V[v][k])))
        .collect()
}

fn cube_obj(at: [f64; 3], size: f64) -> Vec<u8> {
    let mut out = String::new();
    for v in CUBE_V {
        out.push_str(&format!(
            "v {} {} {}\n",
            at[0] + size * v[0],
            at[1] + size * v[1],
            at[2] + size * v[2]
        ));
    }
    for t in CUBE_T {
        out.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
    }
    out.into_bytes()
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

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cli-domains-{name}"));
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
}

fn run<S: AsRef<OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fast-mesh"))
        .args(args)
        .output()
        .unwrap()
}

fn args_vec(list: &[&OsStr]) -> Vec<OsString> {
    list.iter().map(|s| s.to_os_string()).collect()
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn field<'a>(json: &'a str, key: &str) -> &'a str {
    let pat = format!("\"{key}\":");
    let start = json.find(&pat).unwrap_or_else(|| panic!("{key} in {json}")) + pat.len();
    let rest = &json[start..];
    if let Some(rest) = rest.strip_prefix('"') {
        let end = rest.find('"').unwrap();
        &rest[..end]
    } else {
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        &rest[..end]
    }
}

fn number(json: &str, key: &str) -> f64 {
    field(json, key).parse().unwrap()
}

#[test]
fn repair_auto_subcommand_closes_hole_without_auto_flag() {
    let dir = Dir::new("repair-auto");
    let mut soup = cube([0.0; 3], 1.0);
    soup.pop();
    let input = dir.file("open.stl", &ascii_stl(&soup));
    let output = dir.path("closed.stl");

    let args = args_vec(&[
        OsStr::new("repair"),
        OsStr::new("auto"),
        input.as_os_str(),
        OsStr::new("-o"),
        output.as_os_str(),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]);
    let o = run(&args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let report = stdout(&o);
    assert_eq!(field(&report, "command"), "repair");

    let m = run(&args_vec(&[
        OsStr::new("inspect"),
        OsStr::new("measure"),
        output.as_os_str(),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    assert_eq!(code(&m), 0, "{}", stderr(&m));
    assert_eq!(field(&stdout(&m), "closed"), "true");
    assert!((number(&stdout(&m), "volume") - 1.0).abs() < 1e-5);
}

#[test]
fn inspect_measure_and_analyze_subcommands_work() {
    let dir = Dir::new("inspect");
    let mut soup = cube([0.0; 3], 1.0);
    soup.pop();
    let open = dir.file("open.stl", &ascii_stl(&soup));
    let closed = dir.file("closed.obj", &cube_obj([0.0; 3], 1.0));

    let flat = run(&args_vec(&[
        OsStr::new("measure"),
        closed.as_os_str(),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    let hier = run(&args_vec(&[
        OsStr::new("inspect"),
        OsStr::new("measure"),
        closed.as_os_str(),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    assert_eq!(code(&flat), 0);
    assert_eq!(code(&hier), 0);
    assert_eq!(stdout(&flat), stdout(&hier));

    let analyze = run(&args_vec(&[
        OsStr::new("inspect"),
        OsStr::new("analyze"),
        open.as_os_str(),
        OsStr::new("--self-intersection"),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    assert_eq!(code(&analyze), 0, "{}", stderr(&analyze));
    let json = stdout(&analyze);
    assert_eq!(field(&json, "command"), "analyze");
    assert_eq!(number(&json, "boundary_loops"), 1.0);
    assert_eq!(field(&json, "self_intersects"), "false");
}

#[test]
fn boolean_subcommands_match_legacy_ops() {
    let dir = Dir::new("boolean");
    let a = dir.file("a.obj", &cube_obj([0.0; 3], 1.0));
    let b = dir.file("b.obj", &cube_obj([0.5, 0.25, 0.125], 1.0));

    for (sub, expected_vol) in [
        ("union", 2.0 - 21.0 / 64.0),
        ("intersection", 21.0 / 64.0),
        ("difference", 1.0 - 21.0 / 64.0),
    ] {
        let out = dir.path(&format!("{sub}.obj"));
        let o = run(&args_vec(&[
            OsStr::new("boolean"),
            OsStr::new(sub),
            a.as_os_str(),
            b.as_os_str(),
            OsStr::new("-o"),
            out.as_os_str(),
            OsStr::new("--format"),
            OsStr::new("json"),
        ]));
        assert_eq!(code(&o), 0, "{sub}: {}", stderr(&o));
        assert_eq!(field(&stdout(&o), "closed"), "true");

        let m = run(&args_vec(&[
            OsStr::new("inspect"),
            OsStr::new("measure"),
            out.as_os_str(),
            OsStr::new("--format"),
            OsStr::new("json"),
        ]));
        assert_eq!(code(&m), 0);
        assert!(
            (number(&stdout(&m), "volume") - expected_vol).abs() < 1e-6,
            "{sub}: {}",
            stdout(&m)
        );
    }
}

#[test]
fn remesh_subdivide_and_simplify_subcommands_work() {
    let dir = Dir::new("remesh");
    let cube_in = dir.file("cube.obj", &cube_obj([0.0; 3], 1.0));
    let sub_out = dir.path("sub.obj");
    let sim_tri_out = dir.path("sim_tri.obj");
    let sim_ratio_out = dir.path("sim_ratio.obj");

    let sub = run(&args_vec(&[
        OsStr::new("remesh"),
        OsStr::new("subdivide"),
        cube_in.as_os_str(),
        OsStr::new("-o"),
        sub_out.as_os_str(),
        OsStr::new("--levels"),
        OsStr::new("1"),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    assert_eq!(code(&sub), 0, "{}", stderr(&sub));
    let sub_json = stdout(&sub);
    assert_eq!(field(&sub_json, "command"), "subdivide");
    assert_eq!(number(&sub_json, "triangles_added"), 36.0);

    let sim_tri = run(&args_vec(&[
        OsStr::new("remesh"),
        OsStr::new("simplify"),
        sub_out.as_os_str(),
        OsStr::new("-o"),
        sim_tri_out.as_os_str(),
        OsStr::new("--triangles"),
        OsStr::new("24"),
        OsStr::new("--lock-boundary"),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    assert_eq!(code(&sim_tri), 0, "{}", stderr(&sim_tri));
    let sim_json = stdout(&sim_tri);
    assert_eq!(field(&sim_json, "command"), "simplify");
    assert_eq!(field(&sim_json, "preserve_boundary"), "true");
    assert!(number(&sim_json, "collapsed") > 0.0);

    let sim_ratio = run(&args_vec(&[
        OsStr::new("remesh"),
        OsStr::new("simplify"),
        sub_out.as_os_str(),
        OsStr::new("-o"),
        sim_ratio_out.as_os_str(),
        OsStr::new("--ratio"),
        OsStr::new("0.5"),
        OsStr::new("--format"),
        OsStr::new("json"),
    ]));
    assert_eq!(code(&sim_ratio), 0, "{}", stderr(&sim_ratio));
    assert_eq!(number(&stdout(&sim_ratio), "target_triangles"), 24.0);
}

#[test]
fn domain_cli_help_and_usage_errors() {
    let dir = Dir::new("errors");
    let cube_in = dir.file("cube.obj", &cube_obj([0.0; 3], 1.0));
    let out = dir.path("out.obj");

    for help_args in [
        &["inspect", "--help"][..],
        &["remesh", "--help"][..],
        &["repair", "auto", "--help"][..],
        &["boolean", "union", "--help"][..],
    ] {
        let o = run(&help_args.iter().map(OsStr::new).collect::<Vec<_>>());
        assert_eq!(code(&o), 0, "{help_args:?}");
        assert!(!stdout(&o).is_empty(), "{help_args:?}");
    }

    for bad_args in [
        &["inspect"][..],
        &["inspect", "bogus", "in.obj"][..],
        &["remesh"][..],
        &["remesh", "bogus", "in.obj"][..],
        &[
            "remesh",
            "simplify",
            cube_in.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ][..],
        &[
            "remesh",
            "simplify",
            cube_in.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--triangles",
            "10",
            "--ratio",
            "0.5",
        ][..],
        &[
            "remesh",
            "simplify",
            cube_in.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--ratio",
            "1.5",
        ][..],
        &[
            "remesh",
            "subdivide",
            cube_in.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--levels",
            "0",
        ][..],
        &[
            "remesh",
            "simplify",
            cube_in.to_str().unwrap(),
            "-o",
            cube_in.to_str().unwrap(),
            "--triangles",
            "10",
        ][..],
    ] {
        let o = run(&bad_args.iter().map(OsStr::new).collect::<Vec<_>>());
        assert_eq!(code(&o), 2, "{bad_args:?}: {}", stderr(&o));
        assert!(stdout(&o).is_empty(), "{bad_args:?}");
    }
}
