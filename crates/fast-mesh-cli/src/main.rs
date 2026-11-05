//! `fast-mesh`: headless mesh repair, measurement, booleans and format
//! conversion on the Rust engine crates, with no WebAssembly or Node.js.
//!
//! Exit codes: 0 success, 1 the operation or a file failed, 2 usage error.
//! Reports go to stdout and diagnostics to stderr; with `--format json`,
//! stdout holds one JSON object on success and nothing on failure.

mod args;
mod report;

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use args::{Command, Input, Output, OutputFormat};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_cli::io::{self, MeshData};
use fast_mesh_ops::{
    AutoRepairReport, BooleanOp, MeasureReport, VolumeReason, apply_boolean, auto_repair,
    boolean_plan, measure,
};
use fast_mesh_query::Rigid;
use fast_mesh_store::Mesh;
use report::Value;

const EXIT_FAILURE: u8 = 1;
const EXIT_USAGE: u8 = 2;

/// Why a run stopped.
enum Failure {
    /// Exit code 2: bad arguments.
    Usage(String),
    /// Exit code 1: the operation or a file failed.
    Runtime(String),
}

fn runtime(e: impl std::fmt::Display) -> Failure {
    Failure::Runtime(e.to_string())
}

fn main() -> ExitCode {
    // A bug must not look like a signal to a script: exit 1, whichever panic
    // strategy the profile uses. Output is only ever renamed into place after
    // it is complete, so nothing partial can appear.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("fast-mesh: internal error: {info}");
        std::process::exit(i32::from(EXIT_FAILURE));
    }));
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("fast-mesh: {message}");
            eprintln!("Run `fast-mesh --help` for usage.");
            ExitCode::from(EXIT_USAGE)
        }
        Err(Failure::Runtime(message)) => {
            eprintln!("fast-mesh: {message}");
            ExitCode::from(EXIT_FAILURE)
        }
    }
}

fn run() -> Result<(), Failure> {
    let command = args::parse(std::env::args_os().skip(1)).map_err(Failure::Usage)?;
    match command {
        Command::Help(text) => print(text),
        Command::Version => print(&format!("fast-mesh {}\n", env!("CARGO_PKG_VERSION"))),
        Command::Measure { input, report } => emit(&run_measure(&input)?, report),
        Command::Repair {
            input,
            output,
            options,
            report,
        } => {
            refuse_overwrite(&output, &[&input])?;
            let value = run_repair(&input, &output, &options)?;
            emit(&value, report)
        }
        Command::Boolean {
            a,
            b,
            op,
            output,
            report,
        } => {
            refuse_overwrite(&output, &[&a, &b])?;
            let value = run_boolean(&a, &b, op, &output)?;
            emit(&value, report)
        }
        Command::Convert {
            input,
            output,
            report,
        } => {
            refuse_overwrite(&output, &[&input])?;
            let data = read(&input)?;
            write(&data, &output)?;
            let value = Value::Object(vec![
                ("command", Value::Text("convert".into())),
                ("input", path_text(&input.path)),
                ("output", path_text(&output.path)),
                ("vertices", Value::Count(data.positions.len() / 3)),
                ("triangles", Value::Count(data.indices.len() / 3)),
            ]);
            emit(&value, report)
        }
    }
}

fn print(text: &str) -> Result<(), Failure> {
    std::io::stdout()
        .write_all(text.as_bytes())
        .map_err(|e| runtime(format!("cannot write to stdout: {e}")))
}

fn emit(value: &Value, format: OutputFormat) -> Result<(), Failure> {
    match format {
        OutputFormat::Text => print(&value.to_text()),
        OutputFormat::Json => print(&format!("{}\n", value.to_json())),
    }
}

fn path_text(path: &Path) -> Value {
    Value::Text(path.display().to_string())
}

/// Refuses, before any work, an output that names an input (however spelled).
fn refuse_overwrite(output: &Output, inputs: &[&Input]) -> Result<(), Failure> {
    if output.force {
        return Ok(());
    }
    let Ok(target) = fs::canonicalize(&output.path) else {
        return Ok(()); // It does not exist yet.
    };
    for input in inputs {
        if fs::canonicalize(&input.path).is_ok_and(|p| p == target) {
            return Err(Failure::Usage(format!(
                "the output {} is also an input; pass --force to overwrite it",
                output.path.display()
            )));
        }
    }
    Ok(())
}

fn read(input: &Input) -> Result<MeshData, Failure> {
    let in_file =
        |e: &dyn std::fmt::Display| Failure::Runtime(format!("{}: {e}", input.path.display()));
    let bytes = fs::read(&input.path).map_err(|e| in_file(&e))?;
    io::parse(input.format, &bytes).map_err(|e| in_file(&e))
}

fn write(data: &MeshData, output: &Output) -> Result<(), Failure> {
    io::atomic_write(&output.path, |out| {
        io::write_mesh(data, output.format, output.ascii, out)
    })
    .map_err(|e| Failure::Runtime(format!("{}: {e}", output.path.display())))
}

fn to_mesh(data: &MeshData, input: &Input) -> Result<Mesh, Failure> {
    Mesh::from_buffers_f64(&data.positions, &data.indices)
        .map_err(|e| Failure::Runtime(format!("{}: {e}", input.path.display())))
}

/// A mesh's live triangles over all its vertex slots, in world coordinates.
/// Writing compacts away the slots no live triangle uses.
fn from_mesh(mesh: &Mesh) -> Result<MeshData, Failure> {
    let origin = mesh.origin();
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(3 * mesh.vertex_slots())
        .map_err(|_| runtime("out of memory"))?;
    for p in mesh.positions() {
        for k in 0..3 {
            positions.push(origin[k] + f64::from(p[k]));
        }
    }
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(3 * mesh.triangle_count())
        .map_err(|_| runtime("out of memory"))?;
    for (t, corners) in mesh.triangles().iter().enumerate() {
        if mesh.is_triangle_alive(t as u32) {
            indices.extend_from_slice(corners);
        }
    }
    Ok(MeshData { positions, indices })
}

fn run_measure(input: &Input) -> Result<Value, Failure> {
    let data = read(input)?;
    let mesh = to_mesh(&data, input)?;
    let report = measure(&mesh).map_err(runtime)?;
    Ok(Value::Object(vec![
        ("command", Value::Text("measure".into())),
        ("input", path_text(&input.path)),
        ("vertices", Value::Count(mesh.vertex_count())),
        ("triangles", Value::Count(mesh.triangle_count())),
        ("area", Value::Number(report.area)),
        ("signed_volume", Value::Number(report.signed_volume)),
        ("volume", report.volume.map_or(Value::Null, Value::Number)),
        ("closed", Value::Bool(report.closed)),
        (
            "volume_reason",
            match report.volume_reason {
                None => Value::Null,
                Some(VolumeReason::Open) => Value::Text("open".into()),
                Some(VolumeReason::NonManifold) => Value::Text("non_manifold".into()),
            },
        ),
        ("boundary_loops", Value::Count(report.boundary_loops)),
        (
            "non_manifold_edges",
            Value::Count(report.non_manifold_edges),
        ),
        (
            "orientation_conflicts",
            Value::Count(report.orientation_conflicts),
        ),
    ]))
}

fn counts(mesh: &Mesh) -> Value {
    Value::Object(vec![
        ("vertices", Value::Count(mesh.vertex_count())),
        ("triangles", Value::Count(mesh.triangle_count())),
    ])
}

fn run_repair(
    input: &Input,
    output: &Output,
    options: &fast_mesh_ops::AutoRepairOptions,
) -> Result<Value, Failure> {
    let data = read(input)?;
    let mut mesh = to_mesh(&data, input)?;
    drop(data);
    let before = counts(&mesh);
    let mut tx = mesh.begin_edit();
    // On `Err` the transaction drops uncommitted: nothing is kept.
    let repair = auto_repair(&mut tx, options).map_err(runtime)?;
    tx.commit().map_err(runtime)?;
    warn_open_holes(&repair);
    write(&from_mesh(&mesh)?, output)?;
    Ok(Value::Object(vec![
        ("command", Value::Text("repair".into())),
        ("input", path_text(&input.path)),
        ("output", path_text(&output.path)),
        ("before", before),
        ("after", counts(&mesh)),
        ("stages", repair_stages(&repair)),
    ]))
}

fn warn_open_holes(repair: &AutoRepairReport) {
    let (skipped, failed) = (repair.fill.skipped.len(), repair.fill.failed.len());
    if skipped + failed > 0 {
        eprintln!(
            "fast-mesh: warning: {skipped} boundary loop(s) skipped and {failed} failed: the output is still open"
        );
    }
}

fn repair_stages(r: &AutoRepairReport) -> Value {
    let added = |f: fn(&fast_mesh_ops::FilledLoop) -> usize| r.fill.filled.iter().map(f).sum();
    Value::Object(vec![
        (
            "weld",
            Value::Object(vec![
                ("merged_vertices", Value::Count(r.weld.merged_vertices)),
                ("removed_triangles", Value::Count(r.weld.removed_triangles)),
            ]),
        ),
        (
            "degenerate",
            Value::Object(vec![
                (
                    "removed_triangles",
                    Value::Count(r.degenerate.removed_triangles),
                ),
                (
                    "removed_vertices",
                    Value::Count(r.degenerate.removed_vertices),
                ),
            ]),
        ),
        (
            "orient",
            Value::Object(vec![
                (
                    "flipped_triangles",
                    Value::Count(r.orient.flipped_triangles),
                ),
                ("patches", Value::Count(r.orient.patches)),
                (
                    "non_orientable_patches",
                    Value::Count(r.orient.non_orientable_patches),
                ),
                (
                    "flipped_outward_components",
                    Value::Count(r.orient.flipped_outward_components),
                ),
            ]),
        ),
        (
            "components",
            Value::Object(vec![
                (
                    "removed_components",
                    Value::Count(r.components.removed_components),
                ),
                (
                    "removed_triangles",
                    Value::Count(r.components.removed_triangles),
                ),
                (
                    "removed_vertices",
                    Value::Count(r.components.removed_vertices),
                ),
            ]),
        ),
        (
            "fill",
            Value::Object(vec![
                ("filled", Value::Count(r.fill.filled.len())),
                ("skipped", Value::Count(r.fill.skipped.len())),
                ("failed", Value::Count(r.fill.failed.len())),
                (
                    "triangles_added",
                    Value::Count(added(|l| l.triangles_added)),
                ),
                ("vertices_added", Value::Count(added(|l| l.vertices_added))),
            ]),
        ),
    ])
}

fn run_boolean(a: &Input, b: &Input, op: BooleanOp, output: &Output) -> Result<Value, Failure> {
    let mut mesh_a = to_mesh(&read(a)?, a)?;
    let mesh_b = to_mesh(&read(b)?, b)?;
    let build = |mesh: &Mesh, input: &Input| {
        Bvh::build(mesh, &BuildOptions::default())
            .map_err(|e| Failure::Runtime(format!("{}: {e}", input.path.display())))
    };
    let bvh_a = build(&mesh_a, a)?;
    let bvh_b = build(&mesh_b, b)?;
    // Each mesh is stored relative to its own origin: B's coordinates in A's
    // frame are B's own plus the difference of the origins.
    let (from, to) = (mesh_b.origin(), mesh_a.origin());
    let xf = Rigid {
        translation: [0, 1, 2].map(|k| from[k] - to[k]),
        ..Rigid::IDENTITY
    };
    let plan = boolean_plan(&mesh_a, &bvh_a, &mesh_b, &bvh_b, &xf, op).map_err(runtime)?;
    drop(bvh_a);
    let mut tx = mesh_a.begin_edit();
    // On `Err` the transaction drops uncommitted: nothing is kept.
    let applied = apply_boolean(&mut tx, &plan).map_err(runtime)?;
    tx.commit().map_err(runtime)?;
    // Before writing: a refused result must not replace an existing output.
    let result = measure(&mesh_a).map_err(runtime)?;
    if let Some(message) = invalid_boolean_result(&result) {
        return Err(Failure::Runtime(message));
    }
    write(&from_mesh(&mesh_a)?, output)?;
    Ok(Value::Object(vec![
        ("command", Value::Text("boolean".into())),
        ("a", path_text(&a.path)),
        ("b", path_text(&b.path)),
        ("output", path_text(&output.path)),
        ("closed", Value::Bool(result.closed)),
        ("kept_triangles", Value::Count(applied.kept_triangles)),
        ("triangles_added", Value::Count(applied.triangles_added)),
    ]))
}

/// Why a boolean's result cannot be trusted, if so. Closed alone is not enough:
/// the engine can return a closed mesh with flipped patches and a wrong volume.
/// This is a structural check: it cannot tell a sound mesh of the wrong shape.
fn invalid_boolean_result(result: &MeasureReport) -> Option<String> {
    (!result.closed || result.orientation_conflicts > 0).then(|| {
        format!(
            "the engine produced an invalid result ({} boundary loop(s), {} non-manifold edge(s), {} orientation conflict(s)), so nothing was written; see fast-mesh-wasm issue #165",
            result.boundary_loops, result.non_manifold_edges, result.orientation_conflicts
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sound() -> MeasureReport {
        MeasureReport {
            area: 6.0,
            signed_volume: 1.0,
            volume: Some(1.0),
            closed: true,
            volume_reason: None,
            boundary_loops: 0,
            non_manifold_edges: 0,
            orientation_conflicts: 0,
        }
    }

    #[test]
    fn a_sound_result_is_valid() {
        assert_eq!(invalid_boolean_result(&sound()), None);
    }

    /// Each defect alone is enough.
    #[test]
    fn each_defect_is_invalid() {
        let open = MeasureReport {
            closed: false,
            boundary_loops: 2,
            ..sound()
        };
        let non_manifold = MeasureReport {
            closed: false,
            non_manifold_edges: 3,
            ..sound()
        };
        let conflicted = MeasureReport {
            orientation_conflicts: 4,
            ..sound()
        };
        for (report, count) in [
            (open, "2 boundary loop"),
            (non_manifold, "3 non-manifold edge"),
            (conflicted, "4 orientation conflict"),
        ] {
            let message = invalid_boolean_result(&report).unwrap();
            for word in ["invalid result", "#165", count] {
                assert!(message.contains(word), "{word}: {message}");
            }
        }
    }
}
