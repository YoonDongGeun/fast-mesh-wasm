//! The command line: a small hand-written parser, so that the binary adds no
//! dependency.

use std::ffi::OsString;
use std::path::PathBuf;

use fast_mesh_cli::io::Format;
use fast_mesh_ops::boolean::BooleanOp;
use fast_mesh_ops::repair::AutoRepairOptions;

/// How reports are printed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Where a command writes its mesh.
#[derive(Debug, Clone)]
pub struct Output {
    pub path: PathBuf,
    pub format: Format,
    pub ascii: bool,
    pub force: bool,
}

/// A mesh file to read.
#[derive(Debug, Clone)]
pub struct Input {
    pub path: PathBuf,
    pub format: Format,
}

/// How [`Command::Simplify`] determines its target triangle count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimplifyTarget {
    Triangles(usize),
    Ratio(f64),
}

/// What to do.
#[derive(Debug, Clone)]
pub enum Command {
    Help(&'static str),
    Version,
    Repair {
        input: Input,
        output: Output,
        options: AutoRepairOptions,
        report: OutputFormat,
    },
    Measure {
        input: Input,
        report: OutputFormat,
    },
    Analyze {
        input: Input,
        weld_tolerance: f32,
        self_intersection: bool,
        report: OutputFormat,
    },
    Boolean {
        a: Input,
        b: Input,
        op: BooleanOp,
        output: Output,
        report: OutputFormat,
    },
    Simplify {
        input: Input,
        target: SimplifyTarget,
        preserve_boundary: bool,
        output: Output,
        report: OutputFormat,
    },
    Subdivide {
        input: Input,
        levels: u32,
        output: Output,
        report: OutputFormat,
    },
    Convert {
        input: Input,
        output: Output,
        report: OutputFormat,
    },
}

pub const HELP: &str = "\
fast-mesh: mesh repair, inspection, remeshing, booleans and format conversion

Usage:
  fast-mesh repair auto <in> -o <out> [repair options] [output options]
  fast-mesh inspect measure <in> [--format text|json]
  fast-mesh inspect analyze <in> [--weld-tolerance <d>] [--self-intersection] [--format text|json]
  fast-mesh remesh simplify <in> -o <out> (--triangles <n> | --ratio <r>) [--lock-boundary] [output options]
  fast-mesh remesh subdivide <in> -o <out> [--levels <n>] [output options]
  fast-mesh boolean union|intersection|difference <a> <b> -o <out> [output options]

Legacy aliases:
  fast-mesh repair <in> -o <out> --auto [repair options] [output options]
  fast-mesh measure <in> [--format text|json]
  fast-mesh boolean <a> <b> --op union|intersection|difference -o <out> [output options]
  fast-mesh convert <in> -o <out> [output options]

Formats, chosen by file extension: .stl .obj .ply .glb

Run `fast-mesh <command> --help` for a command's options.

Exit codes:
  0  success
  1  the operation or a file failed (bad input, I/O, empty or invalid result); nothing is written
  2  usage error (bad arguments, or an output that would overwrite an input)

Reports go to stdout, diagnostics to stderr. With --format json, stdout holds
one JSON object on success and nothing on failure.
";

const REPAIR_HELP: &str = "\
fast-mesh repair auto <in> -o <out> [options]
fast-mesh repair <in> -o <out> --auto [options]

Runs the standard cleanup: weld, remove degenerate triangles, orient, remove
small components, fill holes.

  --auto                  Run the whole pipeline (required without `auto` subcommand)
  --weld-tolerance <d>    Weld vertices at most this far apart (default 1e-5)
  --min-triangles <n>     Remove components with fewer triangles; the largest
                          always stays (default 500)
  --max-edges <n|any>     Fill holes with at most this many edges (default 64)
  --no-fair               Fill holes flat, without refining and fairing
  -o, --output <path>     The output file
  --ascii                 Text form of .stl and .ply
  --force                 Allow the output to overwrite the input
  --format text|json      The report's format (default text)
";

const INSPECT_HELP: &str = "\
fast-mesh inspect <subcommand> <in> [options]

Subcommands:
  measure <in>            Report vertex/triangle counts, area, volume and manifold status
  analyze <in>            Inspect mesh defects (loops, coincident, degenerate, non-manifold, pinches)

Analyze options:
  --weld-tolerance <d>    Tolerance for coincident vertices (default 0)
  --self-intersection     Build BVH and check for self-intersecting triangles

Common options:
  --format text|json      The report's format (default text)
";

const MEASURE_HELP: &str = "\
fast-mesh measure <in> [options]

Reports vertex and triangle counts, area, volume (when watertight) and defects.

  --format text|json      The report's format (default text)
";

const BOOLEAN_HELP: &str = "\
fast-mesh boolean union|intersection|difference <a> <b> -o <out> [options]
fast-mesh boolean <a> <b> --op <op> -o <out> [options]

Combines two closed, manifold, consistently wound meshes. A result that is
open, non-manifold or has orientation conflicts fails (exit 1), writing nothing.

  --op union|intersection|difference
                          difference is a minus b (legacy flag without subcommand)
  -o, --output <path>     The output file
  --ascii                 Text form of .stl and .ply
  --force                 Allow the output to overwrite an input
  --format text|json      The report's format (default text)
";

const REMESH_HELP: &str = "\
fast-mesh remesh <subcommand> <in> -o <out> [options]

Subcommands:
  simplify <in> -o <out>  Shrink mesh via quadric-error edge collapse
    --triangles <n>       Target triangle count (>= 1; mutually exclusive with --ratio)
    --ratio <r>           Target fraction of input triangles in (0, 1]
    --lock-boundary       Keep boundary vertices in place

  subdivide <in> -o <out> Refine mesh via Loop subdivision
    --levels <n>          Subdivision levels >= 1 (default 1)

Output options:
  -o, --output <path>     The output file
  --ascii                 Text form of .stl and .ply
  --force                 Allow the output to overwrite an input
  --format text|json      The report's format (default text)
";

const CONVERT_HELP: &str = "\
fast-mesh convert <in> -o <out> [options]

  -o, --output <path>     The output file
  --ascii                 Text form of .stl and .ply
  --force                 Allow the output to overwrite the input
  --format text|json      The report's format (default text)
";

/// The arguments of one command, split into positionals and options.
struct Parsed {
    positional: Vec<OsString>,
    options: Vec<(&'static str, Option<OsString>)>,
}

impl Parsed {
    fn take(&mut self, name: &str) -> Option<Option<OsString>> {
        let at = self.options.iter().position(|(n, _)| *n == name)?;
        Some(self.options.remove(at).1)
    }

    fn flag(&mut self, name: &str) -> bool {
        self.take(name).is_some()
    }

    fn value(&mut self, name: &str) -> Option<OsString> {
        self.take(name).flatten()
    }

    fn text(&mut self, name: &str) -> Result<Option<String>, String> {
        match self.value(name) {
            None => Ok(None),
            Some(v) => v
                .into_string()
                .map(Some)
                .map_err(|_| format!("{name}: the value is not valid text")),
        }
    }
}

/// `spec` lists a command's options as (name, takes a value). `-o` is the
/// short form of `--output`. Returns `None` when help was asked for.
fn split(
    args: impl Iterator<Item = OsString>,
    spec: &[(&'static str, bool)],
) -> Result<Option<Parsed>, String> {
    let mut parsed = Parsed {
        positional: Vec::new(),
        options: Vec::new(),
    };
    let mut args = args;
    let mut only_positional = false;
    while let Some(arg) = args.next() {
        let text = arg.to_str().unwrap_or("");
        if only_positional || !text.starts_with('-') || text == "-" {
            parsed.positional.push(arg);
            continue;
        }
        match text {
            "--" => {
                only_positional = true;
                continue;
            }
            "-h" | "--help" => return Ok(None),
            _ => {}
        }
        let (name, inline) = match text.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value)),
            _ => (text, None),
        };
        let name = if name == "-o" { "--output" } else { name };
        let &(name, takes_value) = spec
            .iter()
            .find(|(n, _)| *n == name)
            .ok_or_else(|| format!("unknown option {name}"))?;
        if parsed.options.iter().any(|(n, _)| *n == name) {
            return Err(format!("{name} given twice"));
        }
        let value = match (takes_value, inline) {
            (true, Some(v)) => Some(OsString::from(v)),
            (true, None) => Some(args.next().ok_or_else(|| format!("{name} needs a value"))?),
            (false, None) => None,
            (false, Some(_)) => return Err(format!("{name} takes no value")),
        };
        parsed.options.push((name, value));
    }
    Ok(Some(parsed))
}

const FORMAT: (&str, bool) = ("--format", true);
const OUTPUT: [(&str, bool); 3] = [("--output", true), ("--ascii", false), ("--force", false)];

fn input(path: OsString) -> Result<Input, String> {
    let path = PathBuf::from(path);
    let format = Format::from_path(&path).map_err(|e| e.to_string())?;
    Ok(Input { path, format })
}

fn output(parsed: &mut Parsed) -> Result<Output, String> {
    let path = PathBuf::from(
        parsed
            .value("--output")
            .ok_or("an output file is required (-o <path>)")?,
    );
    let format = Format::from_path(&path).map_err(|e| e.to_string())?;
    let ascii = parsed.flag("--ascii");
    if ascii && !format.has_ascii_form() {
        return Err(format!(
            "--ascii applies to .stl and .ply, not {}",
            path.display()
        ));
    }
    let force = parsed.flag("--force");
    Ok(Output {
        path,
        format,
        ascii,
        force,
    })
}

fn report_format(parsed: &mut Parsed) -> Result<OutputFormat, String> {
    match parsed.text("--format")?.as_deref() {
        None | Some("text") => Ok(OutputFormat::Text),
        Some("json") => Ok(OutputFormat::Json),
        Some(other) => Err(format!("--format must be text or json, not {other}")),
    }
}

fn positionals<const N: usize>(parsed: &mut Parsed, names: &str) -> Result<[OsString; N], String> {
    <[OsString; N]>::try_from(std::mem::take(&mut parsed.positional))
        .map_err(|got| format!("expected {names}, got {} file argument(s)", got.len()))
}

fn number<T: std::str::FromStr>(parsed: &mut Parsed, name: &str) -> Result<Option<T>, String> {
    parsed
        .text(name)?
        .map(|v| {
            v.parse()
                .map_err(|_| format!("{name}: \"{v}\" is not a valid value"))
        })
        .transpose()
}

fn repair_options(parsed: &mut Parsed) -> Result<AutoRepairOptions, String> {
    let mut options = AutoRepairOptions::default();
    if let Some(weld) = number::<f32>(parsed, "--weld-tolerance")? {
        if !weld.is_finite() || weld < 0.0 {
            return Err(format!(
                "--weld-tolerance must be finite and not negative, not {weld}"
            ));
        }
        options.weld_tolerance = weld;
    }
    if let Some(min) = number(parsed, "--min-triangles")? {
        options.min_triangles = min;
    }
    match parsed.text("--max-edges")?.as_deref() {
        None => {}
        Some("any") => options.max_edges = None,
        Some(v) => {
            options.max_edges = Some(v.parse().map_err(|_| {
                format!("--max-edges: \"{v}\" is neither a number of edges nor \"any\"")
            })?);
        }
    }
    options.fair = !parsed.flag("--no-fair");
    Ok(options)
}

fn boolean_op_from_str(s: &str) -> Option<BooleanOp> {
    match s {
        "union" => Some(BooleanOp::Union),
        "intersection" => Some(BooleanOp::Intersect),
        "difference" => Some(BooleanOp::Subtract),
        _ => None,
    }
}

/// Parses the arguments after the program name. An `Err` is a usage error.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter().peekable();
    let first = args.next().ok_or("no command given")?;
    let command = first.to_str().unwrap_or("");
    match command {
        "-h" | "--help" | "help" => return Ok(Command::Help(HELP)),
        "-V" | "--version" => return Ok(Command::Version),
        "inspect" => return parse_inspect(args),
        "remesh" => return parse_remesh(args),
        "repair" => return parse_repair(args),
        "boolean" => return parse_boolean(args),
        _ => {}
    }
    let (spec, help): (&[(&'static str, bool)], &'static str) = match command {
        "measure" => (&[FORMAT], MEASURE_HELP),
        "convert" => (&[OUTPUT[0], OUTPUT[1], OUTPUT[2], FORMAT], CONVERT_HELP),
        other => return Err(format!("unknown command \"{other}\"")),
    };
    let Some(mut parsed) = split(args, spec)? else {
        return Ok(Command::Help(help));
    };
    let report = report_format(&mut parsed)?;
    match command {
        "measure" => {
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Measure {
                input: input(path)?,
                report,
            })
        }
        _ => {
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Convert {
                input: input(path)?,
                output: output(&mut parsed)?,
                report,
            })
        }
    }
}

fn parse_repair(
    mut args: std::iter::Peekable<impl Iterator<Item = OsString>>,
) -> Result<Command, String> {
    let has_auto_sub = args.peek().and_then(|s| s.to_str()) == Some("auto");
    if has_auto_sub {
        args.next();
    }
    const REPAIR_SPEC: [(&str, bool); 9] = [
        OUTPUT[0],
        OUTPUT[1],
        OUTPUT[2],
        FORMAT,
        ("--auto", false),
        ("--weld-tolerance", true),
        ("--min-triangles", true),
        ("--max-edges", true),
        ("--no-fair", false),
    ];
    let Some(mut parsed) = split(args, &REPAIR_SPEC)? else {
        return Ok(Command::Help(REPAIR_HELP));
    };
    let report = report_format(&mut parsed)?;
    let [path] = positionals(&mut parsed, "one input file")?;
    let has_auto_flag = parsed.flag("--auto");
    if !has_auto_sub && !has_auto_flag {
        return Err("repair needs --auto (or `fast-mesh repair auto <in>`)".into());
    }
    Ok(Command::Repair {
        options: repair_options(&mut parsed)?,
        input: input(path)?,
        output: output(&mut parsed)?,
        report,
    })
}

fn parse_boolean(
    mut args: std::iter::Peekable<impl Iterator<Item = OsString>>,
) -> Result<Command, String> {
    if let Some(sub_op) = args
        .peek()
        .and_then(|s| s.to_str())
        .and_then(boolean_op_from_str)
    {
        args.next();
        const SUB_SPEC: [(&str, bool); 4] = [OUTPUT[0], OUTPUT[1], OUTPUT[2], FORMAT];
        let Some(mut parsed) = split(args, &SUB_SPEC)? else {
            return Ok(Command::Help(BOOLEAN_HELP));
        };
        let report = report_format(&mut parsed)?;
        let [a, b] = positionals(&mut parsed, "two input files")?;
        return Ok(Command::Boolean {
            a: input(a)?,
            b: input(b)?,
            op: sub_op,
            output: output(&mut parsed)?,
            report,
        });
    }
    const BOOLEAN_SPEC: [(&str, bool); 5] =
        [OUTPUT[0], OUTPUT[1], OUTPUT[2], FORMAT, ("--op", true)];
    let Some(mut parsed) = split(args, &BOOLEAN_SPEC)? else {
        return Ok(Command::Help(BOOLEAN_HELP));
    };
    let report = report_format(&mut parsed)?;
    let [a, b] = positionals(&mut parsed, "two input files")?;
    let op = match parsed.text("--op")?.as_deref() {
        Some(raw) => boolean_op_from_str(raw)
            .ok_or_else(|| format!("--op must be union, intersection or difference, not {raw}"))?,
        None => return Err("boolean needs --op".into()),
    };
    Ok(Command::Boolean {
        a: input(a)?,
        b: input(b)?,
        op,
        output: output(&mut parsed)?,
        report,
    })
}

fn parse_inspect(mut args: impl Iterator<Item = OsString>) -> Result<Command, String> {
    let sub = args
        .next()
        .ok_or("inspect requires a subcommand: measure or analyze")?;
    let sub_name = sub.to_str().unwrap_or("");
    match sub_name {
        "-h" | "--help" => Ok(Command::Help(INSPECT_HELP)),
        "measure" => {
            let Some(mut parsed) = split(args, &[FORMAT])? else {
                return Ok(Command::Help(INSPECT_HELP));
            };
            let report = report_format(&mut parsed)?;
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Measure {
                input: input(path)?,
                report,
            })
        }
        "analyze" => {
            const ANALYZE_SPEC: [(&str, bool); 3] = [
                FORMAT,
                ("--weld-tolerance", true),
                ("--self-intersection", false),
            ];
            let Some(mut parsed) = split(args, &ANALYZE_SPEC)? else {
                return Ok(Command::Help(INSPECT_HELP));
            };
            let report = report_format(&mut parsed)?;
            let weld_tolerance = match number::<f32>(&mut parsed, "--weld-tolerance")? {
                None => 0.0,
                Some(t) if t.is_finite() && t >= 0.0 => t,
                Some(t) => {
                    return Err(format!(
                        "--weld-tolerance must be finite and not negative, not {t}"
                    ));
                }
            };
            let self_intersection = parsed.flag("--self-intersection");
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Analyze {
                input: input(path)?,
                weld_tolerance,
                self_intersection,
                report,
            })
        }
        other => Err(format!(
            "unknown inspect subcommand \"{other}\" (expected measure or analyze)"
        )),
    }
}

fn parse_remesh(mut args: impl Iterator<Item = OsString>) -> Result<Command, String> {
    let sub = args
        .next()
        .ok_or("remesh requires a subcommand: simplify or subdivide")?;
    let sub_name = sub.to_str().unwrap_or("");
    match sub_name {
        "-h" | "--help" => Ok(Command::Help(REMESH_HELP)),
        "simplify" => {
            const SIMPLIFY_SPEC: [(&str, bool); 7] = [
                OUTPUT[0],
                OUTPUT[1],
                OUTPUT[2],
                FORMAT,
                ("--triangles", true),
                ("--ratio", true),
                ("--lock-boundary", false),
            ];
            let Some(mut parsed) = split(args, &SIMPLIFY_SPEC)? else {
                return Ok(Command::Help(REMESH_HELP));
            };
            let report = report_format(&mut parsed)?;
            let triangles = number::<usize>(&mut parsed, "--triangles")?;
            let ratio = number::<f64>(&mut parsed, "--ratio")?;
            let target = match (triangles, ratio) {
                (Some(_), Some(_)) => {
                    return Err("simplify takes --triangles or --ratio, not both".into());
                }
                (None, None) => {
                    return Err("simplify requires --triangles <n> or --ratio <r>".into());
                }
                (Some(0), None) => {
                    return Err("--triangles must be at least 1".into());
                }
                (Some(n), None) => SimplifyTarget::Triangles(n),
                (None, Some(r)) => {
                    if !r.is_finite() || r <= 0.0 || r > 1.0 {
                        return Err(format!("--ratio must be in (0, 1], not {r}"));
                    }
                    SimplifyTarget::Ratio(r)
                }
            };
            let preserve_boundary = parsed.flag("--lock-boundary");
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Simplify {
                input: input(path)?,
                target,
                preserve_boundary,
                output: output(&mut parsed)?,
                report,
            })
        }
        "subdivide" => {
            const SUBDIVIDE_SPEC: [(&str, bool); 5] =
                [OUTPUT[0], OUTPUT[1], OUTPUT[2], FORMAT, ("--levels", true)];
            let Some(mut parsed) = split(args, &SUBDIVIDE_SPEC)? else {
                return Ok(Command::Help(REMESH_HELP));
            };
            let report = report_format(&mut parsed)?;
            let levels = match number::<u32>(&mut parsed, "--levels")? {
                Some(0) => return Err("--levels must be at least 1".into()),
                Some(n) => n,
                None => 1,
            };
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Subdivide {
                input: input(path)?,
                levels,
                output: output(&mut parsed)?,
                report,
            })
        }
        other => Err(format!(
            "unknown remesh subcommand \"{other}\" (expected simplify or subdivide)"
        )),
    }
}
