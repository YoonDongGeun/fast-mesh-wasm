//! The command line: a small hand-written parser, so that the binary adds no
//! dependency.

use std::ffi::OsString;
use std::path::PathBuf;

use fast_mesh_cli::io::Format;
use fast_mesh_ops::{AutoRepairOptions, BooleanOp};

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
    Boolean {
        a: Input,
        b: Input,
        op: BooleanOp,
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
fast-mesh: mesh repair, measurement, booleans and format conversion

Usage:
  fast-mesh repair <in> -o <out> --auto [repair options] [output options]
  fast-mesh measure <in> [--format text|json]
  fast-mesh boolean <a> <b> --op union|intersection|difference -o <out> [output options]
  fast-mesh convert <in> -o <out> [output options]

Formats, chosen by file extension: .stl .obj .ply .glb

Run `fast-mesh <command> --help` for a command's options.

Exit codes:
  0  success
  1  the operation or a file failed (bad input, I/O, empty result); nothing is written
  2  usage error (bad arguments, or an output that would overwrite an input)

Reports go to stdout, diagnostics to stderr. With --format json, stdout holds
one JSON object on success and nothing on failure.
";

const REPAIR_HELP: &str = "\
fast-mesh repair <in> -o <out> --auto [options]

Runs the standard cleanup: weld, remove degenerate triangles, orient, remove
small components, fill holes.

  --auto                  Run the whole pipeline (required)
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

const MEASURE_HELP: &str = "\
fast-mesh measure <in> [options]

Reports vertex and triangle counts, area, volume (when watertight) and defects.

  --format text|json      The report's format (default text)
";

const BOOLEAN_HELP: &str = "\
fast-mesh boolean <a> <b> --op <op> -o <out> [options]

Combines two closed, manifold, consistently wound meshes.

  --op union|intersection|difference
                          difference is a minus b
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

/// Parses the arguments after the program name. An `Err` is a usage error.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let first = args.next().ok_or("no command given")?;
    let command = first.to_str().unwrap_or("");
    match command {
        "-h" | "--help" | "help" => return Ok(Command::Help(HELP)),
        "-V" | "--version" => return Ok(Command::Version),
        _ => {}
    }
    let (spec, help): (&[(&'static str, bool)], &'static str) = match command {
        "repair" => (
            &[
                OUTPUT[0],
                OUTPUT[1],
                OUTPUT[2],
                FORMAT,
                ("--auto", false),
                ("--weld-tolerance", true),
                ("--min-triangles", true),
                ("--max-edges", true),
                ("--no-fair", false),
            ],
            REPAIR_HELP,
        ),
        "measure" => (&[FORMAT], MEASURE_HELP),
        "boolean" => (
            &[OUTPUT[0], OUTPUT[1], OUTPUT[2], FORMAT, ("--op", true)],
            BOOLEAN_HELP,
        ),
        "convert" => (&[OUTPUT[0], OUTPUT[1], OUTPUT[2], FORMAT], CONVERT_HELP),
        other => return Err(format!("unknown command \"{other}\"")),
    };
    let Some(mut parsed) = split(args, spec)? else {
        return Ok(Command::Help(help));
    };
    let report = report_format(&mut parsed)?;
    match command {
        "repair" => {
            let [path] = positionals(&mut parsed, "one input file")?;
            if !parsed.flag("--auto") {
                return Err("repair needs --auto (the whole pipeline)".into());
            }
            Ok(Command::Repair {
                options: repair_options(&mut parsed)?,
                input: input(path)?,
                output: output(&mut parsed)?,
                report,
            })
        }
        "measure" => {
            let [path] = positionals(&mut parsed, "one input file")?;
            Ok(Command::Measure {
                input: input(path)?,
                report,
            })
        }
        "boolean" => {
            let [a, b] = positionals(&mut parsed, "two input files")?;
            let op = match parsed.text("--op")?.as_deref() {
                Some("union") => BooleanOp::Union,
                Some("intersection") => BooleanOp::Intersect,
                Some("difference") => BooleanOp::Subtract,
                Some(other) => {
                    return Err(format!(
                        "--op must be union, intersection or difference, not {other}"
                    ));
                }
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
