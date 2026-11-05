//! Mesh file formats, selected by file extension: STL (binary and ASCII), OBJ,
//! PLY (ASCII and binary little-endian) and binary glTF (`.glb`), as input and
//! output.
//!
//! Parsers take the whole file as bytes and return an error, never a panic, on
//! anything malformed or truncated. A count in a file is never trusted: it is
//! checked against the bytes actually there before anything is allocated from
//! it, so a small file cannot ask for a huge allocation.

mod glb;
mod json;
mod obj;
mod ply;
mod stl;

use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// An indexed triangle mesh in file coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshData {
    /// `[x, y, z, ...]`.
    pub positions: Vec<f64>,
    /// Three vertex ids per triangle.
    pub indices: Vec<u32>,
}

/// A file could not be read, parsed or written; the message says why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    /// An error with this message.
    pub fn new(message: impl Into<String>) -> Self {
        Error(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

/// A file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `.stl`: binary or ASCII.
    Stl,
    /// `.obj`: Wavefront, always text.
    Obj,
    /// `.ply`: ASCII or binary little-endian.
    Ply,
    /// `.glb`: binary glTF 2.0, always binary.
    Glb,
}

impl Format {
    /// The format a path's extension names, in any case.
    ///
    /// # Errors
    /// When the extension is missing or is not a supported format.
    pub fn from_path(path: &Path) -> Result<Format> {
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase());
        match extension.as_deref() {
            Some("stl") => Ok(Format::Stl),
            Some("obj") => Ok(Format::Obj),
            Some("ply") => Ok(Format::Ply),
            Some("glb") => Ok(Format::Glb),
            Some("gltf") => Err(Error::new(format!(
                "{}: JSON glTF (.gltf) is not supported, use binary glTF (.glb)",
                path.display()
            ))),
            _ => Err(Error::new(format!(
                "{}: unknown file format (expected .stl, .obj, .ply or .glb)",
                path.display()
            ))),
        }
    }

    /// Whether the format has both a binary and an ASCII form.
    pub fn has_ascii_form(self) -> bool {
        matches!(self, Format::Stl | Format::Ply)
    }
}

/// Parses a whole file. The result has at least one triangle, finite
/// positions and every index below the vertex count.
///
/// # Errors
/// When the bytes are not a valid file of `format`, or memory runs out.
pub fn parse(format: Format, bytes: &[u8]) -> Result<MeshData> {
    match format {
        Format::Stl => stl::parse(bytes),
        Format::Obj => obj::parse(bytes),
        Format::Ply => ply::parse(bytes),
        Format::Glb => glb::parse(bytes),
    }
}

/// Writes `data` as a whole file, dropping vertices no triangle uses. `ascii`
/// picks the text form of a format that has both ([`Format::has_ascii_form`])
/// and is ignored by the others. STL, binary PLY and GLB hold 32-bit floats;
/// OBJ and ASCII PLY keep the full double precision.
///
/// # Errors
/// When `data` has no triangles or an index past its vertices, when a
/// coordinate or count does not fit the format, or when writing fails.
pub fn write_mesh<W: Write>(
    data: &MeshData,
    format: Format,
    ascii: bool,
    out: &mut W,
) -> Result<()> {
    let data = compacted(data)?;
    match format {
        Format::Stl => stl::write(&data, ascii, out),
        Format::Obj => obj::write(&data, out),
        Format::Ply => ply::write(&data, ascii, out),
        Format::Glb => glb::write(&data, out),
    }
}

/// `data` with only the vertices its triangles use, in their original order.
fn compacted(data: &MeshData) -> Result<MeshData> {
    if data.indices.is_empty() {
        return Err(Error::new("the mesh has no triangles"));
    }
    let vertices = data.positions.len() / 3;
    const UNUSED: u32 = u32::MAX;
    let mut map = try_vec(vertices, UNUSED)?;
    for &i in &data.indices {
        let Some(slot) = map.get_mut(i as usize) else {
            return Err(Error::new(format!(
                "vertex {i} is past the {vertices} vertices"
            )));
        };
        *slot = 0;
    }
    let mut positions = with_capacity(data.positions.len())?;
    let mut next = 0u32;
    for (v, slot) in map.iter_mut().enumerate() {
        if *slot != UNUSED {
            *slot = next;
            next += 1;
            positions.extend_from_slice(&data.positions[3 * v..3 * v + 3]);
        }
    }
    let mut indices = with_capacity(data.indices.len())?;
    indices.extend(data.indices.iter().map(|&i| map[i as usize]));
    Ok(MeshData { positions, indices })
}

/// Writes a file so that it appears whole or not at all: into a temporary file
/// beside it, then renamed over `path` (through a symbolic link, to its
/// target). A failure removes the temporary file and leaves `path` as it was.
///
/// # Errors
/// When `write` fails or the file system refuses.
pub fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    let target = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let describe = |e: std::io::Error| Error::new(format!("{}: {e}", path.display()));
    let name = target
        .file_name()
        .ok_or_else(|| Error::new(format!("{}: not a file name", path.display())))?;
    let dir = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let mut temporary = OsString::from(".");
    temporary.push(name);
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = dir.join(temporary);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(describe)?;
    let finish = || -> Result<()> {
        let mut out = BufWriter::with_capacity(1 << 20, file);
        write(&mut out)?;
        let file = out.into_inner().map_err(|e| describe(e.into_error()))?;
        file.sync_all().map_err(describe)?;
        drop(file);
        fs::rename(&temporary, &target).map_err(describe)
    };
    let result = finish();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn out_of_memory() -> Error {
    Error::new("out of memory")
}

/// An empty vector with room for `n`, or an error instead of an abort.
fn with_capacity<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| out_of_memory())?;
    Ok(v)
}

/// `n` copies of `value`, or an error instead of an abort.
fn try_vec<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = with_capacity(n)?;
    v.resize(n, value);
    Ok(v)
}

/// Appends, or an error instead of an abort.
fn push_all<T: Copy>(v: &mut Vec<T>, values: &[T]) -> Result<()> {
    v.try_reserve(values.len()).map_err(|_| out_of_memory())?;
    v.extend_from_slice(values);
    Ok(())
}

/// The next whitespace-separated token from `*at`, moving past it.
fn next_token<'a>(bytes: &'a [u8], at: &mut usize) -> Option<&'a [u8]> {
    let start = *at + bytes[*at..].iter().position(|b| !b.is_ascii_whitespace())?;
    let len = bytes[start..]
        .iter()
        .position(|b| b.is_ascii_whitespace())
        .unwrap_or(bytes.len() - start);
    *at = start + len;
    Some(&bytes[start..start + len])
}

/// A token as a finite number.
fn finite_number(token: &[u8]) -> Option<f64> {
    let value: f64 = std::str::from_utf8(token).ok()?.parse().ok()?;
    value.is_finite().then_some(value)
}

/// Some of a token, safe to put in a message.
fn shown(token: &[u8]) -> String {
    let text = String::from_utf8_lossy(&token[..token.len().min(40)]);
    let cut = if token.len() > 40 { "..." } else { "" };
    format!("\"{}{cut}\"", text.escape_debug())
}

/// A coordinate as the 32-bit float a binary format holds.
fn to_f32(value: f64, format: &str) -> Result<f32> {
    let single = value as f32;
    if single.is_finite() {
        Ok(single)
    } else {
        Err(Error::new(format!(
            "{format}: coordinate {value:e} does not fit its 32-bit floats"
        )))
    }
}

/// Refuses a mesh a parser must not return.
fn check_mesh(data: MeshData, format: &str) -> Result<MeshData> {
    if data.indices.is_empty() {
        return Err(Error::new(format!("{format}: the file has no faces")));
    }
    let vertices = data.positions.len() / 3;
    if vertices > u32::MAX as usize {
        return Err(Error::new(format!(
            "{format}: {vertices} vertices is too many for 32-bit indices"
        )));
    }
    if let Some(i) = data.indices.iter().find(|&&i| i as usize >= vertices) {
        return Err(Error::new(format!(
            "{format}: face index {i} is past the {vertices} vertices"
        )));
    }
    if !data.positions.iter().all(|x| x.is_finite()) {
        return Err(Error::new(format!(
            "{format}: a vertex coordinate is not a finite number"
        )));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `name` in a new directory of its own; `remove` deletes it at the end.
    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fast-mesh-cli-{}-{name}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn remove(path: &Path) {
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn atomic_write_keeps_the_old_file_when_the_writer_fails_midway() {
        let path = temp_path("midway.stl");
        fs::write(&path, b"old").unwrap();
        let result = atomic_write(&path, |out| {
            out.write_all(&[7u8; 4 << 20])?;
            Err(Error::new("disk went away"))
        });
        assert_eq!(result.unwrap_err().to_string(), "disk went away");
        assert_eq!(fs::read(&path).unwrap(), b"old");
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("midway"))
            .collect();
        assert_eq!(names, ["midway.stl"], "no temporary file left");
        remove(&path);
    }

    #[test]
    fn atomic_write_replaces() {
        let path = temp_path("replace.stl");
        fs::write(&path, b"old").unwrap();
        atomic_write(&path, |out| Ok(out.write_all(b"new")?)).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        remove(&path);
    }

    #[test]
    fn format_from_path() {
        assert_eq!(
            Format::from_path(Path::new("a/b.STL")).unwrap(),
            Format::Stl
        );
        assert_eq!(Format::from_path(Path::new("b.Glb")).unwrap(), Format::Glb);
        assert!(Format::from_path(Path::new("b")).is_err());
        let message = Format::from_path(Path::new("b.gltf")).unwrap_err();
        assert!(message.to_string().contains(".glb"));
    }

    #[test]
    fn compaction_keeps_vertex_order() {
        let data = MeshData {
            positions: vec![
                9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            indices: vec![4, 1, 3],
        };
        let out = compacted(&data).unwrap();
        assert_eq!(out.positions, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(out.indices, [2, 0, 1]);
    }

    #[test]
    fn tokens() {
        let mut at = 0;
        let bytes = b"  ab\tc\n\n";
        assert_eq!(next_token(bytes, &mut at), Some(&b"ab"[..]));
        assert_eq!(next_token(bytes, &mut at), Some(&b"c"[..]));
        assert_eq!(next_token(bytes, &mut at), None);
        assert_eq!(next_token(b"", &mut 0), None);
    }
}
