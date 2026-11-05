//! Wavefront OBJ: `v` positions and `f` faces (any polygon, fan-triangulated;
//! `i`, `i/t`, `i//n` and `i/t/n` corners; negative indices count back from
//! the latest vertex). Everything else (normals, texture coordinates, groups,
//! materials) is ignored.

use std::io::Write;

use super::{Error, MeshData, Result, check_mesh, finite_number, push_all, shown};

pub(super) fn parse(bytes: &[u8]) -> Result<MeshData> {
    let mut positions: Vec<f64> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut corners: Vec<u32> = Vec::new();
    for (number, line) in bytes.split(|&b| b == b'\n').enumerate() {
        let line = line.trim_ascii();
        let number = number + 1;
        match line.first_chunk::<2>() {
            Some(b"v " | b"v\t") => {
                let mut fields = words(&line[2..]);
                let mut coordinate = || fields.next().and_then(finite_number);
                match (coordinate(), coordinate(), coordinate()) {
                    (Some(x), Some(y), Some(z)) => push_all(&mut positions, &[x, y, z])?,
                    _ => {
                        return Err(Error::new(format!(
                            "OBJ line {number}: bad vertex {}",
                            shown(line)
                        )));
                    }
                }
            }
            Some(b"f " | b"f\t") => {
                corners.clear();
                let vertices = (positions.len() / 3) as i64;
                for word in words(&line[2..]) {
                    push_all(&mut corners, &[corner(word, vertices, number)?])?;
                }
                if corners.len() < 3 {
                    return Err(Error::new(format!(
                        "OBJ line {number}: a face needs 3 corners"
                    )));
                }
                for k in 1..corners.len() - 1 {
                    push_all(&mut indices, &[corners[0], corners[k], corners[k + 1]])?;
                }
            }
            _ => {}
        }
    }
    check_mesh(MeshData { positions, indices }, "OBJ")
}

fn words(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(u8::is_ascii_whitespace)
        .filter(|w| !w.is_empty())
}

/// The vertex id of a face corner. An id past the vertices so far may still
/// come true: the end of the parse checks it.
fn corner(word: &[u8], vertices: i64, line: usize) -> Result<u32> {
    let id = word.split(|&b| b == b'/').next().unwrap_or(word);
    let raw: i64 = std::str::from_utf8(id)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&raw| raw != 0)
        .ok_or_else(|| Error::new(format!("OBJ line {line}: bad face index {}", shown(word))))?;
    let resolved = if raw < 0 {
        vertices.saturating_add(raw)
    } else {
        raw - 1
    };
    if resolved < 0 {
        return Err(Error::new(format!(
            "OBJ line {line}: face index {raw} points before the first vertex"
        )));
    }
    u32::try_from(resolved).map_err(|_| {
        Error::new(format!(
            "OBJ line {line}: bad face index {} (too large)",
            shown(word)
        ))
    })
}

pub(super) fn write<W: Write>(data: &MeshData, out: &mut W) -> Result<()> {
    writeln!(out, "o mesh")?;
    for p in data.positions.as_chunks::<3>().0 {
        writeln!(out, "v {} {} {}", p[0], p[1], p[2])?;
    }
    for t in data.indices.as_chunks::<3>().0 {
        writeln!(
            out,
            "f {} {} {}",
            u64::from(t[0]) + 1,
            u64::from(t[1]) + 1,
            u64::from(t[2]) + 1
        )?;
    }
    Ok(())
}
