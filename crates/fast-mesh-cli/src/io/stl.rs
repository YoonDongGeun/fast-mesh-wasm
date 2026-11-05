//! STL, binary and ASCII. STL is triangle soup: corners with bit-identical
//! positions are merged into one vertex (near-duplicates are left to the weld
//! repair).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::io::Write;

use super::{
    Error, MeshData, Result, check_mesh, finite_number, next_token, out_of_memory, push_all, shown,
    to_f32, try_vec, with_capacity,
};

const HEADER: usize = 80;
const PREAMBLE: u64 = HEADER as u64 + 4;
const RECORD: u64 = 50;

/// Binary when the size matches the triangle count in the header (even if the
/// header text starts with "solid", as some exporters write), else ASCII when
/// it starts with "solid".
pub(super) fn parse(bytes: &[u8]) -> Result<MeshData> {
    let count = bytes
        .get(HEADER..HEADER + 4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    if let Some(count) = count
        && count > 0
        && bytes.len() as u64 == PREAMBLE + RECORD * u64::from(count)
    {
        return check_mesh(parse_binary(bytes, count as usize)?, "STL");
    }
    if bytes.trim_ascii_start().starts_with(b"solid") {
        return check_mesh(parse_ascii(bytes)?, "STL");
    }
    if let Some(count) = count
        && count > 0
    {
        return Err(Error::new(format!(
            "binary STL: the header says {count} triangles ({} bytes), but the file has {} bytes",
            PREAMBLE + RECORD * u64::from(count),
            bytes.len()
        )));
    }
    Err(Error::new("not an STL file (neither binary nor ASCII STL)"))
}

fn parse_binary(bytes: &[u8], count: usize) -> Result<MeshData> {
    let mut corners = with_capacity(9 * count)?;
    for record in bytes[PREAMBLE as usize..]
        .as_chunks::<{ RECORD as usize }>()
        .0
    {
        // Normal (12 bytes), three corners (36), attribute (2).
        for c in record[12..48].as_chunks::<4>().0 {
            let value = f64::from(f32::from_le_bytes(*c));
            if !value.is_finite() {
                return Err(Error::new(
                    "STL: a vertex coordinate is not a finite number",
                ));
            }
            corners.push(value);
        }
    }
    weld_corners(&corners)
}

/// The `solid` and `endsolid` lines hold free text (a solid may well be named
/// "vertex") and are skipped. A solid left open means the file was cut off.
fn parse_ascii(bytes: &[u8]) -> Result<MeshData> {
    let mut corners = Vec::new();
    let mut open = false;
    let mut at = 0;
    while let Some(token) = next_token(bytes, &mut at) {
        match token {
            b"solid" | b"endsolid" => {
                open = token == b"solid";
                at += bytes[at..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .unwrap_or(bytes.len() - at);
            }
            b"vertex" => {
                let mut corner = [0.0; 3];
                for c in &mut corner {
                    let value = next_token(bytes, &mut at)
                        .ok_or_else(|| Error::new("ASCII STL: the file ends inside a vertex"))?;
                    *c = finite_number(value).ok_or_else(|| {
                        Error::new(format!(
                            "ASCII STL: vertex coordinate {} is not a finite number",
                            shown(value)
                        ))
                    })?;
                }
                push_all(&mut corners, &corner)?;
            }
            _ => {}
        }
    }
    if corners.is_empty() {
        return Err(Error::new("STL: the file has no faces"));
    }
    if !corners.len().is_multiple_of(9) {
        return Err(Error::new(format!(
            "ASCII STL: {} vertices, not 3 per facet",
            corners.len() / 3
        )));
    }
    if open {
        return Err(Error::new("ASCII STL: no endsolid, the file is cut off"));
    }
    weld_corners(&corners)
}

/// A triangle soup as an indexed mesh: corners whose positions are
/// bit-identical share a vertex, after -0 is made +0. `HashMap`'s keyed hash
/// keeps this linear on crafted files too.
fn weld_corners(corners: &[f64]) -> Result<MeshData> {
    let n = corners.len() / 3;
    if n > u32::MAX as usize {
        return Err(Error::new(format!(
            "STL: {n} corners is too many for 32-bit indices"
        )));
    }
    let mut ids: HashMap<[u64; 3], u32> = HashMap::new();
    ids.try_reserve(n / 4).map_err(|_| out_of_memory())?;
    let mut positions = Vec::new();
    let mut indices = try_vec(n, 0u32)?;
    for (corner, index) in corners.as_chunks::<3>().0.iter().zip(&mut indices) {
        // Adding 0.0 turns -0.0 into +0.0 and changes nothing else.
        let key = [0, 1, 2].map(|k| (corner[k] + 0.0).to_bits());
        ids.try_reserve(1).map_err(|_| out_of_memory())?;
        *index = match ids.entry(key) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let id = (positions.len() / 3) as u32;
                push_all(
                    &mut positions,
                    &[corner[0] + 0.0, corner[1] + 0.0, corner[2] + 0.0],
                )?;
                *e.insert(id)
            }
        };
    }
    Ok(MeshData { positions, indices })
}

pub(super) fn write<W: Write>(data: &MeshData, ascii: bool, out: &mut W) -> Result<()> {
    if ascii {
        write_ascii(data, out)
    } else {
        write_binary(data, out)
    }
}

fn corner(data: &MeshData, v: u32) -> [f64; 3] {
    let at = 3 * v as usize;
    [
        data.positions[at],
        data.positions[at + 1],
        data.positions[at + 2],
    ]
}

/// The unit normal of a triangle, zero when it has no area.
fn normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 && len.is_finite() {
        n.map(|x| x / len)
    } else {
        [0.0; 3]
    }
}

fn write_binary<W: Write>(data: &MeshData, out: &mut W) -> Result<()> {
    let triangles = u32::try_from(data.indices.len() / 3)
        .map_err(|_| Error::new("STL: too many triangles for its 32-bit count"))?;
    let mut header = [0u8; HEADER];
    let text = b"exported by fast-mesh-cli";
    header[..text.len()].copy_from_slice(text);
    out.write_all(&header)?;
    out.write_all(&triangles.to_le_bytes())?;
    let mut record = [0u8; RECORD as usize];
    for t in data.indices.as_chunks::<3>().0 {
        let (a, b, c) = (corner(data, t[0]), corner(data, t[1]), corner(data, t[2]));
        let values = normal(a, b, c).into_iter().chain(a).chain(b).chain(c);
        for (slot, value) in record.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *slot = to_f32(value, "STL")?.to_le_bytes();
        }
        out.write_all(&record)?;
    }
    Ok(())
}

fn write_ascii<W: Write>(data: &MeshData, out: &mut W) -> Result<()> {
    writeln!(out, "solid mesh")?;
    for t in data.indices.as_chunks::<3>().0 {
        let (a, b, c) = (corner(data, t[0]), corner(data, t[1]), corner(data, t[2]));
        let n = normal(a, b, c);
        writeln!(out, " facet normal {} {} {}", n[0], n[1], n[2])?;
        writeln!(out, "  outer loop")?;
        for p in [a, b, c] {
            writeln!(out, "   vertex {} {} {}", p[0], p[1], p[2])?;
        }
        writeln!(out, "  endloop")?;
        writeln!(out, " endfacet")?;
    }
    writeln!(out, "endsolid mesh")?;
    Ok(())
}
