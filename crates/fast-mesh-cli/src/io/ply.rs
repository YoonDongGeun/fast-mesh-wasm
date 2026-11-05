//! PLY, ASCII and binary little-endian: the `vertex` element's x, y, z and the
//! `face` element's index list (`vertex_indices` or `vertex_index`, any
//! polygon, fan-triangulated). Other properties and elements are skipped.

use std::io::Write;

use super::{
    Error, MeshData, Result, check_mesh, finite_number, next_token, push_all, shown, to_f32,
    try_vec,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Scalar> {
        Some(match name {
            "char" | "int8" => Scalar::I8,
            "uchar" | "uint8" => Scalar::U8,
            "short" | "int16" => Scalar::I16,
            "ushort" | "uint16" => Scalar::U16,
            "int" | "int32" => Scalar::I32,
            "uint" | "uint32" => Scalar::U32,
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }
}

struct Property {
    name: String,
    ty: Scalar,
    /// The type of the list's length; `None` for a plain scalar.
    count: Option<Scalar>,
}

struct Element {
    name: String,
    size: u64,
    properties: Vec<Property>,
}

struct Header {
    ascii: bool,
    elements: Vec<Element>,
    /// The byte offset of the body.
    body: usize,
}

pub(super) fn parse(bytes: &[u8]) -> Result<MeshData> {
    let header = read_header(bytes)?;
    let vertex_at = header.elements.iter().position(|e| e.name == "vertex");
    let face_at = header.elements.iter().position(|e| e.name == "face");
    let (Some(vertex_at), Some(face_at)) = (vertex_at, face_at) else {
        return Err(Error::new("PLY: needs a vertex and a face element"));
    };
    let vertex = &header.elements[vertex_at];
    let face = &header.elements[face_at];
    let axis = |name: &str| {
        vertex
            .properties
            .iter()
            .position(|p| p.name == name && p.count.is_none())
    };
    let (Some(x), Some(y), Some(z)) = (axis("x"), axis("y"), axis("z")) else {
        return Err(Error::new("PLY: the vertex element has no x, y, z"));
    };
    let list = face
        .properties
        .iter()
        .position(|p| p.count.is_some() && (p.name == "vertex_indices" || p.name == "vertex_index"))
        .or_else(|| face.properties.iter().position(|p| p.count.is_some()))
        .ok_or_else(|| Error::new("PLY: the face element has no vertex index list"))?;

    check_counts(&header, bytes.len() - header.body)?;
    if vertex.size > u64::from(u32::MAX) {
        return Err(Error::new(format!(
            "PLY: {} vertices is too many for 32-bit indices",
            vertex.size
        )));
    }

    let vertices = vertex.size;
    let mut positions = try_vec(3 * vertices as usize, 0.0f64)?;
    let mut indices: Vec<u32> = Vec::new();
    let mut corners: Vec<u32> = Vec::new();
    let mut reader = Reader {
        bytes,
        at: header.body,
        ascii: header.ascii,
    };
    for (e, element) in header.elements.iter().enumerate() {
        // No property, nothing to read: a huge size must not spin here.
        if element.properties.is_empty() {
            continue;
        }
        for i in 0..element.size as usize {
            for (p, property) in element.properties.iter().enumerate() {
                let Some(count_type) = property.count else {
                    let value = reader.read(property.ty)?;
                    if e == vertex_at {
                        if p == x {
                            positions[3 * i] = value;
                        } else if p == y {
                            positions[3 * i + 1] = value;
                        } else if p == z {
                            positions[3 * i + 2] = value;
                        }
                    }
                    continue;
                };
                let n = reader.read(count_type)?;
                if n < 0.0 || n.fract() != 0.0 {
                    return Err(Error::new(format!("PLY: bad list length {n}")));
                }
                let n = n as u64;
                if e == face_at && p == list {
                    corners.clear();
                    for _ in 0..n {
                        let index = reader.read(property.ty)?;
                        if index < 0.0 || index >= vertices as f64 || index.fract() != 0.0 {
                            return Err(Error::new(format!(
                                "PLY: face index {index} is out of range ({vertices} vertices)"
                            )));
                        }
                        push_all(&mut corners, &[index as u32])?;
                    }
                    for k in 1..corners.len().saturating_sub(1) {
                        push_all(&mut indices, &[corners[0], corners[k], corners[k + 1]])?;
                    }
                } else {
                    for _ in 0..n {
                        reader.read(property.ty)?;
                    }
                }
            }
        }
    }
    check_mesh(MeshData { positions, indices }, "PLY")
}

/// Refuses counts the body cannot hold before anything is allocated from them.
/// A plain binary property takes its size, a list at least its length field,
/// and an ASCII value at least two bytes (one digit, one separator; the last
/// may lack its separator).
fn check_counts(header: &Header, body: usize) -> Result<()> {
    let mut needed: u128 = 0;
    for element in &header.elements {
        let each: u128 = element
            .properties
            .iter()
            .map(|p| {
                if header.ascii {
                    2
                } else {
                    p.count.unwrap_or(p.ty).size() as u128
                }
            })
            .sum();
        needed += each * u128::from(element.size);
    }
    let room = body as u128 + u128::from(header.ascii);
    if needed > room {
        return Err(Error::new(format!(
            "PLY: the header's element counts need at least {needed} bytes, but the file ends early with {body}"
        )));
    }
    Ok(())
}

fn read_header(bytes: &[u8]) -> Result<Header> {
    let magic_ok = bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r");
    if !magic_ok {
        return Err(Error::new("not a PLY file"));
    }
    let end = header_end(bytes).ok_or_else(|| Error::new("PLY: no end_header"))?;
    let mut body = end + b"end_header".len();
    if bytes.get(body) == Some(&b'\r') {
        body += 1;
    }
    if bytes.get(body) == Some(&b'\n') {
        body += 1;
    }
    let text = String::from_utf8_lossy(&bytes[..end]);
    let mut ascii = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in text.lines() {
        let bad = || Error::new(format!("PLY: bad header line {}", shown(line.as_bytes())));
        let words: Vec<&str> = line.split_ascii_whitespace().collect();
        match words.as_slice() {
            ["format", "binary_big_endian", ..] => {
                return Err(Error::new("PLY: big-endian binary PLY is not supported"));
            }
            ["format", "ascii", ..] => ascii = Some(true),
            ["format", "binary_little_endian", ..] => ascii = Some(false),
            ["format", ..] => return Err(bad()),
            ["element", name, size, ..] => {
                let size = size.parse().map_err(|_| bad())?;
                elements.push(Element {
                    name: (*name).to_string(),
                    size,
                    properties: Vec::new(),
                });
            }
            ["element", ..] => return Err(bad()),
            ["property", rest @ ..] => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| Error::new("PLY: a property before any element"))?;
                element.properties.push(match rest {
                    ["list", count, ty, name, ..] => {
                        let count = Scalar::parse(count)
                            .filter(|c| !matches!(c, Scalar::F32 | Scalar::F64))
                            .ok_or_else(bad)?;
                        Property {
                            name: (*name).to_string(),
                            ty: Scalar::parse(ty).ok_or_else(bad)?,
                            count: Some(count),
                        }
                    }
                    [ty, name, ..] => Property {
                        name: (*name).to_string(),
                        ty: Scalar::parse(ty).ok_or_else(bad)?,
                        count: None,
                    },
                    _ => return Err(bad()),
                });
            }
            _ => {}
        }
    }
    let ascii = ascii.ok_or_else(|| Error::new("PLY: no format line"))?;
    Ok(Header {
        ascii,
        elements,
        body,
    })
}

/// Where the line "end_header" starts (after a line end, followed by one or
/// the file's end): the word inside a comment does not count.
fn header_end(bytes: &[u8]) -> Option<usize> {
    const NEEDLE: &[u8] = b"end_header";
    let mut at = 0;
    loop {
        at += bytes[at..].iter().position(|&b| b == b'\n')? + 1;
        let rest = &bytes[at..];
        if rest.starts_with(NEEDLE)
            && matches!(
                rest.get(NEEDLE.len()),
                None | Some(b'\n' | b'\r' | b' ' | b'\t')
            )
        {
            return Some(at);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    ascii: bool,
}

impl Reader<'_> {
    /// The next value, widened to `f64` (exact for every scalar type). A
    /// non-finite value is refused here for ASCII and by the final check for
    /// binary.
    fn read(&mut self, ty: Scalar) -> Result<f64> {
        if self.ascii {
            let token = next_token(self.bytes, &mut self.at)
                .ok_or_else(|| Error::new("PLY: the file ends early"))?;
            return finite_number(token).ok_or_else(|| {
                Error::new(format!("PLY: {} is not a finite number", shown(token)))
            });
        }
        let size = ty.size();
        let raw = self
            .bytes
            .get(self.at..self.at + size)
            .ok_or_else(|| Error::new("PLY: the file ends early"))?;
        self.at += size;
        let mut b = [0u8; 8];
        b[..size].copy_from_slice(raw);
        Ok(match ty {
            Scalar::I8 => f64::from(b[0] as i8),
            Scalar::U8 => f64::from(b[0]),
            Scalar::I16 => f64::from(i16::from_le_bytes([b[0], b[1]])),
            Scalar::U16 => f64::from(u16::from_le_bytes([b[0], b[1]])),
            Scalar::I32 => f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Scalar::U32 => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Scalar::F32 => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            Scalar::F64 => f64::from_le_bytes(b),
        })
    }
}

pub(super) fn write<W: Write>(data: &MeshData, ascii: bool, out: &mut W) -> Result<()> {
    let vertices = data.positions.len() / 3;
    let triangles = data.indices.len() / 3;
    let (format, scalar) = if ascii {
        ("ascii", "double")
    } else {
        ("binary_little_endian", "float")
    };
    write!(
        out,
        "ply\nformat {format} 1.0\nelement vertex {vertices}\nproperty {scalar} x\nproperty {scalar} y\nproperty {scalar} z\nelement face {triangles}\nproperty list uchar uint vertex_indices\nend_header\n"
    )?;
    if ascii {
        for p in data.positions.as_chunks::<3>().0 {
            writeln!(out, "{} {} {}", p[0], p[1], p[2])?;
        }
        for t in data.indices.as_chunks::<3>().0 {
            writeln!(out, "3 {} {} {}", t[0], t[1], t[2])?;
        }
        return Ok(());
    }
    for &value in &data.positions {
        out.write_all(&to_f32(value, "PLY")?.to_le_bytes())?;
    }
    for t in data.indices.as_chunks::<3>().0 {
        let mut record = [3u8; 13];
        for (slot, v) in record[1..].as_chunks_mut::<4>().0.iter_mut().zip(t) {
            *slot = v.to_le_bytes();
        }
        out.write_all(&record)?;
    }
    Ok(())
}
