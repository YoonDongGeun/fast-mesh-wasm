//! Binary glTF 2.0 with uncompressed geometry in its body chunk. Every
//! (node, primitive) pair joins one mesh, node placements baked to world
//! coordinates. Materials, UVs, normals, colors, skins and morph targets are
//! ignored.
//!
//! What cannot be read faithfully is refused, not approximated: Draco and
//! meshopt compression, GPU instancing, sparse accessors, buffers behind a
//! URI, and accessors without a buffer view (their zeros would otherwise let
//! a small file claim any count).

use std::io::Write;

use super::json::{self, Value};
use super::{Error, MeshData, Result, check_mesh, push_all, to_f32, try_vec, with_capacity};

const MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;
const FLOAT: u64 = 5126;
const UNSIGNED_BYTE: u64 = 5121;
const UNSIGNED_SHORT: u64 = 5123;
const UNSIGNED_INT: u64 = 5125;

/// How many vertices and corners instancing may expand a file to, at least:
/// one node can reuse a primitive any number of times, so a small file could
/// otherwise claim any total. A file holding more itself is still read whole.
const MIN_EXPANSION_LIMIT: u64 = 1 << 28;

/// How many (node, primitive) pairs a file may hold, at least: each costs a
/// fresh read of its accessors, even with no vertices, so reuse in a small file
/// could otherwise cost nodes x primitives. See [`visit_limit`].
const MIN_VISIT_LIMIT: u64 = 1 << 20;

/// The most (node, primitive) pairs for a file of `len` bytes. A pair without
/// reuse takes well over 16 bytes of JSON, so such a file never reaches this.
fn visit_limit(len: usize) -> u64 {
    MIN_VISIT_LIMIT.max(len as u64 / 16)
}

fn bad(what: impl std::fmt::Display) -> Error {
    Error::new(format!("glTF: {what}"))
}

fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

pub(super) fn parse(bytes: &[u8]) -> Result<MeshData> {
    let (doc, bin) = split_container(bytes)?;
    if !matches!(doc, Value::Object(_)) {
        return Err(bad("the asset is not a JSON object"));
    }
    let reader = Document { doc: &doc, bin };

    // Before reading any accessor, so that this costs only the nodes.
    check_visits(&doc, visit_limit(bytes.len()))?;

    // A dry walk first, so that reuse is refused before anything is expanded.
    let limit = MIN_EXPANSION_LIMIT.max(bytes.len() as u64);
    let mut expanded = 0u64;
    walk_nodes(&doc, &mut |node, _| {
        for primitive in node_primitives(&doc, node)? {
            let (positions, indices) = reader.primitive(primitive)?;
            let corners = indices.as_ref().map_or(positions.count, |i| i.count);
            expanded = expanded
                .saturating_add(positions.count as u64)
                .saturating_add(3 * corners as u64)
                .saturating_add(1);
            if expanded > limit {
                return Err(bad(format!(
                    "too many vertices and corners once nodes reuse their meshes (limit {limit})"
                )));
            }
        }
        Ok(())
    })?;

    let mut out = MeshData {
        positions: Vec::new(),
        indices: Vec::new(),
    };
    walk_nodes(&doc, &mut |node, world| {
        for primitive in node_primitives(&doc, node)? {
            reader.append_primitive(primitive, world, &mut out)?;
        }
        Ok(())
    })?;
    if out.indices.is_empty() {
        return Err(bad("the file has no triangle meshes"));
    }
    check_mesh(out, "glTF")
}

/// Refuses a file whose nodes visit more than `limit` primitives in total.
fn check_visits(doc: &Value, limit: u64) -> Result<()> {
    let mut visits = 0u64;
    walk_nodes(doc, &mut |node, _| {
        visits = visits.saturating_add(node_primitives(doc, node)?.len() as u64);
        if visits > limit {
            return Err(bad(format!(
                "too many node-primitive visits once nodes reuse their meshes (limit {limit})"
            )));
        }
        Ok(())
    })
}

/// The primitives of the mesh a node holds, if any.
fn node_primitives<'a>(doc: &'a Value, node: &Value) -> Result<&'a [Value]> {
    let Some(mesh) = node.get("mesh") else {
        return Ok(&[]);
    };
    mesh.as_index()
        .and_then(|m| item(doc, "meshes", m))
        .and_then(|m| m.get("primitives"))
        .and_then(Value::as_array)
        .ok_or_else(|| bad("a node points at a missing mesh"))
}

/// Splits the file into the asset JSON and the body chunk (if any).
fn split_container(bytes: &[u8]) -> Result<(Value, Option<&[u8]>)> {
    if bytes.len() < 12 || &bytes[..4] != MAGIC {
        return Err(Error::new("not a GLB file (binary glTF)"));
    }
    let version = le32(bytes, 4).unwrap_or(0);
    if version != 2 {
        return Err(bad(format!("unsupported GLB version {version} (only 2.0)")));
    }
    let length = le32(bytes, 8).unwrap_or(0) as usize;
    if length > bytes.len() {
        return Err(bad("the GLB length runs past the end of the file"));
    }
    let mut json = None;
    let mut bin = None;
    let mut at = 12;
    while at + 8 <= length {
        let chunk = le32(bytes, at).unwrap_or(0) as usize;
        let kind = le32(bytes, at + 4).unwrap_or(0);
        let start = at + 8;
        let end = start
            .checked_add(chunk)
            .filter(|&end| end <= length)
            .ok_or_else(|| bad("a GLB chunk runs past the end of the file"))?;
        match kind {
            CHUNK_JSON if json.is_none() => json = Some(&bytes[start..end]),
            CHUNK_BIN if bin.is_none() => bin = Some(&bytes[start..end]),
            _ => {}
        }
        at = end;
    }
    let json = json.ok_or_else(|| bad("the GLB has no JSON chunk"))?;
    let doc = json::parse(json).map_err(|e| bad(format!("the asset JSON does not parse ({e})")))?;
    Ok((doc, bin))
}

/// `doc[list][index]`.
fn item<'a>(doc: &'a Value, list: &str, index: u64) -> Option<&'a Value> {
    doc.get(list)?.as_array()?.get(usize::try_from(index).ok()?)
}

/// Column-major affine matrix.
type Matrix = [f64; 16];

const IDENTITY: Matrix = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn numbers<const N: usize>(value: &Value, what: &str) -> Result<[f64; N]> {
    let items = value
        .as_array()
        .filter(|items| items.len() == N)
        .ok_or_else(|| bad(format!("{what} ({N} finite numbers)")))?;
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| bad(format!("{what} ({N} finite numbers)")))?;
    }
    Ok(out)
}

/// The node's local matrix: `matrix`, else TRS composed.
fn local_matrix(node: &Value, index: usize) -> Result<Matrix> {
    if let Some(matrix) = node.get("matrix") {
        return numbers(matrix, &format!("node {index} has a bad matrix"));
    }
    let what = format!("node {index} has a bad TRS transform");
    let t: [f64; 3] = match node.get("translation") {
        Some(v) => numbers(v, &what)?,
        None => [0.0; 3],
    };
    let r: [f64; 4] = match node.get("rotation") {
        Some(v) => numbers(v, &format!("node {index} has a bad rotation"))?,
        None => [0.0, 0.0, 0.0, 1.0],
    };
    let s: [f64; 3] = match node.get("scale") {
        Some(v) => numbers(v, &what)?,
        None => [1.0; 3],
    };
    let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
    if !len.is_finite() || len <= 1e-30 {
        return Err(bad(format!("node {index} has a zero rotation quaternion")));
    }
    let [x, y, z, w] = r.map(|c| c / len);
    Ok([
        (1.0 - 2.0 * (y * y + z * z)) * s[0],
        2.0 * (x * y + z * w) * s[0],
        2.0 * (x * z - y * w) * s[0],
        0.0,
        2.0 * (x * y - z * w) * s[1],
        (1.0 - 2.0 * (x * x + z * z)) * s[1],
        2.0 * (y * z + x * w) * s[1],
        0.0,
        2.0 * (x * z + y * w) * s[2],
        2.0 * (y * z - x * w) * s[2],
        (1.0 - 2.0 * (x * x + y * y)) * s[2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    ])
}

/// `a * b` for column-major affine matrices.
fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..3 {
            out[4 * column + row] = (0..3)
                .map(|k| a[4 * k + row] * b[4 * column + k])
                .sum::<f64>()
                + if column == 3 { a[12 + row] } else { 0.0 };
        }
    }
    out[15] = 1.0;
    out
}

/// The determinant of the 3 × 3 part: negative when the node mirrors.
fn determinant(m: &Matrix) -> f64 {
    m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2])
        + m[8] * (m[1] * m[6] - m[5] * m[2])
}

/// Calls `visit` for every node of the default scene (without `scenes`, of
/// every parentless node) with its world matrix. The hierarchy must be a tree,
/// as glTF requires: a node seen twice is refused, which rules out cycles and
/// the exponential walks of shared children, and an explicit stack means depth
/// cannot overflow anything.
fn walk_nodes(doc: &Value, visit: &mut impl FnMut(&Value, &Matrix) -> Result<()>) -> Result<()> {
    let nodes = doc
        .get("nodes")
        .and_then(Value::as_array)
        .unwrap_or_default();
    let roots = root_nodes(doc, nodes)?;
    let mut seen = try_vec(nodes.len(), false)?;
    let mut stack: Vec<(usize, Matrix)> = Vec::new();
    for &root in roots.iter().rev() {
        stack.push((root, IDENTITY));
    }
    while let Some((index, parent)) = stack.pop() {
        if std::mem::replace(&mut seen[index], true) {
            return Err(bad(format!(
                "node {index} is reached more than once (the hierarchy must be a tree)"
            )));
        }
        let node = &nodes[index];
        if node
            .get("extensions")
            .and_then(|e| e.get("EXT_mesh_gpu_instancing"))
            .is_some()
        {
            return Err(bad(format!(
                "node {index} uses EXT_mesh_gpu_instancing, which is not supported"
            )));
        }
        let world = multiply(&parent, &local_matrix(node, index)?);
        visit(node, &world)?;
        let children = match node.get("children") {
            Some(c) => c
                .as_array()
                .ok_or_else(|| bad(format!("node {index} has a bad children list")))?,
            None => &[],
        };
        for child in children.iter().rev() {
            let child = child
                .as_index()
                .and_then(|c| usize::try_from(c).ok())
                .filter(|&c| c < nodes.len())
                .ok_or_else(|| {
                    bad(format!(
                        "node {index} has a child past the {} nodes",
                        nodes.len()
                    ))
                })?;
            push_all(&mut stack, &[(child, world)])?;
        }
    }
    Ok(())
}

fn root_nodes(doc: &Value, nodes: &[Value]) -> Result<Vec<usize>> {
    let scenes = doc
        .get("scenes")
        .and_then(Value::as_array)
        .unwrap_or_default();
    if scenes.is_empty() {
        let mut parented = try_vec(nodes.len(), false)?;
        for node in nodes {
            let children = node.get("children").and_then(Value::as_array);
            for child in children.unwrap_or_default() {
                if let Some(slot) = child
                    .as_index()
                    .and_then(|c| usize::try_from(c).ok())
                    .and_then(|c| parented.get_mut(c))
                {
                    *slot = true;
                }
            }
        }
        return Ok((0..nodes.len()).filter(|&i| !parented[i]).collect());
    }
    let index = match doc.get("scene") {
        Some(v) => v.as_index().ok_or_else(|| bad("a bad default scene"))?,
        None => 0,
    };
    let scene = usize::try_from(index)
        .ok()
        .and_then(|i| scenes.get(i))
        .ok_or_else(|| {
            bad(format!(
                "scene {index} is missing (the file has {})",
                scenes.len()
            ))
        })?;
    let listed = scene
        .get("nodes")
        .and_then(Value::as_array)
        .unwrap_or_default();
    let mut roots = with_capacity(listed.len())?;
    for root in listed {
        roots.push(
            root.as_index()
                .and_then(|r| usize::try_from(r).ok())
                .filter(|&r| r < nodes.len())
                .ok_or_else(|| bad(format!("scene {index} points at a missing node")))?,
        );
    }
    Ok(roots)
}

/// An accessor's elements, bounds-checked against its buffer view.
struct Accessor<'a> {
    bytes: &'a [u8],
    component_type: u64,
    components: usize,
    count: usize,
    stride: usize,
}

impl Accessor<'_> {
    /// The little-endian bytes of `element`'s component `k`, `N` wide.
    fn component<const N: usize>(&self, element: usize, k: usize) -> [u8; N] {
        let at = element * self.stride + k * N;
        let mut out = [0; N];
        out.copy_from_slice(&self.bytes[at..at + N]);
        out
    }
}

struct Document<'a> {
    doc: &'a Value,
    bin: Option<&'a [u8]>,
}

impl<'a> Document<'a> {
    /// The bytes of a buffer: only the first buffer, without a URI, is the body.
    fn buffer(&self, index: u64) -> Result<&'a [u8]> {
        let declared = item(self.doc, "buffers", index);
        if declared.and_then(|b| b.get("uri")).is_some() {
            return Err(bad(format!(
                "buffer {index} has a URI: external and embedded-URI buffers are not supported, only the GLB body"
            )));
        }
        if index != 0 {
            return Err(bad(format!(
                "buffer {index} is missing (only the GLB body is supported)"
            )));
        }
        let bin = self
            .bin
            .ok_or_else(|| bad("the file has no binary body chunk"))?;
        let declared_length = declared
            .and_then(|b| b.get("byteLength"))
            .and_then(Value::as_index)
            .and_then(|n| usize::try_from(n).ok());
        Ok(&bin[..declared_length.unwrap_or(bin.len()).min(bin.len())])
    }

    fn accessor(&self, index: &Value, what: &str) -> Result<Accessor<'a>> {
        let accessor = index
            .as_index()
            .and_then(|i| item(self.doc, "accessors", i))
            .ok_or_else(|| bad(format!("the {what} accessor is missing")))?;
        let field = |name: &str| accessor.get(name).and_then(Value::as_index);
        let component_type = field("componentType")
            .ok_or_else(|| bad(format!("the {what} accessor has no component type")))?;
        let size = match component_type {
            5120 | UNSIGNED_BYTE => 1,
            5122 | UNSIGNED_SHORT => 2,
            UNSIGNED_INT | FLOAT => 4,
            other => {
                return Err(bad(format!(
                    "the {what} accessor has component type {other}"
                )));
            }
        };
        let components = match accessor.get("type").and_then(Value::as_str) {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") | Some("MAT2") => 4,
            Some("MAT3") => 9,
            Some("MAT4") => 16,
            _ => return Err(bad(format!("the {what} accessor has a bad type"))),
        };
        let count =
            field("count").ok_or_else(|| bad(format!("the {what} accessor has a bad count")))?;
        if accessor.get("sparse").is_some() {
            return Err(bad(format!("sparse accessors are not supported ({what})")));
        }
        let view = accessor
            .get("bufferView")
            .and_then(Value::as_index)
            .ok_or_else(|| bad(format!("the {what} accessor has no buffer view")))?;
        let buffer_view = item(self.doc, "bufferViews", view)
            .ok_or_else(|| bad(format!("the {what} buffer view {view} is missing")))?;
        if let Some(extensions) = buffer_view.get("extensions")
            && (extensions.get("EXT_meshopt_compression").is_some()
                || extensions.get("KHR_meshopt_compression").is_some())
        {
            return Err(bad(format!(
                "the {what} data is meshopt-compressed, which is not supported"
            )));
        }
        let view_field = |name: &str, default: Option<u64>| match buffer_view.get(name) {
            Some(v) => v
                .as_index()
                .ok_or_else(|| bad(format!("the {what} buffer view has a bad {name}"))),
            None => default.ok_or_else(|| bad(format!("the {what} buffer view has no {name}"))),
        };
        let buffer = self.buffer(view_field("buffer", Some(0))?)?;
        let view_offset = view_field("byteOffset", Some(0))?;
        let view_length = view_field("byteLength", None)?;
        let element = components as u64 * size;
        let stride = view_field("byteStride", Some(element))?;
        if stride < element {
            return Err(bad(format!(
                "the {what} stride {stride} is narrower than its elements"
            )));
        }
        let accessor_offset = accessor
            .get("byteOffset")
            .map_or(Some(0), Value::as_index)
            .ok_or_else(|| bad(format!("the {what} accessor has a bad byteOffset")))?;
        // All of it fits u128: every factor is below 2^53.
        let view_end = u128::from(view_offset) + u128::from(view_length);
        let used = if count == 0 {
            0
        } else {
            u128::from(count - 1) * u128::from(stride) + u128::from(element)
        };
        if view_end > buffer.len() as u128 {
            return Err(bad(format!(
                "the {what} buffer view runs past the end of its buffer"
            )));
        }
        if u128::from(accessor_offset) + used > u128::from(view_length) {
            return Err(bad(format!(
                "the {what} accessor runs past the end of its buffer view"
            )));
        }
        // Both now sit inside the buffer, so they fit usize.
        let start = (view_offset + accessor_offset) as usize;
        let end = view_end as usize;
        Ok(Accessor {
            bytes: &buffer[start.min(end)..end],
            component_type,
            components,
            count: count as usize,
            stride: stride as usize,
        })
    }

    /// A primitive's POSITION and index accessors, checked.
    fn primitive(&self, primitive: &Value) -> Result<(Accessor<'a>, Option<Accessor<'a>>)> {
        if primitive
            .get("extensions")
            .and_then(|e| e.get("KHR_draco_mesh_compression"))
            .is_some()
        {
            return Err(bad(
                "Draco-compressed primitives (KHR_draco_mesh_compression) are not supported",
            ));
        }
        let position = primitive
            .get("attributes")
            .and_then(|a| a.get("POSITION"))
            .ok_or_else(|| bad("a primitive has no POSITION"))?;
        let positions = self.accessor(position, "POSITION")?;
        if positions.components != 3 || positions.component_type != FLOAT {
            return Err(bad("POSITION is not VEC3 of floats"));
        }
        let indices = match primitive.get("indices") {
            Some(index) => Some(self.accessor(index, "index")?),
            None => None,
        };
        Ok((positions, indices))
    }

    /// Bakes one primitive through `world` and appends it to `out`.
    fn append_primitive(
        &self,
        primitive: &Value,
        world: &Matrix,
        out: &mut MeshData,
    ) -> Result<()> {
        let (positions, indices) = self.primitive(primitive)?;
        let base = out.positions.len() / 3;
        let total = base + positions.count;
        if total > u32::MAX as usize {
            return Err(bad(format!(
                "{total} vertices is too many for 32-bit indices"
            )));
        }
        out.positions
            .try_reserve(3 * positions.count)
            .map_err(|_| super::out_of_memory())?;
        for v in 0..positions.count {
            let p = [0, 1, 2].map(|k| f64::from(f32::from_le_bytes(positions.component(v, k))));
            for row in 0..3 {
                let value = world[row] * p[0]
                    + world[4 + row] * p[1]
                    + world[8 + row] * p[2]
                    + world[12 + row];
                if !value.is_finite() {
                    return Err(bad("a position is not a finite number"));
                }
                out.positions.push(value);
            }
        }
        let triangles = triangle_corners(primitive, indices.as_ref(), positions.count)?;
        let mirrored = determinant(world) < 0.0;
        out.indices
            .try_reserve(triangles.len())
            .map_err(|_| super::out_of_memory())?;
        for t in triangles.as_chunks::<3>().0 {
            let (b, c) = if mirrored { (t[2], t[1]) } else { (t[1], t[2]) };
            for corner in [t[0], b, c] {
                // Below `positions.count`, so `base + corner` fits: see `total`.
                out.indices.push(base as u32 + corner);
            }
        }
        Ok(())
    }
}

/// A primitive's triangles as 3 vertex ids each: its mode applied to its
/// indices, or to the vertices in order without any.
fn triangle_corners(
    primitive: &Value,
    indices: Option<&Accessor>,
    vertices: usize,
) -> Result<Vec<u32>> {
    let mode = match primitive.get("mode") {
        Some(m) => m.as_index().ok_or_else(|| bad("a bad primitive mode"))?,
        None => 4,
    };
    if !matches!(mode, 4..=6) {
        return Err(bad(format!(
            "primitive mode {mode} (only TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN are supported)"
        )));
    }
    let count = indices.map_or(vertices, |i| i.count);
    let corner: Box<dyn Fn(usize) -> u32 + '_> = match indices {
        None => Box::new(|i| i as u32),
        Some(a) => {
            if a.components != 1 {
                return Err(bad("the index accessor is not SCALAR"));
            }
            match a.component_type {
                UNSIGNED_BYTE => Box::new(|i| u32::from(a.component::<1>(i, 0)[0])),
                UNSIGNED_SHORT => {
                    Box::new(|i| u32::from(u16::from_le_bytes(a.component::<2>(i, 0))))
                }
                UNSIGNED_INT => Box::new(|i| u32::from_le_bytes(a.component::<4>(i, 0))),
                _ => return Err(bad("indices are not unsigned 8, 16 or 32-bit")),
            }
        }
    };
    let mut out = with_capacity(match mode {
        4 => count,
        _ => 3 * count.saturating_sub(2),
    })?;
    if mode == 4 && !count.is_multiple_of(3) {
        return Err(bad(format!(
            "{count} TRIANGLES corners is not a multiple of 3"
        )));
    }
    let checked = |i: usize| -> Result<u32> {
        let c = corner(i);
        if c as usize >= vertices {
            return Err(bad(format!(
                "index {c} is past the primitive's {vertices} vertices"
            )));
        }
        Ok(c)
    };
    match mode {
        4 => {
            for i in 0..count {
                out.push(checked(i)?);
            }
        }
        5 => {
            // A strip: alternate the winding so neighbours face the same way.
            for i in 0..count.saturating_sub(2) {
                let (a, b, c) = (checked(i)?, checked(i + 1)?, checked(i + 2)?);
                out.extend(if i % 2 == 0 { [a, b, c] } else { [a, c, b] });
            }
        }
        _ => {
            for i in 0..count.saturating_sub(2) {
                out.extend([checked(0)?, checked(i + 1)?, checked(i + 2)?]);
            }
        }
    }
    Ok(out)
}

pub(super) fn write<W: Write>(data: &MeshData, out: &mut W) -> Result<()> {
    let vertices = data.positions.len() / 3;
    let corners = data.indices.len();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut singles = with_capacity(data.positions.len())?;
    for p in data.positions.as_chunks::<3>().0 {
        for k in 0..3 {
            let value = to_f32(p[k], "glTF")?;
            min[k] = min[k].min(value);
            max[k] = max[k].max(value);
            singles.push(value);
        }
    }
    let positions_length = 12 * vertices as u64;
    let indices_length = 4 * corners as u64;
    let body = positions_length + indices_length;
    let json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"fast-mesh-cli"}},"#,
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"#,
            r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"mode":4}}]}}],"#,
            r#""accessors":[{{"bufferView":0,"componentType":5126,"count":{vertices},"type":"VEC3","#,
            r#""min":[{},{},{}],"max":[{},{},{}]}},"#,
            r#"{{"bufferView":1,"componentType":5125,"count":{corners},"type":"SCALAR"}}],"#,
            r#""bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{positions_length},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{positions_length},"byteLength":{indices_length},"target":34963}}],"#,
            r#""buffers":[{{"byteLength":{body}}}]}}"#
        ),
        min[0],
        min[1],
        min[2],
        max[0],
        max[1],
        max[2],
        vertices = vertices,
        corners = corners,
        positions_length = positions_length,
        indices_length = indices_length,
        body = body,
    );
    let padding = (4 - json.len() % 4) % 4;
    let json_length = (json.len() + padding) as u64;
    let total = 12 + 8 + json_length + 8 + body;
    let total =
        u32::try_from(total).map_err(|_| bad("the mesh is too large for a GLB's 32-bit length"))?;
    out.write_all(MAGIC)?;
    out.write_all(&2u32.to_le_bytes())?;
    out.write_all(&total.to_le_bytes())?;
    out.write_all(&(json_length as u32).to_le_bytes())?;
    out.write_all(&CHUNK_JSON.to_le_bytes())?;
    out.write_all(json.as_bytes())?;
    out.write_all(&b"   "[..padding])?;
    out.write_all(&(body as u32).to_le_bytes())?;
    out.write_all(&CHUNK_BIN.to_le_bytes())?;
    for value in &singles {
        out.write_all(&value.to_le_bytes())?;
    }
    for index in &data.indices {
        out.write_all(&index.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visits_are_capped_exactly() {
        let json = r#"{"nodes":[{"mesh":0},{"mesh":0},{"mesh":0},{}],
            "meshes":[{"primitives":[{},{}]}]}"#;
        let doc = json::parse(json.as_bytes()).unwrap();
        check_visits(&doc, 6).unwrap();
        let message = check_visits(&doc, 5).unwrap_err().to_string();
        assert!(message.contains("node-primitive visits"), "{message}");
    }

    #[test]
    fn the_visit_limit_grows_with_the_file_only_past_its_floor() {
        assert_eq!(visit_limit(0), MIN_VISIT_LIMIT);
        assert_eq!(visit_limit(16 << 20), MIN_VISIT_LIMIT);
        assert_eq!(visit_limit(32 << 20), 2 << 20);
    }
}
