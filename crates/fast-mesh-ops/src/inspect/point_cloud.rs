//! Point cloud parsing, outlier filtering, voxel downsampling, normal orientation, and surface reconstruction.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use fast_mesh_store::Mesh;

use crate::align::jacobi_eigen_3x3;
use crate::error::{OpsError, Result};
use crate::icp::{add, cross, dot, norm, sub, unit};
use crate::marching_cubes::marching_cubes_raw;
use crate::{allocation_allowed, reserve};

/// Point cloud container storing 3D coordinates with optional normals and RGB colors.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PointCloud {
    /// 3D point coordinates.
    pub positions: Vec<[f32; 3]>,
    /// Optional per-point unit normals matching `positions.len()`.
    pub normals: Option<Vec<[f32; 3]>>,
    /// Optional per-point linear RGB colors in `[0.0, 1.0]` matching `positions.len()`.
    pub colors: Option<Vec<[f32; 3]>>,
}

impl PointCloud {
    /// Create a point cloud from positions only.
    pub fn from_positions(positions: Vec<[f32; 3]>) -> Self {
        Self {
            positions,
            normals: None,
            colors: None,
        }
    }

    /// Parse an ASCII `.xyz` / `.xyzn` / `.xyzrgb` point cloud string.
    pub fn parse_xyz(input: &str) -> Result<Self> {
        parse_xyz(input)
    }

    /// Parse an ASCII `.pcd` (`DATA ascii`) point cloud string.
    pub fn parse_pcd_ascii(input: &str) -> Result<Self> {
        parse_pcd_ascii(input)
    }

    /// Parse an ASCII or binary `.ply` vertex cloud.
    pub fn parse_ply_points(bytes: &[u8]) -> Result<Self> {
        parse_ply_points(bytes)
    }

    /// Number of points in the cloud.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` when the cloud has zero points.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Summary statistics from an outlier removal filter pass.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlierReport {
    /// Number of points retained as inliers.
    pub kept_count: usize,
    /// Number of points removed as outliers.
    pub removed_count: usize,
    /// Mean neighbor distance (SOR) or mean neighbor count (ROR).
    pub mean_distance: f64,
    /// Population standard deviation of neighbor distances (SOR) or `0.0` (ROR).
    pub std_dev: f64,
    /// Effective cutoff threshold applied by the filter.
    pub threshold: f64,
}

/// Surface reconstruction algorithm family for [`reconstruct_surface`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructMethod {
    /// Oriented MLS / Hoppe local signed implicit field extracted via Marching Cubes.
    ImplicitMlsMarchingCubes,
    /// Advancing-front Ball-Pivoting triangulation interpolating input points.
    BallPivoting,
}

/// Options controlling point-cloud surface triangulation in [`reconstruct_surface`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceReconstructOptions {
    /// Triangulation algorithm to use.
    pub method: ReconstructMethod,
    /// Grid cell size for [`ReconstructMethod::ImplicitMlsMarchingCubes`] (`0.0` auto-derives).
    pub voxel_size: f32,
    /// Ball radius for [`ReconstructMethod::BallPivoting`] (`0.0` auto-derives).
    pub ball_radius: f32,
    /// Neighbor count used for normal estimation and MLS smoothing (`0` defaults to `12`).
    pub k_neighbors: usize,
}

impl Default for SurfaceReconstructOptions {
    fn default() -> Self {
        Self {
            method: ReconstructMethod::ImplicitMlsMarchingCubes,
            voxel_size: 0.0,
            ball_radius: 0.0,
            k_neighbors: 12,
        }
    }
}

// ============================================================================
// Parsers: XYZ, PCD (ASCII), PLY (ASCII + Binary)
// ============================================================================

/// Parse an ASCII `.xyz` / `.xyzn` / `.xyzrgb` string into a [`PointCloud`].
pub fn parse_xyz(input: &str) -> Result<PointCloud> {
    allocation_allowed()?;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut expected_cols: Option<usize> = None;

    for raw_line in input.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }
        let mut vals = Vec::with_capacity(9);
        for tok in trimmed.split_whitespace() {
            let v: f32 = tok
                .parse()
                .map_err(|_| OpsError::InvalidInput("invalid float in xyz line"))?;
            if !v.is_finite() {
                return Err(OpsError::InvalidInput("non-finite coordinate in xyz data"));
            }
            vals.push(v);
        }
        if vals.len() < 3 {
            return Err(OpsError::InvalidInput(
                "xyz row must contain at least 3 coordinates",
            ));
        }
        match expected_cols {
            Some(c) if c != vals.len() => {
                return Err(OpsError::InvalidInput("inconsistent column count in xyz"));
            }
            None => expected_cols = Some(vals.len()),
            _ => {}
        }
        rows.push(vals);
    }

    if rows.is_empty() {
        return Ok(PointCloud::default());
    }

    let cols = expected_cols.unwrap_or(3);
    let mut positions = Vec::with_capacity(rows.len());
    for r in &rows {
        positions.push([r[0], r[1], r[2]]);
    }

    let mut normals = None;
    let mut colors = None;

    if cols == 6 {
        // Distinguish nx ny nz (unit-like or signed) vs rgb colors.
        let mut all_rgb_range = true;
        let mut unit_like = 0usize;
        for r in &rows {
            let n2 = r[3] * r[3] + r[4] * r[4] + r[5] * r[5];
            if (n2.sqrt() - 1.0).abs() <= 0.15 {
                unit_like += 1;
            }
            if r[3] < 0.0
                || r[4] < 0.0
                || r[5] < 0.0
                || r[3] > 255.0
                || r[4] > 255.0
                || r[5] > 255.0
            {
                all_rgb_range = false;
            }
        }
        if unit_like * 2 >= rows.len() || !all_rgb_range {
            normals = Some(
                rows.iter()
                    .map(|r| normalize_or_default([r[3], r[4], r[5]]))
                    .collect(),
            );
        } else {
            colors = Some(normalize_rgb_rows(&rows, 3));
        }
    } else if cols == 7 {
        colors = Some(normalize_rgb_rows(&rows, 3));
    } else if cols >= 9 {
        normals = Some(
            rows.iter()
                .map(|r| normalize_or_default([r[3], r[4], r[5]]))
                .collect(),
        );
        colors = Some(normalize_rgb_rows(&rows, 6));
    }

    Ok(PointCloud {
        positions,
        normals,
        colors,
    })
}

fn normalize_rgb_rows(rows: &[Vec<f32>], offset: usize) -> Vec<[f32; 3]> {
    let needs_div_255 = rows
        .iter()
        .any(|r| r[offset] > 1.0 || r[offset + 1] > 1.0 || r[offset + 2] > 1.0);
    let scale = if needs_div_255 { 1.0 / 255.0 } else { 1.0 };
    rows.iter()
        .map(|r| {
            [
                (r[offset] * scale).clamp(0.0, 1.0),
                (r[offset + 1] * scale).clamp(0.0, 1.0),
                (r[offset + 2] * scale).clamp(0.0, 1.0),
            ]
        })
        .collect()
}

/// Parse an ASCII `.pcd` (`DATA ascii`) string into a [`PointCloud`].
pub fn parse_pcd_ascii(input: &str) -> Result<PointCloud> {
    allocation_allowed()?;
    let mut fields: Vec<String> = Vec::new();
    let mut types: Vec<char> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut declared_points: Option<usize> = None;
    let mut data_started = false;

    let mut lines = input.lines();
    for raw in lines.by_ref() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let key = parts
            .next()
            .ok_or(OpsError::InvalidInput("empty pcd header keyword"))?;
        match key.to_ascii_uppercase().as_str() {
            "FIELDS" => {
                fields = parts.map(|s| s.to_ascii_lowercase()).collect();
            }
            "TYPE" => {
                types = parts
                    .filter_map(|s| s.chars().next().map(|c| c.to_ascii_uppercase()))
                    .collect();
            }
            "COUNT" => {
                counts = parts
                    .map(|s| {
                        s.parse::<usize>()
                            .map_err(|_| OpsError::InvalidInput("invalid COUNT in pcd"))
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
            "POINTS" => {
                let p_str = parts
                    .next()
                    .ok_or(OpsError::InvalidInput("missing POINTS value in pcd"))?;
                declared_points = Some(
                    p_str
                        .parse()
                        .map_err(|_| OpsError::InvalidInput("invalid POINTS count in pcd"))?,
                );
            }
            "DATA" => {
                let mode = parts
                    .next()
                    .ok_or(OpsError::InvalidInput("missing DATA mode in pcd"))?;
                if !mode.eq_ignore_ascii_case("ascii") {
                    return Err(OpsError::InvalidInput(
                        "only ASCII PCD files are supported by parse_pcd_ascii",
                    ));
                }
                data_started = true;
                break;
            }
            _ => {}
        }
    }

    if !data_started {
        return Err(OpsError::InvalidInput("pcd header missing DATA directive"));
    }

    // Map field names to flattened token offsets accounting for COUNT.
    let mut col_idx = HashMap::new();
    let mut flat_cols = 0usize;
    for (i, name) in fields.iter().enumerate() {
        let cnt = counts.get(i).copied().unwrap_or(1);
        let ty = types.get(i).copied().unwrap_or('F');
        col_idx.insert(name.as_str(), (flat_cols, ty));
        flat_cols += cnt;
    }

    let ix = col_idx
        .get("x")
        .ok_or(OpsError::InvalidInput("pcd missing x field"))?
        .0;
    let iy = col_idx
        .get("y")
        .ok_or(OpsError::InvalidInput("pcd missing y field"))?
        .0;
    let iz = col_idx
        .get("z")
        .ok_or(OpsError::InvalidInput("pcd missing z field"))?
        .0;

    let normal_cols = col_idx
        .get("normal_x")
        .or_else(|| col_idx.get("nx"))
        .and_then(|&(a, _)| {
            col_idx
                .get("normal_y")
                .or_else(|| col_idx.get("ny"))
                .and_then(|&(b, _)| {
                    col_idx
                        .get("normal_z")
                        .or_else(|| col_idx.get("nz"))
                        .map(|&(c, _)| (a, b, c))
                })
        });

    let packed_rgb = col_idx.get("rgb").or_else(|| col_idx.get("rgba")).copied();
    let split_rgb = col_idx.get("r").and_then(|&(r, rt)| {
        col_idx
            .get("g")
            .and_then(|&(g, gt)| col_idx.get("b").map(|&(b, bt)| ((r, rt), (g, gt), (b, bt))))
    });

    let mut positions = Vec::new();
    if let Some(cnt) = declared_points {
        reserve(&mut positions, cnt.min(1_000_000))?;
    }
    let mut normals = if normal_cols.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let mut colors = if packed_rgb.is_some() || split_rgb.is_some() {
        Some(Vec::new())
    } else {
        None
    };

    for raw in lines {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < flat_cols.max(iz + 1) {
            return Err(OpsError::InvalidInput("truncated data row in pcd"));
        }
        let p = [
            parse_f32_finite(tokens[ix])?,
            parse_f32_finite(tokens[iy])?,
            parse_f32_finite(tokens[iz])?,
        ];
        positions.push(p);

        if let (Some((nx_i, ny_i, nz_i)), Some(nvec)) = (normal_cols, normals.as_mut()) {
            let n = [
                parse_f32_finite(tokens[nx_i])?,
                parse_f32_finite(tokens[ny_i])?,
                parse_f32_finite(tokens[nz_i])?,
            ];
            nvec.push(normalize_or_default(n));
        }

        if let Some(cvec) = colors.as_mut() {
            if let Some((col, ty)) = packed_rgb {
                let bits = if ty == 'F' {
                    parse_f32_finite(tokens[col])?.to_bits()
                } else {
                    tokens[col]
                        .parse::<u32>()
                        .map_err(|_| OpsError::InvalidInput("invalid packed rgb in pcd"))?
                };
                cvec.push([
                    f32::from(((bits >> 16) & 0xff) as u8) / 255.0,
                    f32::from(((bits >> 8) & 0xff) as u8) / 255.0,
                    f32::from((bits & 0xff) as u8) / 255.0,
                ]);
            } else if let Some(((r, rt), (g, gt), (b, bt))) = split_rgb {
                let mut rgb = [
                    parse_f32_finite(tokens[r])?,
                    parse_f32_finite(tokens[g])?,
                    parse_f32_finite(tokens[b])?,
                ];
                if rt == 'U'
                    || gt == 'U'
                    || bt == 'U'
                    || rgb[0] > 1.0
                    || rgb[1] > 1.0
                    || rgb[2] > 1.0
                {
                    rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
                }
                cvec.push([
                    rgb[0].clamp(0.0, 1.0),
                    rgb[1].clamp(0.0, 1.0),
                    rgb[2].clamp(0.0, 1.0),
                ]);
            }
        }
    }

    if let Some(cnt) = declared_points
        && positions.len() != cnt
    {
        return Err(OpsError::InvalidInput(
            "pcd data line count does not match POINTS header",
        ));
    }

    Ok(PointCloud {
        positions,
        normals,
        colors,
    })
}

/// Parse an ASCII or binary `.ply` vertex cloud into a [`PointCloud`].
pub fn parse_ply_points(bytes: &[u8]) -> Result<PointCloud> {
    allocation_allowed()?;
    let header_end = find_ply_header_end(bytes)?;
    let header_str = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| OpsError::InvalidInput("invalid utf-8 in ply header"))?;
    let payload = &bytes[header_end..];

    let mut lines = header_str.lines();
    let magic = lines
        .next()
        .map(str::trim)
        .ok_or(OpsError::InvalidInput("empty ply input"))?;
    if magic != "ply" {
        return Err(OpsError::InvalidInput("missing ply magic header"));
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PlyFormat {
        Ascii,
        LittleEndian,
        BigEndian,
    }
    #[derive(Clone, Copy)]
    enum ScalarTy {
        I8,
        U8,
        I16,
        U16,
        I32,
        U32,
        F32,
        F64,
    }
    impl ScalarTy {
        fn from_ply(s: &str) -> Option<Self> {
            match s {
                "char" | "int8" => Some(Self::I8),
                "uchar" | "uint8" => Some(Self::U8),
                "short" | "int16" => Some(Self::I16),
                "ushort" | "uint16" => Some(Self::U16),
                "int" | "int32" => Some(Self::I32),
                "uint" | "uint32" => Some(Self::U32),
                "float" | "float32" => Some(Self::F32),
                "double" | "float64" => Some(Self::F64),
                _ => None,
            }
        }
        fn byte_len(self) -> usize {
            match self {
                Self::I8 | Self::U8 => 1,
                Self::I16 | Self::U16 => 2,
                Self::I32 | Self::U32 | Self::F32 => 4,
                Self::F64 => 8,
            }
        }
        fn is_int_color(self) -> Option<f32> {
            match self {
                Self::U8 => Some(255.0),
                Self::U16 => Some(65535.0),
                _ => None,
            }
        }
    }

    let mut format: Option<PlyFormat> = None;
    let mut vertex_count: Option<usize> = None;
    let mut in_vertex = false;
    let mut vertex_props: Vec<(String, ScalarTy, usize)> = Vec::new();
    let mut stride = 0usize;

    for raw in lines {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with("comment") || trimmed.starts_with("obj_info") {
            continue;
        }
        let toks: Vec<&str> = trimmed.split_whitespace().collect();
        match toks.as_slice() {
            ["format", fmt, _ver] => {
                format = match *fmt {
                    "ascii" => Some(PlyFormat::Ascii),
                    "binary_little_endian" => Some(PlyFormat::LittleEndian),
                    "binary_big_endian" => Some(PlyFormat::BigEndian),
                    _ => return Err(OpsError::InvalidInput("unsupported ply format")),
                };
            }
            ["element", name, cnt] => {
                if *name == "vertex" {
                    in_vertex = true;
                    vertex_count = Some(
                        cnt.parse::<usize>()
                            .map_err(|_| OpsError::InvalidInput("invalid ply vertex count"))?,
                    );
                } else {
                    in_vertex = false;
                }
            }
            ["property", "list", ..] => {
                if in_vertex {
                    return Err(OpsError::InvalidInput(
                        "list property not supported on ply vertex element",
                    ));
                }
            }
            ["property", ty_str, prop_name] if in_vertex => {
                let ty = ScalarTy::from_ply(ty_str)
                    .ok_or(OpsError::InvalidInput("unsupported ply vertex property"))?;
                vertex_props.push(((*prop_name).to_ascii_lowercase(), ty, stride));
                stride += ty.byte_len();
            }
            ["end_header"] => break,
            _ => {}
        }
    }

    let fmt = format.ok_or(OpsError::InvalidInput("ply missing format declaration"))?;
    let v_count = vertex_count.ok_or(OpsError::InvalidInput("ply missing vertex element"))?;
    if v_count == 0 {
        return Ok(PointCloud::default());
    }

    let find_prop = |names: &[&str]| -> Option<usize> {
        vertex_props
            .iter()
            .position(|(n, _, _)| names.contains(&n.as_str()))
    };
    let ix = find_prop(&["x"]).ok_or(OpsError::InvalidInput("ply missing vertex x"))?;
    let iy = find_prop(&["y"]).ok_or(OpsError::InvalidInput("ply missing vertex y"))?;
    let iz = find_prop(&["z"]).ok_or(OpsError::InvalidInput("ply missing vertex z"))?;

    let nrm_idx = find_prop(&["nx"])
        .and_then(|a| find_prop(&["ny"]).and_then(|b| find_prop(&["nz"]).map(|c| (a, b, c))));
    let col_idx = find_prop(&["red", "r", "diffuse_red"]).and_then(|r| {
        find_prop(&["green", "g", "diffuse_green"])
            .and_then(|g| find_prop(&["blue", "b", "diffuse_blue"]).map(|b| (r, g, b)))
    });

    if fmt == PlyFormat::Ascii {
        let body = std::str::from_utf8(payload)
            .map_err(|_| OpsError::InvalidInput("invalid utf-8 in ascii ply body"))?;
        let mut positions = Vec::new();
        reserve(&mut positions, v_count.min(1_000_000))?;
        let mut normals = if nrm_idx.is_some() {
            Some(Vec::new())
        } else {
            None
        };
        let mut colors = if col_idx.is_some() {
            Some(Vec::new())
        } else {
            None
        };

        for raw in body.lines() {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            if positions.len() >= v_count {
                break;
            }
            let toks: Vec<&str> = trimmed.split_whitespace().collect();
            if toks.len() < vertex_props.len() {
                return Err(OpsError::InvalidInput("truncated ascii ply vertex line"));
            }
            positions.push([
                parse_f32_finite(toks[ix])?,
                parse_f32_finite(toks[iy])?,
                parse_f32_finite(toks[iz])?,
            ]);
            if let (Some((a, b, c)), Some(nvec)) = (nrm_idx, normals.as_mut()) {
                nvec.push(normalize_or_default([
                    parse_f32_finite(toks[a])?,
                    parse_f32_finite(toks[b])?,
                    parse_f32_finite(toks[c])?,
                ]));
            }
            if let (Some((r, g, b)), Some(cvec)) = (col_idx, colors.as_mut()) {
                let mut rgb = [
                    parse_f32_finite(toks[r])?,
                    parse_f32_finite(toks[g])?,
                    parse_f32_finite(toks[b])?,
                ];
                if let Some(div) = vertex_props[r].1.is_int_color() {
                    rgb = [rgb[0] / div, rgb[1] / div, rgb[2] / div];
                } else if rgb[0] > 1.0 || rgb[1] > 1.0 || rgb[2] > 1.0 {
                    rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
                }
                cvec.push([
                    rgb[0].clamp(0.0, 1.0),
                    rgb[1].clamp(0.0, 1.0),
                    rgb[2].clamp(0.0, 1.0),
                ]);
            }
        }
        if positions.len() != v_count {
            return Err(OpsError::InvalidInput(
                "ascii ply truncated before vertex count",
            ));
        }
        return Ok(PointCloud {
            positions,
            normals,
            colors,
        });
    }

    let needed_bytes = v_count
        .checked_mul(stride)
        .ok_or(OpsError::InvalidInput("ply byte size overflow"))?;
    if payload.len() < needed_bytes {
        return Err(OpsError::InvalidInput("truncated binary ply payload"));
    }

    let le = fmt == PlyFormat::LittleEndian;
    let read_val = |row: &[u8], prop_idx: usize| -> Result<f32> {
        let (_, ty, off) = vertex_props[prop_idx];
        let s = &row[off..off + ty.byte_len()];
        let v: f64 = match ty {
            ScalarTy::I8 => f64::from(s[0] as i8),
            ScalarTy::U8 => f64::from(s[0]),
            ScalarTy::I16 => f64::from(if le {
                i16::from_le_bytes([s[0], s[1]])
            } else {
                i16::from_be_bytes([s[0], s[1]])
            }),
            ScalarTy::U16 => f64::from(if le {
                u16::from_le_bytes([s[0], s[1]])
            } else {
                u16::from_be_bytes([s[0], s[1]])
            }),
            ScalarTy::I32 => f64::from(if le {
                i32::from_le_bytes([s[0], s[1], s[2], s[3]])
            } else {
                i32::from_be_bytes([s[0], s[1], s[2], s[3]])
            }),
            ScalarTy::U32 => f64::from(if le {
                u32::from_le_bytes([s[0], s[1], s[2], s[3]])
            } else {
                u32::from_be_bytes([s[0], s[1], s[2], s[3]])
            }),
            ScalarTy::F32 => f64::from(if le {
                f32::from_le_bytes([s[0], s[1], s[2], s[3]])
            } else {
                f32::from_be_bytes([s[0], s[1], s[2], s[3]])
            }),
            ScalarTy::F64 => {
                if le {
                    f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
                } else {
                    f64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
                }
            }
        };
        if !v.is_finite() {
            return Err(OpsError::InvalidInput("non-finite scalar in binary ply"));
        }
        Ok(v as f32)
    };

    let mut positions = Vec::with_capacity(v_count);
    let mut normals = if nrm_idx.is_some() {
        Some(Vec::with_capacity(v_count))
    } else {
        None
    };
    let mut colors = if col_idx.is_some() {
        Some(Vec::with_capacity(v_count))
    } else {
        None
    };

    for vi in 0..v_count {
        let row = &payload[vi * stride..(vi + 1) * stride];
        positions.push([read_val(row, ix)?, read_val(row, iy)?, read_val(row, iz)?]);
        if let (Some((a, b, c)), Some(nvec)) = (nrm_idx, normals.as_mut()) {
            nvec.push(normalize_or_default([
                read_val(row, a)?,
                read_val(row, b)?,
                read_val(row, c)?,
            ]));
        }
        if let (Some((r, g, b)), Some(cvec)) = (col_idx, colors.as_mut()) {
            let mut rgb = [read_val(row, r)?, read_val(row, g)?, read_val(row, b)?];
            if let Some(div) = vertex_props[r].1.is_int_color() {
                rgb = [rgb[0] / div, rgb[1] / div, rgb[2] / div];
            } else if rgb[0] > 1.0 || rgb[1] > 1.0 || rgb[2] > 1.0 {
                rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
            }
            cvec.push([
                rgb[0].clamp(0.0, 1.0),
                rgb[1].clamp(0.0, 1.0),
                rgb[2].clamp(0.0, 1.0),
            ]);
        }
    }

    Ok(PointCloud {
        positions,
        normals,
        colors,
    })
}

fn find_ply_header_end(bytes: &[u8]) -> Result<usize> {
    let limit = bytes.len().min(65536);
    let prefix = &bytes[..limit];
    for needle in [b"end_header\r\n".as_slice(), b"end_header\n".as_slice()] {
        if let Some(pos) = prefix.windows(needle.len()).position(|w| w == needle) {
            return Ok(pos + needle.len());
        }
    }
    Err(OpsError::InvalidInput("ply missing end_header marker"))
}

fn parse_f32_finite(tok: &str) -> Result<f32> {
    let v: f32 = tok
        .parse()
        .map_err(|_| OpsError::InvalidInput("invalid floating-point token"))?;
    if !v.is_finite() {
        return Err(OpsError::InvalidInput("non-finite floating-point value"));
    }
    Ok(v)
}

// ============================================================================
// Outlier Removal & Voxel Downsampling
// ============================================================================

/// Filter points whose average `k_neighbors` distance exceeds `mean + std_ratio * std_dev`.
pub fn statistical_outlier_removal(
    cloud: &PointCloud,
    k_neighbors: usize,
    std_ratio: f32,
) -> Result<(PointCloud, OutlierReport)> {
    validate_cloud(cloud)?;
    if cloud.is_empty() {
        return Err(OpsError::InvalidInput("point cloud is empty"));
    }
    if k_neighbors == 0 {
        return Err(OpsError::InvalidInput("k_neighbors must be >= 1"));
    }
    if !std_ratio.is_finite() || std_ratio < 0.0 {
        return Err(OpsError::InvalidInput(
            "std_ratio must be non-negative and finite",
        ));
    }

    let n = cloud.positions.len();
    if n == 1 {
        return Ok((
            cloud.clone(),
            OutlierReport {
                kept_count: 1,
                removed_count: 0,
                mean_distance: 0.0,
                std_dev: 0.0,
                threshold: 0.0,
            },
        ));
    }

    allocation_allowed()?;
    let pts = to_f64_points(&cloud.positions);
    let tree = KdTree::build(&pts);
    let target_k = k_neighbors.min(n - 1);

    let mut mean_dists = Vec::with_capacity(n);
    let mut heap = Vec::with_capacity(target_k + 1);
    let mut sum = 0.0;
    for (i, &p) in pts.iter().enumerate() {
        heap.clear();
        tree.knn(&pts, p, Some(i), target_k, &mut heap);
        let avg = heap.iter().map(|&(d2, _)| d2.sqrt()).sum::<f64>() / (heap.len().max(1) as f64);
        sum += avg;
        mean_dists.push(avg);
    }

    let mean = sum / (n as f64);
    let var = mean_dists
        .iter()
        .map(|&d| {
            let diff = d - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n as f64);
    let std_dev = var.sqrt();
    let threshold = mean + f64::from(std_ratio) * std_dev;

    let mut keep_indices = Vec::new();
    for (i, &d) in mean_dists.iter().enumerate() {
        if d <= threshold + 1e-12 {
            keep_indices.push(i);
        }
    }

    let kept_count = keep_indices.len();
    let filtered = subset_cloud(cloud, &keep_indices);
    Ok((
        filtered,
        OutlierReport {
            kept_count,
            removed_count: n - kept_count,
            mean_distance: mean,
            std_dev,
            threshold,
        },
    ))
}

/// Filter points having fewer than `min_neighbors` neighbors within `radius`.
pub fn radius_outlier_removal(
    cloud: &PointCloud,
    radius: f32,
    min_neighbors: usize,
) -> Result<(PointCloud, OutlierReport)> {
    validate_cloud(cloud)?;
    if cloud.is_empty() {
        return Err(OpsError::InvalidInput("point cloud is empty"));
    }
    if !radius.is_finite() || radius <= 0.0 {
        return Err(OpsError::InvalidInput("radius must be positive and finite"));
    }

    allocation_allowed()?;
    let n = cloud.positions.len();
    let pts = to_f64_points(&cloud.positions);
    let tree = KdTree::build(&pts);
    let r2 = f64::from(radius) * f64::from(radius);

    let mut keep_indices = Vec::new();
    let mut nbr_sum = 0usize;
    let mut buf = Vec::new();
    for (i, &p) in pts.iter().enumerate() {
        buf.clear();
        tree.radius_search(&pts, p, Some(i), r2, &mut buf);
        nbr_sum += buf.len();
        if buf.len() >= min_neighbors {
            keep_indices.push(i);
        }
    }

    let kept_count = keep_indices.len();
    let filtered = subset_cloud(cloud, &keep_indices);
    Ok((
        filtered,
        OutlierReport {
            kept_count,
            removed_count: n - kept_count,
            mean_distance: (nbr_sum as f64) / (n as f64),
            std_dev: 0.0,
            threshold: min_neighbors as f64,
        },
    ))
}

/// Downsample a point cloud by averaging points (and attributes) within cubic voxels.
pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f32) -> Result<PointCloud> {
    validate_cloud(cloud)?;
    if cloud.is_empty() {
        return Err(OpsError::InvalidInput("point cloud is empty"));
    }
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(OpsError::InvalidInput(
            "voxel_size must be positive and finite",
        ));
    }

    allocation_allowed()?;
    let (mut bmin, mut bmax) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &p in &cloud.positions {
        for c in 0..3 {
            let v = f64::from(p[c]);
            bmin[c] = bmin[c].min(v);
            bmax[c] = bmax[c].max(v);
        }
    }
    let vsize = f64::from(voxel_size);
    for c in 0..3 {
        if (bmax[c] - bmin[c]) / vsize > 1e14 {
            return Err(OpsError::InvalidInput(
                "voxel_size is too small for cloud bounding extent",
            ));
        }
    }

    let n = cloud.positions.len();
    let mut keyed: Vec<([i64; 3], usize)> = Vec::with_capacity(n);
    for (i, &p) in cloud.positions.iter().enumerate() {
        let key = [
            ((f64::from(p[0]) - bmin[0]) / vsize).floor() as i64,
            ((f64::from(p[1]) - bmin[1]) / vsize).floor() as i64,
            ((f64::from(p[2]) - bmin[2]) / vsize).floor() as i64,
        ];
        keyed.push((key, i));
    }
    keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut out_pos = Vec::new();
    let mut out_nrm = cloud.normals.as_ref().map(|_| Vec::new());
    let mut out_col = cloud.colors.as_ref().map(|_| Vec::new());

    let mut start = 0usize;
    while start < n {
        let key = keyed[start].0;
        let mut end = start + 1;
        while end < n && keyed[end].0 == key {
            end += 1;
        }
        let cnt = (end - start) as f64;
        let mut p_sum = [0.0f64; 3];
        let mut n_sum = [0.0f64; 3];
        let mut c_sum = [0.0f64; 3];
        for &(_, idx) in &keyed[start..end] {
            let p = cloud.positions[idx];
            p_sum[0] += f64::from(p[0]);
            p_sum[1] += f64::from(p[1]);
            p_sum[2] += f64::from(p[2]);
            if let Some(nrms) = &cloud.normals {
                let nv = nrms[idx];
                n_sum[0] += f64::from(nv[0]);
                n_sum[1] += f64::from(nv[1]);
                n_sum[2] += f64::from(nv[2]);
            }
            if let Some(cols) = &cloud.colors {
                let cv = cols[idx];
                c_sum[0] += f64::from(cv[0]);
                c_sum[1] += f64::from(cv[1]);
                c_sum[2] += f64::from(cv[2]);
            }
        }
        out_pos.push([
            (p_sum[0] / cnt) as f32,
            (p_sum[1] / cnt) as f32,
            (p_sum[2] / cnt) as f32,
        ]);
        if let Some(onrm) = out_nrm.as_mut() {
            let fallback = cloud.normals.as_ref().unwrap()[keyed[start].1];
            let avg_n = unit(n_sum)
                .map(|u| [u[0] as f32, u[1] as f32, u[2] as f32])
                .unwrap_or(fallback);
            onrm.push(avg_n);
        }
        if let Some(ocol) = out_col.as_mut() {
            ocol.push([
                (c_sum[0] / cnt) as f32,
                (c_sum[1] / cnt) as f32,
                (c_sum[2] / cnt) as f32,
            ]);
        }
        start = end;
    }

    Ok(PointCloud {
        positions: out_pos,
        normals: out_nrm,
        colors: out_col,
    })
}

// ============================================================================
// Consistent PCA + Hoppe Riemannian MST Normal Estimation
// ============================================================================

/// Estimate surface normals via local covariance PCA and Riemannian MST sign propagation.
pub fn estimate_normals(
    cloud: &mut PointCloud,
    k_neighbors: usize,
    camera_hint: Option<[f32; 3]>,
) -> Result<()> {
    validate_cloud(cloud)?;
    if cloud.positions.len() < 3 {
        return Err(OpsError::InvalidInput(
            "at least 3 points are required for normal estimation",
        ));
    }
    if k_neighbors < 2 {
        return Err(OpsError::InvalidInput("k_neighbors must be >= 2"));
    }
    let cam_f64 = match camera_hint {
        Some(c) => {
            if !c.iter().all(|v| v.is_finite()) {
                return Err(OpsError::InvalidInput("camera_hint must be finite"));
            }
            Some([f64::from(c[0]), f64::from(c[1]), f64::from(c[2])])
        }
        None => None,
    };

    allocation_allowed()?;
    let pts = to_f64_points(&cloud.positions);
    let n = pts.len();
    let target_k = k_neighbors.min(n - 1);
    let tree = KdTree::build(&pts);

    let mut raw_normals = vec![[0.0f64, 0.0, 1.0]; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut heap = Vec::with_capacity(target_k + 1);

    for (i, &p) in pts.iter().enumerate() {
        heap.clear();
        tree.knn(&pts, p, Some(i), target_k, &mut heap);
        let mut mean = p;
        for &(_, j) in &heap {
            mean = add(mean, pts[j]);
            adj[i].push(j);
            adj[j].push(i);
        }
        let cnt = (heap.len() + 1) as f64;
        mean = [mean[0] / cnt, mean[1] / cnt, mean[2] / cnt];

        let mut cov = [[0.0f64; 3]; 3];
        let mut accum = |q: [f64; 3]| {
            let d = sub(q, mean);
            for r in 0..3 {
                for c in 0..3 {
                    cov[r][c] += d[r] * d[c];
                }
            }
        };
        accum(p);
        for &(_, j) in &heap {
            accum(pts[j]);
        }
        let (_vals, vecs) = jacobi_eigen_3x3(cov);
        raw_normals[i] = unit([vecs[0][2], vecs[1][2], vecs[2][2]]).unwrap_or([0.0, 0.0, 1.0]);
    }

    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }

    // Connected-component MST sign propagation (Prim's algorithm on weight 1 - |dot(n_u, n_v)|).
    let mut visited = vec![false; n];
    let mut queue = BinaryHeap::<Reverse<(u64, usize, usize)>>::new();

    for start_seed in 0..n {
        if visited[start_seed] {
            continue;
        }

        // Gather connected component vertices via DFS to locate extreme or camera-closest seed.
        let mut stack = vec![start_seed];
        let mut comp = Vec::new();
        let mut in_comp = HashSet::new();
        in_comp.insert(start_seed);
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &v in &adj[u] {
                if !visited[v] && in_comp.insert(v) {
                    stack.push(v);
                }
            }
        }

        let seed = if let Some(cam) = cam_f64 {
            *comp
                .iter()
                .min_by(|&&a, &&b| {
                    let da = sub(pts[a], cam);
                    let db = sub(pts[b], cam);
                    dot(da, da).total_cmp(&dot(db, db)).then_with(|| a.cmp(&b))
                })
                .unwrap_or(&start_seed)
        } else {
            *comp
                .iter()
                .max_by(|&&a, &&b| {
                    pts[a][2]
                        .total_cmp(&pts[b][2])
                        .then_with(|| pts[a][1].total_cmp(&pts[b][1]))
                        .then_with(|| pts[a][0].total_cmp(&pts[b][0]))
                })
                .unwrap_or(&start_seed)
        };

        if let Some(cam) = cam_f64 {
            if dot(raw_normals[seed], sub(cam, pts[seed])) < 0.0 {
                raw_normals[seed] = negate(raw_normals[seed]);
            }
        } else if raw_normals[seed][2] < 0.0 {
            raw_normals[seed] = negate(raw_normals[seed]);
        }

        visited[seed] = true;
        queue.clear();
        for &v in &adj[seed] {
            let cost = 1.0 - dot(raw_normals[seed], raw_normals[v]).abs().min(1.0);
            queue.push(Reverse((cost.to_bits(), v, seed)));
        }

        while let Some(Reverse((_, u, parent))) = queue.pop() {
            if visited[u] {
                continue;
            }
            if dot(raw_normals[parent], raw_normals[u]) < 0.0 {
                raw_normals[u] = negate(raw_normals[u]);
            }
            visited[u] = true;
            for &v in &adj[u] {
                if !visited[v] {
                    let cost = 1.0 - dot(raw_normals[u], raw_normals[v]).abs().min(1.0);
                    queue.push(Reverse((cost.to_bits(), v, u)));
                }
            }
        }

        // Verify global outward orientation of this component when no camera viewpoint was supplied.
        if cam_f64.is_none() && comp.len() >= 4 {
            let mut centroid = [0.0f64; 3];
            for &u in &comp {
                centroid = add(centroid, pts[u]);
            }
            let inv_c = 1.0 / (comp.len() as f64);
            centroid = [
                centroid[0] * inv_c,
                centroid[1] * inv_c,
                centroid[2] * inv_c,
            ];

            let mut div_sum = 0.0;
            let mut scale_sum = 0.0;
            for &u in &comp {
                let d = sub(pts[u], centroid);
                scale_sum += norm(d);
                div_sum += dot(d, raw_normals[u]);
            }
            if div_sum < -1e-4 * scale_sum.max(1e-12) {
                for &u in &comp {
                    raw_normals[u] = negate(raw_normals[u]);
                }
            }
        }
    }

    cloud.normals = Some(
        raw_normals
            .into_iter()
            .map(|n| [n[0] as f32, n[1] as f32, n[2] as f32])
            .collect(),
    );
    Ok(())
}

// ============================================================================
// Surface Reconstruction: Implicit MLS Marching Cubes & Ball-Pivoting
// ============================================================================

/// Reconstruct a triangle mesh surface from a point cloud.
pub fn reconstruct_surface(
    cloud: &PointCloud,
    options: &SurfaceReconstructOptions,
) -> Result<Mesh> {
    validate_cloud(cloud)?;
    if cloud.positions.len() < 4 {
        return Err(OpsError::InvalidInput(
            "at least 4 points are required for surface reconstruction",
        ));
    }
    if !options.voxel_size.is_finite() || options.voxel_size < 0.0 {
        return Err(OpsError::InvalidInput(
            "voxel_size must be non-negative and finite",
        ));
    }
    if !options.ball_radius.is_finite() || options.ball_radius < 0.0 {
        return Err(OpsError::InvalidInput(
            "ball_radius must be non-negative and finite",
        ));
    }

    allocation_allowed()?;
    let k = if options.k_neighbors == 0 {
        12
    } else {
        options.k_neighbors.max(3)
    };

    // Estimate consistent normals on a scratch copy if input lacks normals.
    let owned_normals: Vec<[f64; 3]>;
    if let Some(nrms) = &cloud.normals {
        owned_normals = nrms
            .iter()
            .map(|&n| {
                unit([f64::from(n[0]), f64::from(n[1]), f64::from(n[2])]).unwrap_or([0.0, 0.0, 1.0])
            })
            .collect();
    } else {
        let mut scratch = cloud.clone();
        estimate_normals(&mut scratch, k, None)?;
        owned_normals = scratch
            .normals
            .as_ref()
            .map(|v| {
                v.iter()
                    .map(|&n| [f64::from(n[0]), f64::from(n[1]), f64::from(n[2])])
                    .collect()
            })
            .unwrap_or_default();
    }

    let pts = to_f64_points(&cloud.positions);
    match options.method {
        ReconstructMethod::ImplicitMlsMarchingCubes => {
            reconstruct_implicit_mls(&pts, &owned_normals, options.voxel_size, k)
        }
        ReconstructMethod::BallPivoting => {
            reconstruct_ball_pivoting(&cloud.positions, &pts, &owned_normals, options.ball_radius)
        }
    }
}

fn reconstruct_implicit_mls(
    pts: &[[f64; 3]],
    normals: &[[f64; 3]],
    voxel_size_opt: f32,
    k: usize,
) -> Result<Mesh> {
    let n = pts.len();
    let tree = KdTree::build(pts);
    let spacing = median_nn_spacing(pts, &tree).max(1e-6);

    let (mut bmin, mut bmax) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &p in pts {
        for c in 0..3 {
            bmin[c] = bmin[c].min(p[c]);
            bmax[c] = bmax[c].max(p[c]);
        }
    }
    let extent = [
        (bmax[0] - bmin[0]).max(spacing),
        (bmax[1] - bmin[1]).max(spacing),
        (bmax[2] - bmin[2]).max(spacing),
    ];
    let max_ext = extent[0].max(extent[1]).max(extent[2]);

    let dx = if voxel_size_opt > 0.0 {
        f64::from(voxel_size_opt)
    } else {
        (0.55 * spacing).max(max_ext / 64.0)
    };

    let pad = (2.5 * dx).max(1.5 * spacing);
    let grid_min = [bmin[0] - pad, bmin[1] - pad, bmin[2] - pad];
    let dims = [
        (((bmax[0] + pad - grid_min[0]) / dx).ceil() as usize + 1).max(3),
        (((bmax[1] + pad - grid_min[1]) / dx).ceil() as usize + 1).max(3),
        (((bmax[2] + pad - grid_min[2]) / dx).ceil() as usize + 1).max(3),
    ];

    let total_voxels = dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .ok_or(OpsError::InvalidInput("marching cubes grid size overflow"))?;
    if total_voxels > 1_200_000 {
        return Err(OpsError::InvalidInput(
            "voxel_size produces too many Marching Cubes cells",
        ));
    }

    let sigma = (1.15 * spacing).max(0.85 * dx);
    let inv_two_sig2 = 1.0 / (2.0 * sigma * sigma);
    let support_k = k.clamp(4, 16).min(n);

    let mut values = Vec::with_capacity(total_voxels);
    let mut heap = Vec::with_capacity(support_k + 1);

    for iz in 0..dims[2] {
        let z = grid_min[2] + (iz as f64) * dx;
        for iy in 0..dims[1] {
            let y = grid_min[1] + (iy as f64) * dx;
            for ix in 0..dims[0] {
                let x = grid_min[0] + (ix as f64) * dx;
                let g = [x, y, z];

                heap.clear();
                tree.knn(pts, g, None, support_k, &mut heap);
                let &(d0_sq, idx0) = &heap[0];
                let d0 = d0_sq.sqrt();

                // Guarantee outer border shell of grid is strictly outside so isosurface is closed.
                if ix == 0
                    || iy == 0
                    || iz == 0
                    || ix + 1 == dims[0]
                    || iy + 1 == dims[1]
                    || iz + 1 == dims[2]
                {
                    values.push(d0.max(dx) as f32);
                    continue;
                }

                // Compute smooth Gaussian-weighted signed distance (Hoppe/MLS projection).
                let mut w_sum = 0.0;
                let mut f_sum = 0.0;
                for &(d2, idx) in &heap {
                    let w = (-(d2 - d0_sq) * inv_two_sig2).exp();
                    let signed = dot(sub(g, pts[idx]), normals[idx]);
                    w_sum += w;
                    f_sum += w * signed;
                }

                let val = if w_sum > 1e-18 {
                    f_sum / w_sum
                } else {
                    dot(sub(g, pts[idx0]), normals[idx0])
                };
                values.push(val as f32);
            }
        }
    }

    let iso = marching_cubes_raw(
        [grid_min[0] as f32, grid_min[1] as f32, grid_min[2] as f32],
        dims,
        dx as f32,
        &values,
        0.0,
    )?;

    if iso.triangles.is_empty() {
        return Err(OpsError::InvalidInput(
            "implicit MLS Marching Cubes produced an empty mesh",
        ));
    }

    let flat_pos: Vec<f32> = iso.positions.iter().flat_map(|p| *p).collect();
    let flat_idx: Vec<u32> = iso.triangles.iter().flat_map(|t| *t).collect();
    Ok(Mesh::from_buffers(&flat_pos, &flat_idx)?)
}

fn reconstruct_ball_pivoting(
    raw_positions: &[[f32; 3]],
    pts: &[[f64; 3]],
    normals: &[[f64; 3]],
    ball_radius_opt: f32,
) -> Result<Mesh> {
    let n = pts.len();
    let tree = KdTree::build(pts);
    let spacing = median_nn_spacing(pts, &tree).max(1e-6);

    let radii: Vec<f64> = if ball_radius_opt > 0.0 {
        vec![f64::from(ball_radius_opt)]
    } else {
        vec![1.35 * spacing, 2.0 * spacing, 2.9 * spacing]
    };

    let mut bpa = BpaMesh::new(n);
    let mut nbr_buf = Vec::new();

    for &rho in &radii {
        let mut active_front: VecDeque<(usize, usize, usize, [f64; 3])> = VecDeque::new();
        let search_r2 = (2.05 * rho) * (2.05 * rho);
        let empty_r2 = (rho * (1.0 - 1e-5)) * (rho * (1.0 - 1e-5));
        let bctx = BpaCtx {
            pts,
            normals,
            tree: &tree,
            rho,
            search_r2,
            empty_r2,
        };

        // Seed from any existing active boundary half-edges first on multi-radius passes.
        for &[a, b, c] in &bpa.tris {
            for (u, v, opp) in [(a, b, c), (b, c, a), (c, a, b)] {
                if !bpa.directed_edges.contains(&(v, u))
                    && let Some((_, center)) = ball_center_triangle(pts, normals, u, v, opp, rho)
                {
                    active_front.push_back((u, v, opp, center));
                }
            }
        }

        for seed_i in 0..n {
            // Advance current front queue before trying new disconnected seed triangles.
            while let Some((i, j, k, c_old)) = active_front.pop_front() {
                if bpa.directed_edges.contains(&(j, i)) {
                    continue;
                }
                if let Some((m, c_new)) =
                    find_pivot_candidate(&bctx, (i, j, k), c_old, &bpa, &mut nbr_buf)
                {
                    bpa.emit(j, i, m);
                    active_front.push_back((i, m, j, c_new));
                    active_front.push_back((m, j, i, c_new));
                }
            }

            if bpa.vert_tri_count[seed_i] > 0 {
                continue;
            }
            if let Some((b, c, c_ball)) = find_seed_triangle(&bctx, seed_i, &bpa, &mut nbr_buf) {
                bpa.emit(seed_i, b, c);
                active_front.push_back((seed_i, b, c, c_ball));
                active_front.push_back((b, c, seed_i, c_ball));
                active_front.push_back((c, seed_i, b, c_ball));
            }
        }

        // Drain any remaining front queued from the final seed.
        while let Some((i, j, k, c_old)) = active_front.pop_front() {
            if bpa.directed_edges.contains(&(j, i)) {
                continue;
            }
            if let Some((m, c_new)) =
                find_pivot_candidate(&bctx, (i, j, k), c_old, &bpa, &mut nbr_buf)
            {
                bpa.emit(j, i, m);
                active_front.push_back((i, m, j, c_new));
                active_front.push_back((m, j, i, c_new));
            }
        }
    }

    if bpa.tris.is_empty() {
        return Err(OpsError::InvalidInput(
            "ball pivoting could not form any valid triangles at specified radius",
        ));
    }

    // Compact vertex buffer to vertices actually referenced by generated triangles.
    let mut remap = vec![u32::MAX; n];
    let mut out_pos = Vec::new();
    let mut out_idx = Vec::with_capacity(bpa.tris.len() * 3);
    for [a, b, c] in bpa.tris {
        for v in [a, b, c] {
            if remap[v] == u32::MAX {
                remap[v] = (out_pos.len() / 3) as u32;
                let p = raw_positions[v];
                out_pos.push(p[0]);
                out_pos.push(p[1]);
                out_pos.push(p[2]);
            }
            out_idx.push(remap[v]);
        }
    }

    Ok(Mesh::from_buffers(&out_pos, &out_idx)?)
}

struct BpaCtx<'a> {
    pts: &'a [[f64; 3]],
    normals: &'a [[f64; 3]],
    tree: &'a KdTree,
    rho: f64,
    search_r2: f64,
    empty_r2: f64,
}

struct BpaMesh {
    tris: Vec<[usize; 3]>,
    directed_edges: HashSet<(usize, usize)>,
    undirected_counts: HashMap<(usize, usize), u8>,
    vert_tri_count: Vec<usize>,
    vert_open_edges: Vec<usize>,
}

impl BpaMesh {
    fn new(n: usize) -> Self {
        Self {
            tris: Vec::new(),
            directed_edges: HashSet::new(),
            undirected_counts: HashMap::new(),
            vert_tri_count: vec![0; n],
            vert_open_edges: vec![0; n],
        }
    }

    fn can_add(&self, a: usize, b: usize, c: usize) -> bool {
        if a == b || b == c || c == a {
            return false;
        }
        for v in [a, b, c] {
            if self.vert_tri_count[v] > 0 && self.vert_open_edges[v] == 0 {
                return false;
            }
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            if self.directed_edges.contains(&(u, v)) {
                return false;
            }
            if self
                .undirected_counts
                .get(&edge_key(u, v))
                .copied()
                .unwrap_or(0)
                >= 2
            {
                return false;
            }
        }
        true
    }

    fn emit(&mut self, a: usize, b: usize, c: usize) {
        self.tris.push([a, b, c]);
        for (u, v) in [(a, b), (b, c), (c, a)] {
            self.directed_edges.insert((u, v));
            *self.undirected_counts.entry(edge_key(u, v)).or_insert(0) += 1;
            if self.directed_edges.contains(&(v, u)) {
                self.vert_open_edges[u] = self.vert_open_edges[u].saturating_sub(1);
                self.vert_open_edges[v] = self.vert_open_edges[v].saturating_sub(1);
            } else {
                self.vert_open_edges[u] += 1;
                self.vert_open_edges[v] += 1;
            }
        }
        self.vert_tri_count[a] += 1;
        self.vert_tri_count[b] += 1;
        self.vert_tri_count[c] += 1;
    }
}

fn find_seed_triangle(
    bctx: &BpaCtx<'_>,
    a: usize,
    bpa: &BpaMesh,
    nbr_buf: &mut Vec<(f64, usize)>,
) -> Option<(usize, usize, [f64; 3])> {
    nbr_buf.clear();
    bctx.tree
        .radius_search(bctx.pts, bctx.pts[a], Some(a), bctx.search_r2, nbr_buf);
    nbr_buf.sort_by(|x, y| x.0.total_cmp(&y.0));
    let cand: Vec<usize> = nbr_buf.iter().take(24).map(|&(_, idx)| idx).collect();

    let mut empty_check_buf = Vec::new();
    for (bi, &b) in cand.iter().enumerate() {
        for &c in &cand[(bi + 1)..] {
            let Some((oriented_bc, c_ball)) =
                ball_center_triangle(bctx.pts, bctx.normals, a, b, c, bctx.rho)
            else {
                continue;
            };
            let (u, v) = if oriented_bc { (b, c) } else { (c, b) };
            if !bpa.can_add(a, u, v) {
                continue;
            }
            empty_check_buf.clear();
            bctx.tree
                .radius_search(bctx.pts, c_ball, None, bctx.empty_r2, &mut empty_check_buf);
            let blocked = empty_check_buf
                .iter()
                .any(|&(_, q)| q != a && q != u && q != v);
            if !blocked {
                return Some((u, v, c_ball));
            }
        }
    }
    None
}

fn find_pivot_candidate(
    bctx: &BpaCtx<'_>,
    tri: (usize, usize, usize),
    c_old: [f64; 3],
    bpa: &BpaMesh,
    nbr_buf: &mut Vec<(f64, usize)>,
) -> Option<(usize, [f64; 3])> {
    let (i, j, k) = tri;
    let pts = bctx.pts;
    let normals = bctx.normals;
    let rho = bctx.rho;

    let m_edge = [
        0.5 * (pts[i][0] + pts[j][0]),
        0.5 * (pts[i][1] + pts[j][1]),
        0.5 * (pts[i][2] + pts[j][2]),
    ];
    let e_dir = unit(sub(pts[j], pts[i]))?;
    let r_vec = sub(c_old, m_edge);
    let r_pivot = norm(r_vec);
    if r_pivot <= 1e-12 {
        return None;
    }
    let u_piv = [r_vec[0] / r_pivot, r_vec[1] / r_pivot, r_vec[2] / r_pivot];
    let v_piv = unit(cross(e_dir, u_piv))?;

    // Search points potentially reachable by pivoting the rho-ball centered near m_edge.
    nbr_buf.clear();
    let query_r2 = (r_pivot + rho) * (r_pivot + rho);
    bctx.tree
        .radius_search(pts, m_edge, None, query_r2.max(bctx.search_r2), nbr_buf);

    let mut best: Option<(f64, usize, [f64; 3])> = None;
    let mut empty_check_buf = Vec::new();
    let tau = std::f64::consts::TAU;

    for &(_, m_idx) in nbr_buf.iter() {
        if m_idx == i || m_idx == j || m_idx == k {
            continue;
        }
        // New triangle will be (j, i, m_idx).
        if !bpa.can_add(j, i, m_idx) {
            continue;
        }
        let pm = pts[m_idx];
        let q = sub(pm, m_edge);
        let q_par = dot(q, e_dir);
        let a = dot(q, u_piv);
        let b = dot(q, v_piv);
        let d = a.hypot(b);
        if d <= 1e-12 {
            continue;
        }
        let rhs = (r_pivot * r_pivot + a * a + b * b + q_par * q_par - rho * rho) / (2.0 * r_pivot);
        if rhs.abs() > d * (1.0 + 1e-10) {
            continue;
        }
        let phi = b.atan2(a);
        let alpha = (rhs / d).clamp(-1.0, 1.0).acos();
        let mut theta = (phi - alpha).rem_euclid(tau);
        if theta <= 1e-7 {
            theta += tau;
        }
        if theta >= tau - 1e-7 {
            continue;
        }
        if let Some((best_theta, _, _)) = best
            && theta >= best_theta
        {
            continue;
        }

        let (sin_t, cos_t) = theta.sin_cos();
        let c_cand = [
            m_edge[0] + r_pivot * (cos_t * u_piv[0] + sin_t * v_piv[0]),
            m_edge[1] + r_pivot * (cos_t * u_piv[1] + sin_t * v_piv[1]),
            m_edge[2] + r_pivot * (cos_t * u_piv[2] + sin_t * v_piv[2]),
        ];

        // Check outward triangle orientation vs vertex normals and sphere center side.
        let tri_n = match unit(cross(sub(pts[i], pts[j]), sub(pm, pts[j]))) {
            Some(tn) => tn,
            None => continue,
        };
        let n_avg = add(add(normals[j], normals[i]), normals[m_idx]);
        if dot(tri_n, n_avg) <= 0.0 || dot(tri_n, sub(c_cand, m_edge)) <= 0.0 {
            continue;
        }

        empty_check_buf.clear();
        bctx.tree
            .radius_search(pts, c_cand, None, bctx.empty_r2, &mut empty_check_buf);
        let blocked = empty_check_buf
            .iter()
            .any(|&(_, q_idx)| q_idx != i && q_idx != j && q_idx != m_idx);
        if !blocked {
            best = Some((theta, m_idx, c_cand));
        }
    }

    best.map(|(_, m_idx, c_new)| (m_idx, c_new))
}

fn ball_center_triangle(
    pts: &[[f64; 3]],
    normals: &[[f64; 3]],
    a: usize,
    b: usize,
    c: usize,
    rho: f64,
) -> Option<(bool, [f64; 3])> {
    let pa = pts[a];
    let e1 = sub(pts[b], pa);
    let e2 = sub(pts[c], pa);
    let cross_e = cross(e1, e2);
    let cross_sq = dot(cross_e, cross_e);
    if cross_sq <= 1e-18 {
        return None;
    }

    let n_avg = add(add(normals[a], normals[b]), normals[c]);
    let oriented_bc = dot(cross_e, n_avg) >= 0.0;
    let raw_n = cross_e;
    let cross_len = cross_sq.sqrt();
    let tri_n = [
        raw_n[0] / cross_len,
        raw_n[1] / cross_len,
        raw_n[2] / cross_len,
    ];
    let out_n = if oriented_bc { tri_n } else { negate(tri_n) };

    let e1_sq = dot(e1, e1);
    let e2_sq = dot(e2, e2);
    let num_a = [
        e1_sq * e2[0] - e2_sq * e1[0],
        e1_sq * e2[1] - e2_sq * e1[1],
        e1_sq * e2[2] - e2_sq * e1[2],
    ];
    let d_vec = cross(num_a, cross_e);
    let inv_2c = 1.0 / (2.0 * cross_sq);
    let offset = [d_vec[0] * inv_2c, d_vec[1] * inv_2c, d_vec[2] * inv_2c];
    let r_circ_sq = dot(offset, offset);
    if r_circ_sq > rho * rho * (1.0 - 1e-7) {
        return None;
    }

    let h = (rho * rho - r_circ_sq).max(0.0).sqrt();
    let circumcenter = add(pa, offset);
    let center = [
        circumcenter[0] + h * out_n[0],
        circumcenter[1] + h * out_n[1],
        circumcenter[2] + h * out_n[2],
    ];
    Some((oriented_bc, center))
}

fn edge_key(u: usize, v: usize) -> (usize, usize) {
    if u < v { (u, v) } else { (v, u) }
}

// ============================================================================
// Validation, Flat KD-Tree & Spatial Utilities
// ============================================================================

fn validate_cloud(cloud: &PointCloud) -> Result<()> {
    let n = cloud.positions.len();
    for &p in &cloud.positions {
        if !p.iter().all(|v| v.is_finite()) {
            return Err(OpsError::InvalidInput(
                "point cloud contains non-finite position",
            ));
        }
    }
    if let Some(nrms) = &cloud.normals {
        if nrms.len() != n {
            return Err(OpsError::InvalidInput(
                "point cloud normals length must match positions",
            ));
        }
        for &nv in nrms {
            if !nv.iter().all(|v| v.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "point cloud contains non-finite normal",
                ));
            }
        }
    }
    if let Some(cols) = &cloud.colors {
        if cols.len() != n {
            return Err(OpsError::InvalidInput(
                "point cloud colors length must match positions",
            ));
        }
        for &cv in cols {
            if !cv.iter().all(|v| v.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "point cloud contains non-finite color",
                ));
            }
        }
    }
    Ok(())
}

fn subset_cloud(cloud: &PointCloud, indices: &[usize]) -> PointCloud {
    let positions = indices.iter().map(|&i| cloud.positions[i]).collect();
    let normals = cloud
        .normals
        .as_ref()
        .map(|v| indices.iter().map(|&i| v[i]).collect());
    let colors = cloud
        .colors
        .as_ref()
        .map(|v| indices.iter().map(|&i| v[i]).collect());
    PointCloud {
        positions,
        normals,
        colors,
    }
}

fn to_f64_points(pts: &[[f32; 3]]) -> Vec<[f64; 3]> {
    pts.iter()
        .map(|&p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        .collect()
}

fn normalize_or_default(v: [f32; 3]) -> [f32; 3] {
    let d = unit([f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).unwrap_or([0.0, 0.0, 1.0]);
    [d[0] as f32, d[1] as f32, d[2] as f32]
}

fn negate(v: [f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

fn median_nn_spacing(pts: &[[f64; 3]], tree: &KdTree) -> f64 {
    if pts.len() < 2 {
        return 1.0;
    }
    let mut heap = Vec::with_capacity(2);
    let mut dists = Vec::with_capacity(pts.len());
    for (i, &p) in pts.iter().enumerate() {
        heap.clear();
        tree.knn(pts, p, Some(i), 1, &mut heap);
        if let Some(&(d2, _)) = heap.first()
            && d2 > 0.0
        {
            dists.push(d2.sqrt());
        }
    }
    if dists.is_empty() {
        return 1.0;
    }
    dists.sort_by(f64::total_cmp);
    dists[dists.len() / 2]
}

struct KdNode {
    point_idx: usize,
    axis: u8,
    left: usize,
    right: usize,
}

struct KdTree {
    nodes: Vec<KdNode>,
    root: usize,
}

impl KdTree {
    fn build(pts: &[[f64; 3]]) -> Self {
        let mut order: Vec<usize> = (0..pts.len()).collect();
        let mut nodes = Vec::with_capacity(pts.len());
        let root = Self::build_rec(pts, &mut order, &mut nodes);
        Self { nodes, root }
    }

    fn build_rec(pts: &[[f64; 3]], slice: &mut [usize], nodes: &mut Vec<KdNode>) -> usize {
        if slice.is_empty() {
            return usize::MAX;
        }
        let (mut bmin, mut bmax) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &idx in slice.iter() {
            let p = pts[idx];
            for c in 0..3 {
                bmin[c] = bmin[c].min(p[c]);
                bmax[c] = bmax[c].max(p[c]);
            }
        }
        let ext = sub(bmax, bmin);
        let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
            0usize
        } else if ext[1] >= ext[2] {
            1usize
        } else {
            2usize
        };
        let mid = slice.len() / 2;
        slice.select_nth_unstable_by(mid, |&a, &b| pts[a][axis].total_cmp(&pts[b][axis]));
        let point_idx = slice[mid];
        let node_id = nodes.len();
        nodes.push(KdNode {
            point_idx,
            axis: axis as u8,
            left: usize::MAX,
            right: usize::MAX,
        });
        let (left_s, right_with_mid) = slice.split_at_mut(mid);
        let left = Self::build_rec(pts, left_s, nodes);
        let right = Self::build_rec(pts, &mut right_with_mid[1..], nodes);
        nodes[node_id].left = left;
        nodes[node_id].right = right;
        node_id
    }

    fn knn(
        &self,
        pts: &[[f64; 3]],
        target: [f64; 3],
        skip: Option<usize>,
        k: usize,
        best: &mut Vec<(f64, usize)>,
    ) {
        self.knn_rec(pts, self.root, target, skip, k, best);
    }

    fn knn_rec(
        &self,
        pts: &[[f64; 3]],
        node_id: usize,
        target: [f64; 3],
        skip: Option<usize>,
        k: usize,
        best: &mut Vec<(f64, usize)>,
    ) {
        if node_id == usize::MAX || k == 0 {
            return;
        }
        let node = &self.nodes[node_id];
        let p = pts[node.point_idx];
        if skip != Some(node.point_idx) {
            let d = sub(p, target);
            let d2 = dot(d, d);
            if best.len() < k {
                best.push((d2, node.point_idx));
                if best.len() == k {
                    best.sort_by(|a, b| a.0.total_cmp(&b.0));
                }
            } else if d2 < best[k - 1].0 {
                best[k - 1] = (d2, node.point_idx);
                let mut pos = k - 1;
                while pos > 0 && best[pos].0 < best[pos - 1].0 {
                    best.swap(pos, pos - 1);
                    pos -= 1;
                }
            }
        }
        let axis = usize::from(node.axis);
        let diff = target[axis] - p[axis];
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        self.knn_rec(pts, near, target, skip, k, best);
        let worst = if best.len() < k {
            f64::INFINITY
        } else {
            best[k - 1].0
        };
        if diff * diff < worst {
            self.knn_rec(pts, far, target, skip, k, best);
        }
    }

    fn radius_search(
        &self,
        pts: &[[f64; 3]],
        target: [f64; 3],
        skip: Option<usize>,
        r2: f64,
        out: &mut Vec<(f64, usize)>,
    ) {
        self.radius_rec(pts, self.root, target, skip, r2, out);
    }

    fn radius_rec(
        &self,
        pts: &[[f64; 3]],
        node_id: usize,
        target: [f64; 3],
        skip: Option<usize>,
        r2: f64,
        out: &mut Vec<(f64, usize)>,
    ) {
        if node_id == usize::MAX {
            return;
        }
        let node = &self.nodes[node_id];
        let p = pts[node.point_idx];
        if skip != Some(node.point_idx) {
            let d = sub(p, target);
            let d2 = dot(d, d);
            if d2 <= r2 {
                out.push((d2, node.point_idx));
            }
        }
        let axis = usize::from(node.axis);
        let diff = target[axis] - p[axis];
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        self.radius_rec(pts, near, target, skip, r2, out);
        if diff * diff <= r2 {
            self.radius_rec(pts, far, target, skip, r2, out);
        }
    }
}
