//! Stroke-start rest position and normal channels for multi-dab brushes.
//!
//! [`capture_stroke_state`] snapshots live vertex positions and smooth normals
//! into two width-3 [`ChannelDesc::bound_to_topology`] source channels
//! ([`STROKE_POS_CHANNEL`] and [`STROKE_NORMAL_CHANNEL`]). Because both
//! channels are source channels with [`TopologyPolicy::Interpolate`](fast_mesh_store::TopologyPolicy::Interpolate)
//! and `bound_to_topology`, [`split_stretched`](crate::stretch::split_stretched)
//! automatically interpolates parent stroke values onto newly split midpoints
//! while any other topology edit invalidates them undoably.

use fast_mesh::compute_vertex_normals;
use fast_mesh_store::{ChannelDesc, Mesh, VertexId};

use crate::{OpsError, Result, allocation_allowed, reserve};

/// Topology-bound width-3 source channel storing stroke-start rest positions.
pub const STROKE_POS_CHANNEL: &str = "stroke_rest_pos";

/// Topology-bound width-3 source channel storing stroke-start smooth normals.
pub const STROKE_NORMAL_CHANNEL: &str = "stroke_rest_normal";

/// Explicit slot-aligned stroke-start rest buffers (e.g. from a scene's lazy
/// stroke cache). Any out-of-bounds or non-finite entry falls back to the mesh
/// stroke channels and then to the current mesh state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeSlice<'a> {
    /// Slot-aligned rest positions (`[f32::NAN; 3]` when uncaptured).
    pub positions: &'a [[f32; 3]],
    /// Slot-aligned rest smooth normals (`[f32::NAN; 3]` when uncaptured).
    pub normals: &'a [[f32; 3]],
}

/// Captures every live vertex's current position and smooth normal into
/// [`STROKE_POS_CHANNEL`] and [`STROKE_NORMAL_CHANNEL`], replacing any prior
/// stroke capture.
///
/// This is metadata setup rather than a mesh edit: it records no undo step and
/// does not bump the mesh's revision.
///
/// # Errors
/// - [`OpsError::InvalidInput`] if the mesh buffers fail validation.
/// - [`OpsError::Store`] if memory allocation fails.
pub fn capture_stroke_state(mesh: &mut Mesh) -> Result<()> {
    let slots = mesh.vertex_slots();
    let positions = mesh.positions();
    let mut flat: Vec<f32> = Vec::new();
    reserve(&mut flat, slots * 3)?;
    flat.extend(positions.iter().flatten().copied());

    let mut indices: Vec<u32> = Vec::new();
    reserve(&mut indices, mesh.triangle_slots() * 3)?;
    for t in 0..mesh.triangle_slots() as u32 {
        if let Some(tri) = mesh.triangle(t) {
            indices.extend(tri);
        }
    }

    allocation_allowed()?;
    let mut normals = compute_vertex_normals(&flat, &indices)
        .map_err(|_| OpsError::InvalidInput("mesh buffers are not a valid mesh"))?;

    for v in 0..slots {
        if !mesh.is_vertex_alive(v as VertexId) {
            let base = 3 * v;
            flat[base..base + 3].fill(f32::NAN);
            normals[base..base + 3].fill(f32::NAN);
        }
    }

    // Save any prior stroke channels fallibly before mutating so OOM leaves prior state intact.
    let old_pos = if let Some(ch) = mesh.channel(STROKE_POS_CHANNEL) {
        let mut copy = Vec::new();
        reserve(&mut copy, ch.len())?;
        copy.extend_from_slice(ch);
        Some(copy)
    } else {
        None
    };
    let old_nrm = if let Some(ch) = mesh.channel(STROKE_NORMAL_CHANNEL) {
        let mut copy = Vec::new();
        reserve(&mut copy, ch.len())?;
        copy.extend_from_slice(ch);
        Some(copy)
    } else {
        None
    };

    allocation_allowed()?;
    if old_pos.is_some() {
        mesh.remove_channel(STROKE_POS_CHANNEL)?;
    }
    if old_nrm.is_some() {
        mesh.remove_channel(STROKE_NORMAL_CHANNEL)?;
    }
    let pos_desc = || {
        ChannelDesc::source(STROKE_POS_CHANNEL, 3)
            .bound_to_topology()
            .with_default(f32::NAN)
    };
    let nrm_desc = || {
        ChannelDesc::source(STROKE_NORMAL_CHANNEL, 3)
            .bound_to_topology()
            .with_default(f32::NAN)
    };
    if let Err(err) = mesh.add_channel_with_values(pos_desc(), &flat) {
        if let Some(prev) = &old_pos {
            let _ = mesh.add_channel_with_values(pos_desc(), prev);
        }
        if let Some(prev) = &old_nrm {
            let _ = mesh.add_channel_with_values(nrm_desc(), prev);
        }
        return Err(err.into());
    }
    if let Err(err) = mesh.add_channel_with_values(nrm_desc(), &normals) {
        let _ = mesh.remove_channel(STROKE_POS_CHANNEL);
        if let Some(prev) = &old_pos {
            let _ = mesh.add_channel_with_values(pos_desc(), prev);
        }
        if let Some(prev) = &old_nrm {
            let _ = mesh.add_channel_with_values(nrm_desc(), prev);
        }
        return Err(err.into());
    }
    Ok(())
}

/// Calls [`capture_stroke_state`] unless both [`STROKE_POS_CHANNEL`] and
/// [`STROKE_NORMAL_CHANNEL`] are already present on `mesh`.
///
/// # Errors
/// As for [`capture_stroke_state`].
pub fn ensure_stroke_state(mesh: &mut Mesh) -> Result<()> {
    if mesh.channel(STROKE_POS_CHANNEL).is_some() && mesh.channel(STROKE_NORMAL_CHANNEL).is_some() {
        return Ok(());
    }
    capture_stroke_state(mesh)
}

/// Reads vertex `v`'s stroke-start rest position from `stroke` or
/// [`STROKE_POS_CHANNEL`], falling back to its current position when absent or
/// non-finite.
pub(crate) fn stroke_position_at(
    mesh: &Mesh,
    stroke: Option<StrokeSlice<'_>>,
    v: VertexId,
) -> [f32; 3] {
    let idx = v as usize;
    if let Some(s) = stroke
        && let Some(&p) = s.positions.get(idx)
        && p.iter().all(|x| x.is_finite())
    {
        return p;
    }
    if let Some(ch) = mesh.channel(STROKE_POS_CHANNEL)
        && let Some(slice) = ch.get(3 * idx..3 * idx + 3)
    {
        let p = [slice[0], slice[1], slice[2]];
        if p.iter().all(|x| x.is_finite()) {
            return p;
        }
    }
    mesh.positions()[idx]
}

/// Reads vertex `v`'s stroke-start smooth normal from `stroke` or
/// [`STROKE_NORMAL_CHANNEL`], re-normalizing interpolated non-unit vectors (such
/// as midpoints created by [`split_stretched`](crate::stretch::split_stretched))
/// and falling back to `fallback` when absent or non-finite.
pub(crate) fn stroke_normal_at(
    mesh: &Mesh,
    stroke: Option<StrokeSlice<'_>>,
    v: VertexId,
    fallback: [f32; 3],
) -> [f32; 3] {
    let idx = v as usize;
    let raw = if let Some(s) = stroke
        && let Some(&n) = s.normals.get(idx)
        && n.iter().all(|x| x.is_finite())
    {
        Some(n)
    } else if let Some(ch) = mesh.channel(STROKE_NORMAL_CHANNEL)
        && let Some(slice) = ch.get(3 * idx..3 * idx + 3)
    {
        let n = [slice[0], slice[1], slice[2]];
        n.iter().all(|x| x.is_finite()).then_some(n)
    } else {
        None
    };

    let Some(n) = raw else {
        return fallback;
    };
    let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if len_sq > 0.0 && (len_sq - 1.0).abs() > 1e-5 {
        let len = len_sq.sqrt();
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        n
    }
}
