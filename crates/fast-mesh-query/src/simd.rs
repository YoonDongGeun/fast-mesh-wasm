//! 4-wide SoA and Wasm SIMD128 leaf triangle batch kernels.

use crate::tri::{Cull, Feature, RayPre, TriClosest, TriHit, closest_on_triangle};
use fast_mesh_store::{Mesh, TriangleId};

/// Four triangles stored in Structure-of-Arrays form (`a, b, c` corners x `x, y, z`),
/// 16-byte aligned for single-instruction `v128_load` on `wasm32 +simd128`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tri4 {
    pub ax: [f32; 4],
    pub ay: [f32; 4],
    pub az: [f32; 4],
    pub bx: [f32; 4],
    pub by: [f32; 4],
    pub bz: [f32; 4],
    pub cx: [f32; 4],
    pub cy: [f32; 4],
    pub cz: [f32; 4],
}

impl Tri4 {
    /// Packs four AoS triangles `[[[f32; 3]; 3]; 4]` into 16-byte aligned SoA form.
    #[cfg(test)]
    #[inline]
    pub fn from_array(tris: &[[[f32; 3]; 3]; 4]) -> Self {
        Self {
            ax: [tris[0][0][0], tris[1][0][0], tris[2][0][0], tris[3][0][0]],
            ay: [tris[0][0][1], tris[1][0][1], tris[2][0][1], tris[3][0][1]],
            az: [tris[0][0][2], tris[1][0][2], tris[2][0][2], tris[3][0][2]],
            bx: [tris[0][1][0], tris[1][1][0], tris[2][1][0], tris[3][1][0]],
            by: [tris[0][1][1], tris[1][1][1], tris[2][1][1], tris[3][1][1]],
            bz: [tris[0][1][2], tris[1][1][2], tris[2][1][2], tris[3][1][2]],
            cx: [tris[0][2][0], tris[1][2][0], tris[2][2][0], tris[3][2][0]],
            cy: [tris[0][2][1], tris[1][2][1], tris[2][2][1], tris[3][2][1]],
            cz: [tris[0][2][2], tris[1][2][2], tris[2][2][2], tris[3][2][2]],
        }
    }

    /// Packs four triangles directly from mesh buffers.
    #[inline]
    pub fn from_mesh(mesh: &Mesh, ids: [TriangleId; 4]) -> Self {
        let positions = mesh.positions();
        let triangles = mesh.triangles();
        let t0 = triangles[ids[0] as usize];
        let t1 = triangles[ids[1] as usize];
        let t2 = triangles[ids[2] as usize];
        let t3 = triangles[ids[3] as usize];
        let p00 = positions[t0[0] as usize];
        let p01 = positions[t0[1] as usize];
        let p02 = positions[t0[2] as usize];
        let p10 = positions[t1[0] as usize];
        let p11 = positions[t1[1] as usize];
        let p12 = positions[t1[2] as usize];
        let p20 = positions[t2[0] as usize];
        let p21 = positions[t2[1] as usize];
        let p22 = positions[t2[2] as usize];
        let p30 = positions[t3[0] as usize];
        let p31 = positions[t3[1] as usize];
        let p32 = positions[t3[2] as usize];
        Self {
            ax: [p00[0], p10[0], p20[0], p30[0]],
            ay: [p00[1], p10[1], p20[1], p30[1]],
            az: [p00[2], p10[2], p20[2], p30[2]],
            bx: [p01[0], p11[0], p21[0], p31[0]],
            by: [p01[1], p11[1], p21[1], p31[1]],
            bz: [p01[2], p11[2], p21[2], p31[2]],
            cx: [p02[0], p12[0], p22[0], p32[0]],
            cy: [p02[1], p12[1], p22[1], p32[1]],
            cz: [p02[2], p12[2], p22[2], p32[2]],
        }
    }

    #[inline]
    pub fn lane(&self, i: usize) -> [[f32; 3]; 3] {
        [
            [self.ax[i], self.ay[i], self.az[i]],
            [self.bx[i], self.by[i], self.bz[i]],
            [self.cx[i], self.cy[i], self.cz[i]],
        ]
    }

    #[inline]
    fn coord_arrays(&self) -> (CoordLanes<'_>, CoordLanes<'_>, CoordLanes<'_>) {
        (
            [&self.ax, &self.ay, &self.az],
            [&self.bx, &self.by, &self.bz],
            [&self.cx, &self.cy, &self.cz],
        )
    }
}

type CoordLanes<'a> = [&'a [f32; 4]; 3];

/// Watertight Woop-Benthin-Wald 4-triangle ray intersection kernel.
#[inline]
pub(crate) fn ray_triangles4(
    pre: &RayPre,
    tris: &Tri4,
    cull: Cull,
    min_t: f32,
    max_t: f32,
) -> [Option<TriHit>; 4] {
    ray_triangles4_impl(pre, tris, cull, min_t, max_t)
}

/// 4-triangle closest-point Ericson + sliver-segment evaluation matching
/// [`closest_on_triangle`].
#[inline]
pub(crate) fn closest_on_triangles4(q: [f32; 3], tris: &Tri4) -> [TriClosest<f32>; 4] {
    closest_on_triangles4_impl(q, tris)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn decode_feature(tag: f32) -> Feature {
    let code = tag as u32;
    match code {
        0 => Feature::Vertex(0),
        1 => Feature::Vertex(1),
        2 => Feature::Vertex(2),
        3 => Feature::Edge(0),
        4 => Feature::Edge(1),
        5 => Feature::Edge(2),
        _ => Feature::Face,
    }
}

#[inline]
fn ray_f64_edge_fallback(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> (f32, f32, f32) {
    let (ax, ay, bx, by, cx, cy) = (
        f64::from(ax),
        f64::from(ay),
        f64::from(bx),
        f64::from(by),
        f64::from(cx),
        f64::from(cy),
    );
    (
        (cx * by - cy * bx) as f32,
        (ax * cy - ay * cx) as f32,
        (bx * ay - by * ax) as f32,
    )
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod backend {
    use super::*;
    use core::arch::wasm32::*;

    #[inline(always)]
    unsafe fn load(p: &[f32; 4]) -> v128 {
        unsafe { v128_load(p.as_ptr().cast::<v128>()) }
    }

    #[inline(always)]
    unsafe fn store(dst: &mut [f32; 4], v: v128) {
        unsafe { v128_store(dst.as_mut_ptr().cast::<v128>(), v) }
    }

    #[inline(always)]
    fn dot4(ax: v128, ay: v128, az: v128, bx: v128, by: v128, bz: v128) -> v128 {
        f32x4_add(
            f32x4_add(f32x4_mul(ax, bx), f32x4_mul(ay, by)),
            f32x4_mul(az, bz),
        )
    }

    pub fn ray_triangles4_impl(
        pre: &RayPre,
        tris: &Tri4,
        cull: Cull,
        min_t: f32,
        max_t: f32,
    ) -> [Option<TriHit>; 4] {
        let (a_arr, b_arr, c_arr) = tris.coord_arrays();
        let (kx, ky, kz) = (pre.kx, pre.ky, pre.kz);

        unsafe {
            let ox = f32x4_splat(pre.origin[kx]);
            let oy = f32x4_splat(pre.origin[ky]);
            let oz = f32x4_splat(pre.origin[kz]);

            let akx = f32x4_sub(load(a_arr[kx]), ox);
            let aky = f32x4_sub(load(a_arr[ky]), oy);
            let akz = f32x4_sub(load(a_arr[kz]), oz);
            let bkx = f32x4_sub(load(b_arr[kx]), ox);
            let bky = f32x4_sub(load(b_arr[ky]), oy);
            let bkz = f32x4_sub(load(b_arr[kz]), oz);
            let ckx = f32x4_sub(load(c_arr[kx]), ox);
            let cky = f32x4_sub(load(c_arr[ky]), oy);
            let ckz = f32x4_sub(load(c_arr[kz]), oz);

            let sx = f32x4_splat(pre.sx);
            let sy = f32x4_splat(pre.sy);
            let sz = f32x4_splat(pre.sz);

            let ax_v = f32x4_sub(akx, f32x4_mul(sx, akz));
            let ay_v = f32x4_sub(aky, f32x4_mul(sy, akz));
            let bx_v = f32x4_sub(bkx, f32x4_mul(sx, bkz));
            let by_v = f32x4_sub(bky, f32x4_mul(sy, bkz));
            let cx_v = f32x4_sub(ckx, f32x4_mul(sx, ckz));
            let cy_v = f32x4_sub(cky, f32x4_mul(sy, ckz));

            let mut u_v = f32x4_sub(f32x4_mul(cx_v, by_v), f32x4_mul(cy_v, bx_v));
            let mut v_v = f32x4_sub(f32x4_mul(ax_v, cy_v), f32x4_mul(ay_v, cx_v));
            let mut w_v = f32x4_sub(f32x4_mul(bx_v, ay_v), f32x4_mul(by_v, ax_v));

            let zero = f32x4_splat(0.0);
            let zmask = i32x4_bitmask(v128_or(
                v128_or(f32x4_eq(u_v, zero), f32x4_eq(v_v, zero)),
                f32x4_eq(w_v, zero),
            )) & 0x0F;

            let mut u = [0.0f32; 4];
            let mut v = [0.0f32; 4];
            let mut w = [0.0f32; 4];
            store(&mut u, u_v);
            store(&mut v, v_v);
            store(&mut w, w_v);

            if zmask != 0 {
                let mut ax_s = [0.0f32; 4];
                let mut ay_s = [0.0f32; 4];
                let mut bx_s = [0.0f32; 4];
                let mut by_s = [0.0f32; 4];
                let mut cx_s = [0.0f32; 4];
                let mut cy_s = [0.0f32; 4];
                store(&mut ax_s, ax_v);
                store(&mut ay_s, ay_v);
                store(&mut bx_s, bx_v);
                store(&mut by_s, by_v);
                store(&mut cx_s, cx_v);
                store(&mut cy_s, cy_v);
                for lane in 0..4 {
                    if (zmask & (1 << lane)) != 0 {
                        let (uu, vv, ww) = ray_f64_edge_fallback(
                            ax_s[lane], ay_s[lane], bx_s[lane], by_s[lane], cx_s[lane], cy_s[lane],
                        );
                        u[lane] = uu;
                        v[lane] = vv;
                        w[lane] = ww;
                    }
                }
                u_v = load(&u);
                v_v = load(&v);
                w_v = load(&w);
            }

            let pos = v128_and(
                v128_and(f32x4_ge(u_v, zero), f32x4_ge(v_v, zero)),
                f32x4_ge(w_v, zero),
            );
            let neg = v128_and(
                v128_and(f32x4_le(u_v, zero), f32x4_le(v_v, zero)),
                f32x4_le(w_v, zero),
            );
            let inside = match cull {
                Cull::None => v128_or(pos, neg),
                Cull::Back => pos,
                Cull::Front => neg,
            };
            let det_v = f32x4_add(f32x4_add(u_v, v_v), w_v);
            let valid_v = v128_and(inside, f32x4_ne(det_v, zero));
            let valid_mask = (i32x4_bitmask(valid_v) as u8) & 0x0F;
            if valid_mask == 0 {
                return [None, None, None, None];
            }

            let t_scaled = f32x4_add(
                f32x4_add(
                    f32x4_mul(u_v, f32x4_mul(sz, akz)),
                    f32x4_mul(v_v, f32x4_mul(sz, bkz)),
                ),
                f32x4_mul(w_v, f32x4_mul(sz, ckz)),
            );
            let t_v = f32x4_div(t_scaled, det_v);
            let inf = f32x4_splat(f32::INFINITY);
            let t_ok = v128_and(
                v128_and(
                    f32x4_lt(f32x4_abs(t_v), inf),
                    f32x4_le(f32x4_splat(min_t), t_v),
                ),
                f32x4_le(t_v, f32x4_splat(max_t)),
            );

            let bu_v = f32x4_div(u_v, det_v);
            let bv_v = f32x4_div(v_v, det_v);
            let bw_v = f32x4_div(w_v, det_v);
            let b_ok = v128_and(
                v128_and(
                    f32x4_lt(f32x4_abs(bu_v), inf),
                    f32x4_lt(f32x4_abs(bv_v), inf),
                ),
                f32x4_lt(f32x4_abs(bw_v), inf),
            );
            let hit_mask = (i32x4_bitmask(v128_and(v128_and(valid_v, t_ok), b_ok)) as u8) & 0x0F;
            if hit_mask == 0 {
                return [None, None, None, None];
            }

            let mut t = [0.0f32; 4];
            let mut bu = [0.0f32; 4];
            let mut bv = [0.0f32; 4];
            let mut bw = [0.0f32; 4];
            store(&mut t, t_v);
            store(&mut bu, bu_v);
            store(&mut bv, bv_v);
            store(&mut bw, bw_v);

            std::array::from_fn(|i| {
                ((hit_mask & (1 << i)) != 0).then_some(TriHit {
                    t: t[i],
                    barycentric: [bu[i], bv[i], bw[i]],
                })
            })
        }
    }

    pub fn closest_on_triangles4_impl(q: [f32; 3], tris: &Tri4) -> [TriClosest<f32>; 4] {
        unsafe {
            let ax = load(&tris.ax);
            let ay = load(&tris.ay);
            let az = load(&tris.az);
            let bx = load(&tris.bx);
            let by = load(&tris.by);
            let bz = load(&tris.bz);
            let cx = load(&tris.cx);
            let cy = load(&tris.cy);
            let cz = load(&tris.cz);

            let qx = f32x4_splat(q[0]);
            let qy = f32x4_splat(q[1]);
            let qz = f32x4_splat(q[2]);

            let abx = f32x4_sub(bx, ax);
            let aby = f32x4_sub(by, ay);
            let abz = f32x4_sub(bz, az);
            let acx = f32x4_sub(cx, ax);
            let acy = f32x4_sub(cy, ay);
            let acz = f32x4_sub(cz, az);

            let apx = f32x4_sub(qx, ax);
            let apy = f32x4_sub(qy, ay);
            let apz = f32x4_sub(qz, az);
            let bpx = f32x4_sub(qx, bx);
            let bpy = f32x4_sub(qy, by);
            let bpz = f32x4_sub(qz, bz);
            let cpx = f32x4_sub(qx, cx);
            let cpy = f32x4_sub(qy, cy);
            let cpz = f32x4_sub(qz, cz);

            let d1 = dot4(abx, aby, abz, apx, apy, apz);
            let d2 = dot4(acx, acy, acz, apx, apy, apz);
            let d3 = dot4(abx, aby, abz, bpx, bpy, bpz);
            let d4 = dot4(acx, acy, acz, bpx, bpy, bpz);
            let d5 = dot4(abx, aby, abz, cpx, cpy, cpz);
            let d6 = dot4(acx, acy, acz, cpx, cpy, cpz);

            let vc = f32x4_sub(f32x4_mul(d1, d4), f32x4_mul(d3, d2));
            let vb = f32x4_sub(f32x4_mul(d5, d2), f32x4_mul(d1, d6));
            let va = f32x4_sub(f32x4_mul(d3, d6), f32x4_mul(d5, d4));
            let sum = f32x4_add(f32x4_add(va, vb), vc);

            let ab2 = dot4(abx, aby, abz, abx, aby, abz);
            let ac2 = dot4(acx, acy, acz, acx, acy, acz);
            let sliver_lim = f32x4_mul(f32x4_mul(f32x4_splat(1e-5), ab2), ac2);
            let inf = f32x4_splat(f32::INFINITY);
            let sum_finite = f32x4_lt(f32x4_abs(sum), inf);
            let band = v128_or(v128_not(sum_finite), f32x4_le(sum, sliver_lim));
            let sliver_mask = (i32x4_bitmask(band) as u8) & 0x0F;

            let z = f32x4_splat(0.0);
            let one = f32x4_splat(1.0);

            let m_v0 = v128_and(f32x4_le(d1, z), f32x4_le(d2, z));
            let m_v1 = v128_and(v128_not(m_v0), v128_and(f32x4_ge(d3, z), f32x4_le(d4, d3)));
            let taken01 = v128_or(m_v0, m_v1);
            let m_e0 = v128_and(
                v128_not(taken01),
                v128_and(f32x4_le(vc, z), v128_and(f32x4_ge(d1, z), f32x4_le(d3, z))),
            );
            let taken012 = v128_or(taken01, m_e0);
            let m_v2 = v128_and(
                v128_not(taken012),
                v128_and(f32x4_ge(d6, z), f32x4_le(d5, d6)),
            );
            let taken0123 = v128_or(taken012, m_v2);
            let m_e2 = v128_and(
                v128_not(taken0123),
                v128_and(f32x4_le(vb, z), v128_and(f32x4_ge(d2, z), f32x4_le(d6, z))),
            );
            let taken01234 = v128_or(taken0123, m_e2);
            let d4_3 = f32x4_sub(d4, d3);
            let d5_6 = f32x4_sub(d5, d6);
            let m_e1 = v128_and(
                v128_not(taken01234),
                v128_and(
                    f32x4_le(va, z),
                    v128_and(f32x4_ge(d4_3, z), f32x4_ge(d5_6, z)),
                ),
            );

            // Start from Face values, then blend edge and vertex candidates.
            let vf = f32x4_div(vb, sum);
            let wf = f32x4_div(vc, sum);
            let mut px = f32x4_add(f32x4_add(ax, f32x4_mul(abx, vf)), f32x4_mul(acx, wf));
            let mut py = f32x4_add(f32x4_add(ay, f32x4_mul(aby, vf)), f32x4_mul(acy, wf));
            let mut pz = f32x4_add(f32x4_add(az, f32x4_mul(abz, vf)), f32x4_mul(acz, wf));
            let mut b0 = f32x4_sub(f32x4_sub(one, vf), wf);
            let mut b1 = vf;
            let mut b2 = wf;
            let mut feat = f32x4_splat(6.0);

            // Edge 1 (vertices 1 -> 2)
            let te1 = f32x4_div(d4_3, f32x4_add(d4_3, d5_6));
            let e1_x = f32x4_add(bx, f32x4_mul(te1, f32x4_sub(cx, bx)));
            let e1_y = f32x4_add(by, f32x4_mul(te1, f32x4_sub(cy, by)));
            let e1_z = f32x4_add(bz, f32x4_mul(te1, f32x4_sub(cz, bz)));
            px = v128_bitselect(e1_x, px, m_e1);
            py = v128_bitselect(e1_y, py, m_e1);
            pz = v128_bitselect(e1_z, pz, m_e1);
            b0 = v128_bitselect(z, b0, m_e1);
            b1 = v128_bitselect(f32x4_sub(one, te1), b1, m_e1);
            b2 = v128_bitselect(te1, b2, m_e1);
            feat = v128_bitselect(f32x4_splat(4.0), feat, m_e1);

            // Edge 2 (vertices 0 -> 2)
            let te2 = f32x4_div(d2, f32x4_sub(d2, d6));
            let e2_x = f32x4_add(ax, f32x4_mul(te2, acx));
            let e2_y = f32x4_add(ay, f32x4_mul(te2, acy));
            let e2_z = f32x4_add(az, f32x4_mul(te2, acz));
            px = v128_bitselect(e2_x, px, m_e2);
            py = v128_bitselect(e2_y, py, m_e2);
            pz = v128_bitselect(e2_z, pz, m_e2);
            b0 = v128_bitselect(f32x4_sub(one, te2), b0, m_e2);
            b1 = v128_bitselect(z, b1, m_e2);
            b2 = v128_bitselect(te2, b2, m_e2);
            feat = v128_bitselect(f32x4_splat(5.0), feat, m_e2);

            // Vertex 2
            px = v128_bitselect(cx, px, m_v2);
            py = v128_bitselect(cy, py, m_v2);
            pz = v128_bitselect(cz, pz, m_v2);
            b0 = v128_bitselect(z, b0, m_v2);
            b1 = v128_bitselect(z, b1, m_v2);
            b2 = v128_bitselect(one, b2, m_v2);
            feat = v128_bitselect(f32x4_splat(2.0), feat, m_v2);

            // Edge 0 (vertices 0 -> 1)
            let te0 = f32x4_div(d1, f32x4_sub(d1, d3));
            let e0_x = f32x4_add(ax, f32x4_mul(te0, abx));
            let e0_y = f32x4_add(ay, f32x4_mul(te0, aby));
            let e0_z = f32x4_add(az, f32x4_mul(te0, abz));
            px = v128_bitselect(e0_x, px, m_e0);
            py = v128_bitselect(e0_y, py, m_e0);
            pz = v128_bitselect(e0_z, pz, m_e0);
            b0 = v128_bitselect(f32x4_sub(one, te0), b0, m_e0);
            b1 = v128_bitselect(te0, b1, m_e0);
            b2 = v128_bitselect(z, b2, m_e0);
            feat = v128_bitselect(f32x4_splat(3.0), feat, m_e0);

            // Vertex 1
            px = v128_bitselect(bx, px, m_v1);
            py = v128_bitselect(by, py, m_v1);
            pz = v128_bitselect(bz, pz, m_v1);
            b0 = v128_bitselect(z, b0, m_v1);
            b1 = v128_bitselect(one, b1, m_v1);
            b2 = v128_bitselect(z, b2, m_v1);
            feat = v128_bitselect(one, feat, m_v1);

            // Vertex 0
            px = v128_bitselect(ax, px, m_v0);
            py = v128_bitselect(ay, py, m_v0);
            pz = v128_bitselect(az, pz, m_v0);
            b0 = v128_bitselect(one, b0, m_v0);
            b1 = v128_bitselect(z, b1, m_v0);
            b2 = v128_bitselect(z, b2, m_v0);
            feat = v128_bitselect(z, feat, m_v0);

            let dx = f32x4_sub(qx, px);
            let dy = f32x4_sub(qy, py);
            let dz = f32x4_sub(qz, pz);
            let d2_v = dot4(dx, dy, dz, dx, dy, dz);

            let mut out_px = [0.0f32; 4];
            let mut out_py = [0.0f32; 4];
            let mut out_pz = [0.0f32; 4];
            let mut out_d2 = [0.0f32; 4];
            let mut out_b0 = [0.0f32; 4];
            let mut out_b1 = [0.0f32; 4];
            let mut out_b2 = [0.0f32; 4];
            let mut out_feat = [0.0f32; 4];
            store(&mut out_px, px);
            store(&mut out_py, py);
            store(&mut out_pz, pz);
            store(&mut out_d2, d2_v);
            store(&mut out_b0, b0);
            store(&mut out_b1, b1);
            store(&mut out_b2, b2);
            store(&mut out_feat, feat);

            std::array::from_fn(|i| {
                if (sliver_mask & (1 << i)) != 0 {
                    closest_on_triangle(q, tris.lane(i))
                } else {
                    TriClosest {
                        point: [out_px[i], out_py[i], out_pz[i]],
                        dist2: out_d2[i],
                        barycentric: [out_b0[i], out_b1[i], out_b2[i]],
                        feature: decode_feature(out_feat[i]),
                    }
                }
            })
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod backend {
    use super::*;

    #[inline(always)]
    fn dot_soa(
        ax: &[f32; 4],
        ay: &[f32; 4],
        az: &[f32; 4],
        bx: &[f32; 4],
        by: &[f32; 4],
        bz: &[f32; 4],
    ) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = (ax[i] * bx[i] + ay[i] * by[i]) + az[i] * bz[i];
        }
        out
    }

    pub fn ray_triangles4_impl(
        pre: &RayPre,
        tris: &Tri4,
        cull: Cull,
        min_t: f32,
        max_t: f32,
    ) -> [Option<TriHit>; 4] {
        let (a_arr, b_arr, c_arr) = tris.coord_arrays();
        let (kx, ky, kz) = (pre.kx, pre.ky, pre.kz);
        let (ox, oy, oz) = (pre.origin[kx], pre.origin[ky], pre.origin[kz]);

        let mut akx = [0.0f32; 4];
        let mut aky = [0.0f32; 4];
        let mut akz = [0.0f32; 4];
        let mut bkx = [0.0f32; 4];
        let mut bky = [0.0f32; 4];
        let mut bkz = [0.0f32; 4];
        let mut ckx = [0.0f32; 4];
        let mut cky = [0.0f32; 4];
        let mut ckz = [0.0f32; 4];

        let mut ax = [0.0f32; 4];
        let mut ay = [0.0f32; 4];
        let mut bx = [0.0f32; 4];
        let mut by = [0.0f32; 4];
        let mut cx = [0.0f32; 4];
        let mut cy = [0.0f32; 4];

        let mut u = [0.0f32; 4];
        let mut v = [0.0f32; 4];
        let mut w = [0.0f32; 4];

        for i in 0..4 {
            akx[i] = a_arr[kx][i] - ox;
            aky[i] = a_arr[ky][i] - oy;
            akz[i] = a_arr[kz][i] - oz;
            bkx[i] = b_arr[kx][i] - ox;
            bky[i] = b_arr[ky][i] - oy;
            bkz[i] = b_arr[kz][i] - oz;
            ckx[i] = c_arr[kx][i] - ox;
            cky[i] = c_arr[ky][i] - oy;
            ckz[i] = c_arr[kz][i] - oz;

            ax[i] = akx[i] - pre.sx * akz[i];
            ay[i] = aky[i] - pre.sy * akz[i];
            bx[i] = bkx[i] - pre.sx * bkz[i];
            by[i] = bky[i] - pre.sy * bkz[i];
            cx[i] = ckx[i] - pre.sx * ckz[i];
            cy[i] = cky[i] - pre.sy * ckz[i];

            u[i] = cx[i] * by[i] - cy[i] * bx[i];
            v[i] = ax[i] * cy[i] - ay[i] * cx[i];
            w[i] = bx[i] * ay[i] - by[i] * ax[i];
        }

        for i in 0..4 {
            if u[i] == 0.0 || v[i] == 0.0 || w[i] == 0.0 {
                let (uu, vv, ww) = ray_f64_edge_fallback(ax[i], ay[i], bx[i], by[i], cx[i], cy[i]);
                u[i] = uu;
                v[i] = vv;
                w[i] = ww;
            }
        }

        let mut hits = [None; 4];
        for i in 0..4 {
            let inside = match cull {
                Cull::None => {
                    (u[i] >= 0.0 && v[i] >= 0.0 && w[i] >= 0.0)
                        || (u[i] <= 0.0 && v[i] <= 0.0 && w[i] <= 0.0)
                }
                Cull::Back => u[i] >= 0.0 && v[i] >= 0.0 && w[i] >= 0.0,
                Cull::Front => u[i] <= 0.0 && v[i] <= 0.0 && w[i] <= 0.0,
            };
            if !inside {
                continue;
            }
            let det = u[i] + v[i] + w[i];
            if det == 0.0 {
                continue;
            }
            let t_scaled =
                u[i] * (pre.sz * akz[i]) + v[i] * (pre.sz * bkz[i]) + w[i] * (pre.sz * ckz[i]);
            let t = t_scaled / det;
            if !(t.is_finite() && min_t <= t && t <= max_t) {
                continue;
            }
            let barycentric = [u[i] / det, v[i] / det, w[i] / det];
            if !barycentric.iter().all(|b| b.is_finite()) {
                continue;
            }
            hits[i] = Some(TriHit { t, barycentric });
        }
        hits
    }

    pub fn closest_on_triangles4_impl(q: [f32; 3], tris: &Tri4) -> [TriClosest<f32>; 4] {
        let mut abx = [0.0f32; 4];
        let mut aby = [0.0f32; 4];
        let mut abz = [0.0f32; 4];
        let mut acx = [0.0f32; 4];
        let mut acy = [0.0f32; 4];
        let mut acz = [0.0f32; 4];
        let mut apx = [0.0f32; 4];
        let mut apy = [0.0f32; 4];
        let mut apz = [0.0f32; 4];
        let mut bpx = [0.0f32; 4];
        let mut bpy = [0.0f32; 4];
        let mut bpz = [0.0f32; 4];
        let mut cpx = [0.0f32; 4];
        let mut cpy = [0.0f32; 4];
        let mut cpz = [0.0f32; 4];

        for i in 0..4 {
            abx[i] = tris.bx[i] - tris.ax[i];
            aby[i] = tris.by[i] - tris.ay[i];
            abz[i] = tris.bz[i] - tris.az[i];
            acx[i] = tris.cx[i] - tris.ax[i];
            acy[i] = tris.cy[i] - tris.ay[i];
            acz[i] = tris.cz[i] - tris.az[i];
            apx[i] = q[0] - tris.ax[i];
            apy[i] = q[1] - tris.ay[i];
            apz[i] = q[2] - tris.az[i];
            bpx[i] = q[0] - tris.bx[i];
            bpy[i] = q[1] - tris.by[i];
            bpz[i] = q[2] - tris.bz[i];
            cpx[i] = q[0] - tris.cx[i];
            cpy[i] = q[1] - tris.cy[i];
            cpz[i] = q[2] - tris.cz[i];
        }

        let d1 = dot_soa(&abx, &aby, &abz, &apx, &apy, &apz);
        let d2 = dot_soa(&acx, &acy, &acz, &apx, &apy, &apz);
        let d3 = dot_soa(&abx, &aby, &abz, &bpx, &bpy, &bpz);
        let d4 = dot_soa(&acx, &acy, &acz, &bpx, &bpy, &bpz);
        let d5 = dot_soa(&abx, &aby, &abz, &cpx, &cpy, &cpz);
        let d6 = dot_soa(&acx, &acy, &acz, &cpx, &cpy, &cpz);
        let ab2 = dot_soa(&abx, &aby, &abz, &abx, &aby, &abz);
        let ac2 = dot_soa(&acx, &acy, &acz, &acx, &acy, &acz);

        std::array::from_fn(|i| {
            let vc = d1[i] * d4[i] - d3[i] * d2[i];
            let vb = d5[i] * d2[i] - d1[i] * d6[i];
            let va = d3[i] * d6[i] - d5[i] * d4[i];
            let sum = va + vb + vc;
            let band = !sum.is_finite() || sum <= 1e-5 * ab2[i] * ac2[i];
            if band {
                return closest_on_triangle(q, tris.lane(i));
            }
            let (point, barycentric, feature) = if d1[i] <= 0.0 && d2[i] <= 0.0 {
                (
                    [tris.ax[i], tris.ay[i], tris.az[i]],
                    [1.0, 0.0, 0.0],
                    Feature::Vertex(0),
                )
            } else if d3[i] >= 0.0 && d4[i] <= d3[i] {
                (
                    [tris.bx[i], tris.by[i], tris.bz[i]],
                    [0.0, 1.0, 0.0],
                    Feature::Vertex(1),
                )
            } else if vc <= 0.0 && d1[i] >= 0.0 && d3[i] <= 0.0 {
                let t = d1[i] / (d1[i] - d3[i]);
                (
                    [
                        tris.ax[i] + t * abx[i],
                        tris.ay[i] + t * aby[i],
                        tris.az[i] + t * abz[i],
                    ],
                    [1.0 - t, t, 0.0],
                    Feature::Edge(0),
                )
            } else if d6[i] >= 0.0 && d5[i] <= d6[i] {
                (
                    [tris.cx[i], tris.cy[i], tris.cz[i]],
                    [0.0, 0.0, 1.0],
                    Feature::Vertex(2),
                )
            } else if vb <= 0.0 && d2[i] >= 0.0 && d6[i] <= 0.0 {
                let t = d2[i] / (d2[i] - d6[i]);
                (
                    [
                        tris.ax[i] + t * acx[i],
                        tris.ay[i] + t * acy[i],
                        tris.az[i] + t * acz[i],
                    ],
                    [1.0 - t, 0.0, t],
                    Feature::Edge(2),
                )
            } else if va <= 0.0 && d4[i] - d3[i] >= 0.0 && d5[i] - d6[i] >= 0.0 {
                let n = d4[i] - d3[i];
                let t = n / (n + (d5[i] - d6[i]));
                (
                    [
                        tris.bx[i] + t * (tris.cx[i] - tris.bx[i]),
                        tris.by[i] + t * (tris.cy[i] - tris.by[i]),
                        tris.bz[i] + t * (tris.cz[i] - tris.bz[i]),
                    ],
                    [0.0, 1.0 - t, t],
                    Feature::Edge(1),
                )
            } else {
                let vf = vb / sum;
                let wf = vc / sum;
                (
                    [
                        tris.ax[i] + abx[i] * vf + acx[i] * wf,
                        tris.ay[i] + aby[i] * vf + acy[i] * wf,
                        tris.az[i] + abz[i] * vf + acz[i] * wf,
                    ],
                    [1.0 - vf - wf, vf, wf],
                    Feature::Face,
                )
            };
            let dx = q[0] - point[0];
            let dy = q[1] - point[1];
            let dz = q[2] - point[2];
            TriClosest {
                point,
                dist2: (dx * dx + dy * dy) + dz * dz,
                barycentric,
                feature,
            }
        })
    }
}

use backend::{closest_on_triangles4_impl, ray_triangles4_impl};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tri::ray_triangle;
    use proptest::prelude::*;

    #[test]
    fn degenerate_sliver_and_edge_ray_match_scalar() {
        let tris = [
            // Degenerate zero-area (collinear) triangle
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            // Extremely thin sliver
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 1e-4, 0.0]],
            // Collapsed point triangle
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            // Clean triangle whose shared edge the ray passes through
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ];
        let t4 = Tri4::from_array(&tris);
        let pre = RayPre::new([0.5, 0.0, -1.0], [0.0, 0.0, 1.0]);

        for cull in [Cull::None, Cull::Back, Cull::Front] {
            let batch = ray_triangles4(&pre, &t4, cull, 0.0, 10.0);
            for i in 0..4 {
                let scalar = ray_triangle(&pre, tris[i], cull, 0.0, 10.0);
                assert_eq!(batch[i].is_some(), scalar.is_some());
                if let (Some(a), Some(b)) = (batch[i], scalar) {
                    assert!((a.t - b.t).abs() <= 1e-6);
                    for k in 0..3 {
                        assert!((a.barycentric[k] - b.barycentric[k]).abs() <= 1e-6);
                    }
                }
            }
        }

        let q = [5.0, 0.25, 0.5];
        let cbatch = closest_on_triangles4(q, &t4);
        for i in 0..4 {
            let s = closest_on_triangle(q, tris[i]);
            assert_eq!(cbatch[i].feature, s.feature);
            assert!((cbatch[i].dist2 - s.dist2).abs() <= 1e-6);
            for k in 0..3 {
                assert!((cbatch[i].point[k] - s.point[k]).abs() <= 1e-6);
                assert!((cbatch[i].barycentric[k] - s.barycentric[k]).abs() <= 1e-6);
            }
        }
    }

    fn arb_tri() -> impl Strategy<Value = [[f32; 3]; 3]> {
        prop::array::uniform3(prop::array::uniform3(-20.0f32..20.0f32))
    }

    proptest! {
        #[test]
        fn ray_and_closest4_match_scalar(
            t0 in arb_tri(),
            t1 in arb_tri(),
            t2 in arb_tri(),
            t3 in arb_tri(),
            orig in prop::array::uniform3(-25.0f32..25.0f32),
            dir in prop::array::uniform3(-1.0f32..1.0f32),
            q in prop::array::uniform3(-25.0f32..25.0f32),
            min_t in 0.0f32..2.0f32,
            span in 0.0f32..100.0f32,
        ) {
            let tris = [t0, t1, t2, t3];
            let t4 = Tri4::from_array(&tris);
            let dir = if dir == [0.0; 3] { [0.0, 0.0, 1.0] } else { dir };
            let pre = RayPre::new(orig, dir);
            let max_t = min_t + span;

            for cull in [Cull::None, Cull::Back, Cull::Front] {
                let batch = ray_triangles4(&pre, &t4, cull, min_t, max_t);
                for i in 0..4 {
                    let s = ray_triangle(&pre, tris[i], cull, min_t, max_t);
                    prop_assert_eq!(batch[i].is_some(), s.is_some());
                    if let (Some(a), Some(b)) = (batch[i], s) {
                        prop_assert!((a.t - b.t).abs() <= 1e-6);
                        for k in 0..3 {
                            prop_assert!((a.barycentric[k] - b.barycentric[k]).abs() <= 1e-6);
                        }
                    }
                }
            }

            let cbatch = closest_on_triangles4(q, &t4);
            for i in 0..4 {
                let s = closest_on_triangle(q, tris[i]);
                prop_assert_eq!(cbatch[i].feature, s.feature);
                prop_assert!((cbatch[i].dist2 - s.dist2).abs() <= 1e-6);
                for k in 0..3 {
                    prop_assert!((cbatch[i].point[k] - s.point[k]).abs() <= 1e-6);
                    prop_assert!((cbatch[i].barycentric[k] - s.barycentric[k]).abs() <= 1e-6);
                }
            }
        }
    }
}
