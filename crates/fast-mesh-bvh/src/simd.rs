//! 4-wide SoA and Wasm SIMD128 kernels for axis-aligned bounding boxes.

use crate::Aabb;
use fast_mesh_store::{Mesh, TriangleId};

/// Conservative error factor for Woop/Ize watertight ray-box slab tests,
/// matching `fast_mesh_query::ray::ray_box`.
const GAMMA3: f32 = 3.0 * (f32::EPSILON / 2.0) / (1.0 - 3.0 * (f32::EPSILON / 2.0));
const GROW: f32 = 1.0 + 2.0 * GAMMA3;
const SHRINK: f32 = 1.0 - 2.0 * GAMMA3;

/// Four [`Aabb`] bounds laid out in Structure-of-Arrays form (`min_*`, `max_*`)
/// and 16-byte aligned for single-instruction 4-lane SIMD loading.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb4 {
    pub min_x: [f32; 4],
    pub min_y: [f32; 4],
    pub min_z: [f32; 4],
    pub max_x: [f32; 4],
    pub max_y: [f32; 4],
    pub max_z: [f32; 4],
}

impl Aabb4 {
    /// Four empty boxes (`Aabb::EMPTY`).
    pub const EMPTY: Self = Self {
        min_x: [f32::INFINITY; 4],
        min_y: [f32::INFINITY; 4],
        min_z: [f32::INFINITY; 4],
        max_x: [f32::NEG_INFINITY; 4],
        max_y: [f32::NEG_INFINITY; 4],
        max_z: [f32::NEG_INFINITY; 4],
    };

    /// Packs four [`Aabb`] references into SoA lanes `0..4`.
    #[inline]
    pub fn from_aabbs(boxes: [&Aabb; 4]) -> Self {
        Self {
            min_x: [
                boxes[0].min[0],
                boxes[1].min[0],
                boxes[2].min[0],
                boxes[3].min[0],
            ],
            min_y: [
                boxes[0].min[1],
                boxes[1].min[1],
                boxes[2].min[1],
                boxes[3].min[1],
            ],
            min_z: [
                boxes[0].min[2],
                boxes[1].min[2],
                boxes[2].min[2],
                boxes[3].min[2],
            ],
            max_x: [
                boxes[0].max[0],
                boxes[1].max[0],
                boxes[2].max[0],
                boxes[3].max[0],
            ],
            max_y: [
                boxes[0].max[1],
                boxes[1].max[1],
                boxes[2].max[1],
                boxes[3].max[1],
            ],
            max_z: [
                boxes[0].max[2],
                boxes[1].max[2],
                boxes[2].max[2],
                boxes[3].max[2],
            ],
        }
    }

    /// Packs two [`Aabb`] references into lanes `0..2`, leaving lanes `2..4`
    /// as [`Aabb::EMPTY`] so inactive lanes never hit or match.
    #[inline]
    pub fn from_two(a: &Aabb, b: &Aabb) -> Self {
        Self::from_aabbs([a, b, &Aabb::EMPTY, &Aabb::EMPTY])
    }

    /// Packs up to four boxes from `boxes`, padding trailing lanes with
    /// [`Aabb::EMPTY`].
    #[inline]
    pub fn from_slice(boxes: &[Aabb]) -> Self {
        let get = |i: usize| boxes.get(i).unwrap_or(&Aabb::EMPTY);
        Self::from_aabbs([get(0), get(1), get(2), get(3)])
    }

    /// Packs the bounding boxes of up to four triangles from `mesh` starting at
    /// `tri_ids`.
    #[inline]
    pub fn of_triangles(mesh: &Mesh, tri_ids: [TriangleId; 4]) -> Self {
        let positions = mesh.positions();
        let triangles = mesh.triangles();
        let b0 = Aabb::of_triangle(positions, triangles[tri_ids[0] as usize]);
        let b1 = Aabb::of_triangle(positions, triangles[tri_ids[1] as usize]);
        let b2 = Aabb::of_triangle(positions, triangles[tri_ids[2] as usize]);
        let b3 = Aabb::of_triangle(positions, triangles[tri_ids[3] as usize]);
        Self::from_aabbs([&b0, &b1, &b2, &b3])
    }

    /// Extracts lane `i` (`0..4`) back into an [`Aabb`].
    #[inline]
    pub fn lane(&self, i: usize) -> Aabb {
        Aabb {
            min: [self.min_x[i], self.min_y[i], self.min_z[i]],
            max: [self.max_x[i], self.max_y[i], self.max_z[i]],
        }
    }

    /// Tests all four boxes against a single box `other`, returning a 4-bit
    /// bitmask (bit `i` set iff lane `i` overlaps `other`).
    #[inline]
    pub fn overlaps_aabb(&self, other: &Aabb) -> u8 {
        overlaps_aabb_impl(self, other)
    }

    /// Lane-wise box overlap against `other`, returning a 4-bit bitmask.
    #[inline]
    pub fn overlaps4(&self, other: &Self) -> u8 {
        overlaps4_impl(self, other)
    }

    /// Conservative watertight slab test of a ray against all four boxes in
    /// `[min_t, max_t]`. Returns `(hit_mask, entry_t)`, where bit `i` of
    /// `hit_mask` is set iff lane `i` intersects the ray interval and
    /// `entry_t[i]` holds the clamped entry parameter `enter.max(min_t)`.
    #[inline]
    pub fn ray_slab(
        &self,
        origin: [f32; 3],
        inv_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> (u8, [f32; 4]) {
        ray_slab_impl(self, origin, inv_dir, min_t, max_t)
    }

    /// Returns the squared Euclidean distance from point `q` to each of the
    /// four boxes (0 inside the box, `+inf` for an empty lane).
    #[inline]
    pub fn point_dist2(&self, q: [f32; 3]) -> [f32; 4] {
        point_dist2_impl(self, q)
    }

    /// Tests all four boxes against a closed sphere `(center, radius2)`,
    /// returning `(hit_mask, dist2)` where bit `i` is set iff
    /// `dist2[i] <= radius2`.
    #[inline]
    pub fn sphere_overlaps(&self, center: [f32; 3], radius2: f32) -> (u8, [f32; 4]) {
        sphere_overlaps_impl(self, center, radius2)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod backend {
    use super::*;
    use core::arch::wasm32::*;

    #[inline(always)]
    unsafe fn load(p: &[f32; 4]) -> v128 {
        unsafe { v128_load(p.as_ptr().cast::<v128>()) }
    }

    #[inline]
    pub fn overlaps_aabb_impl(boxes: &Aabb4, other: &Aabb) -> u8 {
        unsafe {
            let omin_x = f32x4_splat(other.min[0]);
            let omin_y = f32x4_splat(other.min[1]);
            let omin_z = f32x4_splat(other.min[2]);
            let omax_x = f32x4_splat(other.max[0]);
            let omax_y = f32x4_splat(other.max[1]);
            let omax_z = f32x4_splat(other.max[2]);

            let hx = v128_and(
                f32x4_le(load(&boxes.min_x), omax_x),
                f32x4_le(omin_x, load(&boxes.max_x)),
            );
            let hy = v128_and(
                f32x4_le(load(&boxes.min_y), omax_y),
                f32x4_le(omin_y, load(&boxes.max_y)),
            );
            let hz = v128_and(
                f32x4_le(load(&boxes.min_z), omax_z),
                f32x4_le(omin_z, load(&boxes.max_z)),
            );
            (i32x4_bitmask(v128_and(v128_and(hx, hy), hz)) as u8) & 0x0F
        }
    }

    #[inline]
    pub fn overlaps4_impl(a: &Aabb4, b: &Aabb4) -> u8 {
        unsafe {
            let hx = v128_and(
                f32x4_le(load(&a.min_x), load(&b.max_x)),
                f32x4_le(load(&b.min_x), load(&a.max_x)),
            );
            let hy = v128_and(
                f32x4_le(load(&a.min_y), load(&b.max_y)),
                f32x4_le(load(&b.min_y), load(&a.max_y)),
            );
            let hz = v128_and(
                f32x4_le(load(&a.min_z), load(&b.max_z)),
                f32x4_le(load(&b.min_z), load(&a.max_z)),
            );
            (i32x4_bitmask(v128_and(v128_and(hx, hy), hz)) as u8) & 0x0F
        }
    }

    #[inline]
    pub fn ray_slab_impl(
        boxes: &Aabb4,
        origin: [f32; 3],
        inv_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> (u8, [f32; 4]) {
        unsafe {
            let mut enter = f32x4_splat(f32::NEG_INFINITY);
            let mut exit = f32x4_splat(f32::INFINITY);

            macro_rules! slab_axis {
                ($min:expr, $max:expr, $axis:expr) => {
                    let inv = inv_dir[$axis];
                    let bmin = load($min);
                    let bmax = load($max);
                    let (near, far) = if inv < 0.0 {
                        (bmax, bmin)
                    } else {
                        (bmin, bmax)
                    };
                    let orig = f32x4_splat(origin[$axis]);
                    let inv_v = f32x4_splat(inv);
                    let tmin = f32x4_mul(f32x4_sub(near, orig), inv_v);
                    let tmax = f32x4_mul(f32x4_sub(far, orig), inv_v);
                    enter = f32x4_pmax(enter, tmin);
                    exit = f32x4_pmin(exit, tmax);
                };
            }

            slab_axis!(&boxes.min_x, &boxes.max_x, 0);
            slab_axis!(&boxes.min_y, &boxes.max_y, 1);
            slab_axis!(&boxes.min_z, &boxes.max_z, 2);

            let zero = f32x4_splat(0.0);
            let grow = f32x4_splat(GROW);
            let shrink = f32x4_splat(SHRINK);
            let enter = f32x4_mul(enter, v128_bitselect(grow, shrink, f32x4_lt(enter, zero)));
            let exit = f32x4_mul(exit, v128_bitselect(grow, shrink, f32x4_gt(exit, zero)));

            let min_v = f32x4_splat(min_t);
            let max_v = f32x4_splat(max_t);
            let hit = v128_and(
                v128_and(f32x4_le(enter, exit), f32x4_le(enter, max_v)),
                f32x4_ge(exit, min_v),
            );
            let clamped = f32x4_pmax(enter, min_v);
            let mut out = [0.0f32; 4];
            v128_store(out.as_mut_ptr().cast::<v128>(), clamped);
            ((i32x4_bitmask(hit) as u8) & 0x0F, out)
        }
    }

    #[inline]
    pub fn point_dist2_impl(boxes: &Aabb4, q: [f32; 3]) -> [f32; 4] {
        unsafe {
            let zero = f32x4_splat(0.0);

            let qx = f32x4_splat(q[0]);
            let dx = f32x4_pmax(
                f32x4_pmax(zero, f32x4_sub(load(&boxes.min_x), qx)),
                f32x4_sub(qx, load(&boxes.max_x)),
            );
            let qy = f32x4_splat(q[1]);
            let dy = f32x4_pmax(
                f32x4_pmax(zero, f32x4_sub(load(&boxes.min_y), qy)),
                f32x4_sub(qy, load(&boxes.max_y)),
            );
            let qz = f32x4_splat(q[2]);
            let dz = f32x4_pmax(
                f32x4_pmax(zero, f32x4_sub(load(&boxes.min_z), qz)),
                f32x4_sub(qz, load(&boxes.max_z)),
            );

            let d2 = f32x4_add(
                f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy)),
                f32x4_mul(dz, dz),
            );
            let mut out = [0.0f32; 4];
            v128_store(out.as_mut_ptr().cast::<v128>(), d2);
            out
        }
    }

    #[inline]
    pub fn sphere_overlaps_impl(boxes: &Aabb4, center: [f32; 3], radius2: f32) -> (u8, [f32; 4]) {
        let dist2 = point_dist2_impl(boxes, center);
        unsafe {
            let d2 = load(&dist2);
            let r2 = f32x4_splat(radius2);
            let mask = (i32x4_bitmask(f32x4_le(d2, r2)) as u8) & 0x0F;
            (mask, dist2)
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod backend {
    use super::*;

    #[inline]
    pub fn overlaps_aabb_impl(boxes: &Aabb4, other: &Aabb) -> u8 {
        let mut mask = 0u8;
        for i in 0..4 {
            let hit = boxes.min_x[i] <= other.max[0]
                && other.min[0] <= boxes.max_x[i]
                && boxes.min_y[i] <= other.max[1]
                && other.min[1] <= boxes.max_y[i]
                && boxes.min_z[i] <= other.max[2]
                && other.min[2] <= boxes.max_z[i];
            mask |= (hit as u8) << i;
        }
        mask
    }

    #[inline]
    pub fn overlaps4_impl(a: &Aabb4, b: &Aabb4) -> u8 {
        let mut mask = 0u8;
        for i in 0..4 {
            let hit = a.min_x[i] <= b.max_x[i]
                && b.min_x[i] <= a.max_x[i]
                && a.min_y[i] <= b.max_y[i]
                && b.min_y[i] <= a.max_y[i]
                && a.min_z[i] <= b.max_z[i]
                && b.min_z[i] <= a.max_z[i];
            mask |= (hit as u8) << i;
        }
        mask
    }

    #[inline]
    pub fn ray_slab_impl(
        boxes: &Aabb4,
        origin: [f32; 3],
        inv_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> (u8, [f32; 4]) {
        let mut enter = [f32::NEG_INFINITY; 4];
        let mut exit = [f32::INFINITY; 4];

        let mins = [&boxes.min_x, &boxes.min_y, &boxes.min_z];
        let maxs = [&boxes.max_x, &boxes.max_y, &boxes.max_z];
        for axis in 0..3 {
            let inv = inv_dir[axis];
            let orig = origin[axis];
            let (near, far) = if inv < 0.0 {
                (maxs[axis], mins[axis])
            } else {
                (mins[axis], maxs[axis])
            };
            for i in 0..4 {
                enter[i] = enter[i].max((near[i] - orig) * inv);
                exit[i] = exit[i].min((far[i] - orig) * inv);
            }
        }

        let mut mask = 0u8;
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            let en = enter[i] * if enter[i] < 0.0 { GROW } else { SHRINK };
            let ex = exit[i] * if exit[i] > 0.0 { GROW } else { SHRINK };
            let hit = en <= ex && en <= max_t && ex >= min_t;
            mask |= (hit as u8) << i;
            out[i] = en.max(min_t);
        }
        (mask, out)
    }

    #[inline]
    pub fn point_dist2_impl(boxes: &Aabb4, q: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, dst) in out.iter_mut().enumerate() {
            let dx = (boxes.min_x[i] - q[0]).max(q[0] - boxes.max_x[i]).max(0.0);
            let dy = (boxes.min_y[i] - q[1]).max(q[1] - boxes.max_y[i]).max(0.0);
            let dz = (boxes.min_z[i] - q[2]).max(q[2] - boxes.max_z[i]).max(0.0);
            *dst = ((dx * dx) + (dy * dy)) + (dz * dz);
        }
        out
    }

    #[inline]
    pub fn sphere_overlaps_impl(boxes: &Aabb4, center: [f32; 3], radius2: f32) -> (u8, [f32; 4]) {
        let dist2 = point_dist2_impl(boxes, center);
        let mut mask = 0u8;
        for (i, &d2) in dist2.iter().enumerate() {
            mask |= ((d2 <= radius2) as u8) << i;
        }
        (mask, dist2)
    }
}

use backend::{
    overlaps_aabb_impl, overlaps4_impl, point_dist2_impl, ray_slab_impl, sphere_overlaps_impl,
};

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn scalar_ray_box(
        origin: [f32; 3],
        inv_dir: [f32; 3],
        b: &Aabb,
        min_t: f32,
        max_t: f32,
    ) -> Option<f32> {
        let mut enter = f32::NEG_INFINITY;
        let mut exit = f32::INFINITY;
        for i in 0..3 {
            let inv = inv_dir[i];
            let (near, far) = if inv < 0.0 {
                (b.max[i], b.min[i])
            } else {
                (b.min[i], b.max[i])
            };
            enter = enter.max((near - origin[i]) * inv);
            exit = exit.min((far - origin[i]) * inv);
        }
        let enter = enter * if enter < 0.0 { GROW } else { SHRINK };
        let exit = exit * if exit > 0.0 { GROW } else { SHRINK };
        (enter <= exit && enter <= max_t && exit >= min_t).then_some(enter.max(min_t))
    }

    fn scalar_box_dist2(q: [f32; 3], b: &Aabb) -> f32 {
        let mut d2 = 0.0f32;
        for (i, &c) in q.iter().enumerate() {
            let d = (b.min[i] - c).max(c - b.max[i]).max(0.0);
            d2 += d * d;
        }
        d2
    }

    #[test]
    fn edge_cases_empty_flat_and_parallel_rays() {
        let b0 = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b1 = Aabb::new([2.0, 0.0, 0.0], [2.0, 3.0, 4.0]); // zero-thickness slab
        let b2 = Aabb::EMPTY;
        let b3 = Aabb::new([-5.0, -5.0, -5.0], [-4.0, -4.0, -4.0]);
        let a4 = Aabb4::from_aabbs([&b0, &b1, &b2, &b3]);

        // Ray running along x=0, y=0 slab boundary (0.0 * inf edge case).
        let origin = [0.0, 0.0, -2.0];
        let dir = [0.0, 0.0, 1.0];
        let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
        let (mask, enter) = a4.ray_slab(origin, inv, 0.0, f32::INFINITY);
        for (i, b) in [&b0, &b1, &b2, &b3].into_iter().enumerate() {
            let expected = scalar_ray_box(origin, inv, b, 0.0, f32::INFINITY);
            assert_eq!((mask & (1 << i)) != 0, expected.is_some());
            if let Some(et) = expected {
                assert!((enter[i] - et).abs() <= 1e-6);
            }
        }

        let d2 = a4.point_dist2([0.5, 0.5, 0.5]);
        assert_eq!(d2[0], 0.0);
        assert_eq!(d2[1], 2.25);
        assert_eq!(d2[2], f32::INFINITY);
        assert!((d2[3] - scalar_box_dist2([0.5, 0.5, 0.5], &b3)).abs() <= 1e-6);

        let (smask, sd2) = a4.sphere_overlaps([0.5, 0.5, 0.5], 2.25);
        assert_eq!(smask, 0b0011);
        assert_eq!(sd2, d2);
    }

    fn arb_box() -> impl Strategy<Value = Aabb> {
        prop_oneof![
            Just(Aabb::EMPTY),
            (
                prop::array::uniform3(-100.0f32..100.0f32),
                prop::array::uniform3(0.0f32..50.0f32),
            )
                .prop_map(|(min, ext)| {
                    Aabb::new(min, [min[0] + ext[0], min[1] + ext[1], min[2] + ext[2]])
                }),
        ]
    }

    proptest! {
        #[test]
        fn ray_dist_and_overlap_match_scalar(
            b0 in arb_box(),
            b1 in arb_box(),
            b2 in arb_box(),
            b3 in arb_box(),
            q in prop::array::uniform3(-120.0f32..120.0f32),
            dir in prop::array::uniform3(-1.0f32..1.0f32),
            min_t in 0.0f32..5.0f32,
            span in 0.0f32..200.0f32,
            r2 in 0.0f32..10_000.0f32,
            qbox in arb_box(),
        ) {
            let boxes = [&b0, &b1, &b2, &b3];
            let a4 = Aabb4::from_aabbs(boxes);

            let dir = if dir == [0.0; 3] { [1.0, 0.0, 0.0] } else { dir };
            let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
            let max_t = min_t + span;
            let (rmask, renter) = a4.ray_slab(q, inv, min_t, max_t);
            let d2s = a4.point_dist2(q);
            let (smask, sd2) = a4.sphere_overlaps(q, r2);
            let omask = a4.overlaps_aabb(&qbox);

            for (i, b) in boxes.into_iter().enumerate() {
                let s_ray = scalar_ray_box(q, inv, b, min_t, max_t);
                prop_assert_eq!((rmask & (1 << i)) != 0, s_ray.is_some());
                if let Some(et) = s_ray {
                    prop_assert!((renter[i] - et).abs() <= 1e-6);
                }
                let s_d2 = scalar_box_dist2(q, b);
                if s_d2.is_infinite() {
                    prop_assert!(d2s[i].is_infinite() && d2s[i].is_sign_positive());
                } else {
                    prop_assert!((d2s[i] - s_d2).abs() <= 1e-6);
                }
                prop_assert_eq!((smask & (1 << i)) != 0, s_d2 <= r2);
                prop_assert_eq!(sd2[i], d2s[i]);
                prop_assert_eq!((omask & (1 << i)) != 0, b.overlaps(&qbox));
            }
        }
    }
}
