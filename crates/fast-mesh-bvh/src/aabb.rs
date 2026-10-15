use std::cmp::Ordering;

/// Axis-aligned bounding box in mesh-local coordinates.
///
/// [`Aabb::EMPTY`] (min = +∞, max = −∞) contains nothing and overlaps nothing.
/// Non-finite coordinates are tolerated: `f32::min`/`max` ignore NaN, so a NaN
/// coordinate never widens a box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Box of one triangle, given slot-aligned positions.
    pub fn of_triangle(positions: &[[f32; 3]], tri: [u32; 3]) -> Self {
        let mut b = Self::EMPTY;
        for v in tri {
            b.grow_point(positions[v as usize]);
        }
        b
    }

    pub fn grow_point(&mut self, p: [f32; 3]) {
        for (k, c) in p.into_iter().enumerate() {
            self.min[k] = self.min[k].min(c);
            self.max[k] = self.max[k].max(c);
        }
    }

    pub fn grow(&mut self, other: &Aabb) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(other.min[k]);
            self.max[k] = self.max[k].max(other.max[k]);
        }
    }

    pub fn union(mut self, other: &Aabb) -> Aabb {
        self.grow(other);
        self
    }

    /// True when any axis has `min > max` (or a NaN bound).
    pub fn is_empty(&self) -> bool {
        (0..3).any(|k| {
            !matches!(
                self.min[k].partial_cmp(&self.max[k]),
                Some(Ordering::Less | Ordering::Equal)
            )
        })
    }

    /// True when `other` lies inside this box. An empty `other` is always contained.
    pub fn contains(&self, other: &Aabb) -> bool {
        other.is_empty()
            || (0..3).all(|k| self.min[k] <= other.min[k] && other.max[k] <= self.max[k])
    }

    /// True when the boxes share at least one point (touching counts).
    pub fn overlaps(&self, other: &Aabb) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && (0..3).all(|k| self.min[k] <= other.max[k] && other.min[k] <= self.max[k])
    }

    /// Surface area; 0 for an empty box.
    pub fn surface_area(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let d = [0, 1, 2].map(|k| self.max[k] - self.min[k]);
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    /// Box center; components that are not finite (empty or infinite boxes) read 0.
    pub fn center(&self) -> [f32; 3] {
        [0, 1, 2].map(|k| {
            let c = (self.min[k] + self.max[k]) * 0.5;
            if c.is_finite() { c } else { 0.0 }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_box_contains_and_overlaps_nothing() {
        let unit = Aabb::new([0.0; 3], [1.0; 3]);
        assert!(Aabb::EMPTY.is_empty());
        assert!(!Aabb::EMPTY.overlaps(&unit));
        assert!(unit.contains(&Aabb::EMPTY));
        assert!(!Aabb::EMPTY.contains(&unit));
        assert_eq!(Aabb::EMPTY.surface_area(), 0.0);
    }

    #[test]
    fn touching_boxes_overlap() {
        let a = Aabb::new([0.0; 3], [1.0; 3]);
        let b = Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(a.overlaps(&b));
        let c = Aabb::new([1.5, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn triangle_box_and_area() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 1.0]];
        let b = Aabb::of_triangle(&positions, [0, 1, 2]);
        assert_eq!(b, Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 1.0]));
        assert_eq!(b.surface_area(), 2.0 * (6.0 + 3.0 + 2.0));
        assert_eq!(b.center(), [1.0, 1.5, 0.5]);
    }

    #[test]
    fn nan_coordinates_never_widen_a_box() {
        let positions = [[f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, f32::NAN, 2.0]];
        let b = Aabb::of_triangle(&positions, [0, 1, 2]);
        assert_eq!(b, Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 2.0]));
        let all_nan = Aabb::of_triangle(&[[f32::NAN; 3]], [0, 0, 0]);
        assert!(all_nan.is_empty());
        assert_eq!(all_nan.center(), [0.0; 3]);
    }
}
