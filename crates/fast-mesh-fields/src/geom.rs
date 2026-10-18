//! Small `f64` vector helpers and a point-triangle closest-feature routine.

use fast_mesh_query::Feature;

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// `a` scaled to unit length, or zero when `a` is zero.
pub(crate) fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > 0.0 { scale(a, 1.0 / n) } else { [0.0; 3] }
}

pub(crate) fn widen(p: [f32; 3]) -> [f64; 3] {
    p.map(f64::from)
}

pub(crate) fn narrow(p: [f64; 3]) -> [f32; 3] {
    p.map(|c| c as f32)
}

/// The (unnormalized) normal of a triangle, `(b - a) × (c - a)` in `f64`.
///
/// The coordinate differences of `f32` inputs are exact in `f64` and so are
/// the products, so the result is exactly zero for exactly collinear or
/// coincident corners: that is the definition of a zero-area triangle here.
pub(crate) fn triangle_normal(p: [[f32; 3]; 3]) -> [f64; 3] {
    let [a, b, c] = p.map(widen);
    cross(sub(b, a), sub(c, a))
}

/// The point of the non-degenerate triangle `p` nearest to `q`: its squared
/// distance, the point and the feature it lies on (Ericson, "Real-Time
/// Collision Detection" §5.1.5), with the same feature numbering as
/// [`fast_mesh_query::Feature`].
pub(crate) fn closest_on_triangle(q: [f64; 3], p: [[f64; 3]; 3]) -> (f64, [f64; 3], Feature) {
    let [a, b, c] = p;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(q, a);
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let at = |x: [f64; 3], f: Feature| (dot(sub(q, x), sub(q, x)), x, f);
    if d1 <= 0.0 && d2 <= 0.0 {
        return at(a, Feature::Vertex(0));
    }
    let bp = sub(q, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return at(b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return at(add(a, scale(ab, t)), Feature::Edge(0));
    }
    let cp = sub(q, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return at(c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return at(add(a, scale(ac, t)), Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return at(add(b, scale(sub(c, b), t)), Feature::Edge(1));
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    at(add(a, add(scale(ab, v), scale(ac, w))), Feature::Face)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_feature_numbering_matches_query() {
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(
            closest_on_triangle([-1.0, -1.0, 1.0], tri).2,
            Feature::Vertex(0)
        );
        assert_eq!(
            closest_on_triangle([2.0, -0.5, 0.0], tri).2,
            Feature::Vertex(1)
        );
        assert_eq!(
            closest_on_triangle([-0.5, 2.0, 0.0], tri).2,
            Feature::Vertex(2)
        );
        assert_eq!(
            closest_on_triangle([0.5, -1.0, 0.0], tri).2,
            Feature::Edge(0)
        );
        assert_eq!(
            closest_on_triangle([1.0, 1.0, 0.0], tri).2,
            Feature::Edge(1)
        );
        assert_eq!(
            closest_on_triangle([-1.0, 0.5, 0.0], tri).2,
            Feature::Edge(2)
        );
        let (d2, x, f) = closest_on_triangle([0.25, 0.25, 2.0], tri);
        assert_eq!(f, Feature::Face);
        assert_eq!(x, [0.25, 0.25, 0.0]);
        assert_eq!(d2, 4.0);
    }
}
