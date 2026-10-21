# Wide-range drag profile (drag-speed part A, Task 1)

Native, single-threaded (`parallel` feature off), i5-13600K, release build,
2026-10-21, at HEAD `f64cc25` plus the new bench.

## Setup

`cargo bench -p fast-mesh-scene --bench scene -- "wide drag"`, group
`scene wide drag 1M`: two Lazy `terrain(708)` meshes (1,002,528 triangles,
502,681 vertices each), B flipped over A through 4 of the 64 touching poses
of `bench_fields` (indices 0, 16, 32 and 48, `WIDE_POSES`), `max_distance`
8, colour range 0 to 6 (the 9a JS "wide" case). `field on A` has one field (A to B); `fields on A and B` adds B to A.
Each pose runs `set_placement`, `update_fields`, `colors` and
`take_changes`. `FAST_MESH_BENCH_10M=1` adds `scene wide drag 10M`
(`terrain(2237)`, 10,008,338 triangles), which was not run.

## Method

No sampling profiler was usable on this machine: `cargo flamegraph`, `samply`,
`xperf` and DTrace are not installed, and `wpr` needs an elevated shell and
gives ETL files with no analyser here. So the phases were measured with
temporary `Instant` timers (atomic counters in a scratch module), in the
working copy only, and reverted before the commit:

- in `compute_field`: `triangles_near`, the rest of the candidate assembly,
  the transform of A's points into B, `signed_distances`, and the plan
  (values, `changed`, `finite`);
- in `signed_distances` (sequential path), per point: `closest_point` and
  the sign test (`SignData::is_inside`);
- `prepare_field`, `apply_field` and its recolour (`ColorState::update` and
  the report);
- in the bench: `set_placement`, `update_fields`, `colors`, `take_changes`.

One timer costs about 52 ns. The per-point timers add about 0.5M of them per
update (about 26 ms, 0.2% of the total), so the shares are not corrected.

Because one pose takes about 15 s, the phase profile used the first 8 of
the 64 poses after a 1-pose setup: 7 real updates (pose 0 is the setup pose
and updates nothing). The phase numbers below are means per update over
poses 1 to 7. They were taken before the bench was cut to `WIDE_POSES`.
The sign mode was switched through a temporary env var in the bench: `auto`
(resolves to pseudonormal, the terrain is open) and forced `Winding`.

## Baseline per pose

**The baseline for Tasks 2 and 5** is the committed bench: poses 0, 16, 32
and 48, auto sign mode, built without the profiling timers. It comes from a
full criterion run (10 samples, each iteration runs the 4 poses in turn), as
the criterion median divided by 4. The setup pass of the same run gave
per-pose means close to it.

| Variant | Criterion median per iteration | **Per pose** | Setup pass, per pose (mean, range) |
| --- | --- | --- | --- |
| field on A | 55.559 s [55.462, 55.659] | **13.89 s** | 14.04 s (12.89 to 15.48) |
| fields on A and B | 111.09 s [110.82, 111.35] | **27.77 s** | 27.66 s (25.71 to 30.74) |

Those 4 poses recolour 221,381 vertices per pose on average for `field on
A`, and 442,762 for both fields.

From the phase profile (poses 1 to 7, with the timers):

| Variant | Sign mode | Per pose |
| --- | --- | --- |
| field on A | auto (pseudonormal) | 14.95 s |
| fields on A and B | auto (pseudonormal) | 29.95 s |
| field on A | winding | 22.11 s |
| fields on A and B | winding | 44.14 s |

`set_placement`, `colors` and `take_changes` are under 0.1 ms per pose.
Per update and field: 506,403 near triangles of A (of 1,002,528), 261,621
candidate vertices (of 502,681), 246,192 of them within range and so
sign-tested.

## Where the time goes (per pose, 1M)

| Phase | field on A, auto | fields on A and B, auto | field on A, winding |
| --- | --- | --- | --- |
| candidate search (`triangles_near`) | 14,435 ms (96.6%) | 28,911 ms (96.5%) | 14,452 ms (65.4%) |
| closest-point queries | 413 ms (2.8%) | 839 ms (2.8%) | 445 ms (2.0%) |
| sign tests | 10 ms (0.07%) | 21 ms (0.07%) | 7,123 ms (32.2%) |
| apply/recolour | 4 ms (0.03%) | 8 ms (0.03%) | 4 ms (0.02%) |
| other | 84 ms (0.6%) | 170 ms (0.6%) | 88 ms (0.4%) |

"Other" is mostly the candidate sort and dedup (56 ms per field), the plan
(6 ms), the point transform (3 ms) and the per-point loop in
`signed_distances` around the two timed calls (19 ms). Per point:
`closest_point` is about 1.6 µs; the pseudonormal sign test about 40 ns; the
winding sign test about 29 µs.

For comparison, the contact drag (`max_distance` 0.08, the same
instrumentation, 192 poses) takes 1.54 ms per pose for `field on A`, of which
`triangles_near` is 1.40 ms (91%) and `signed_distances` 0.12 ms (270
candidates).

### Why the candidate search dominates

The bench takes the incremental path: the field is not fresh, `max_distance`
is finite and only B's placement changed, so `prepare_field` asks for A's
triangles near B. `triangles_near` enumerates every triangle pair within the
margin (8 plus the rounding pad) and pushes both triangles once per pair
before sorting and deduplicating. At this range each A triangle has hundreds
of B triangles within reach, so it tests and records an estimated hundreds of
millions of pairs per pose to find half of A's triangles. That figure is an
estimate, about 400 B triangles within 8 units of each of 500k near A
triangles; the pairs were not counted. The search alone costs about
30 times what recomputing every candidate costs.

The 9a JS figure (~0.4 s per pose) is close to the closest-point plus sign
cost measured here (0.42 s). The JS bench uses a different terrain (other
amplitude and frequencies, B with a phase shift), so the two are not the
same workload; this note does not explain the gap. The native baseline for
Tasks 2 and 5 is the table above.

## Ranked exact optimisations for Task 2

Each keeps every value, colour and change record bit-identical, because a
vertex's value depends only on its position, B, B's sign data and
`max_distance`, and the plan compares each new value with the old one bit
for bit. Recomputing more vertices than needed never changes any output.

1. **Skip the pair enumeration when the near set would be large.**
   Expected: 14.9 s to about 0.5 to 0.9 s per pose for `field on A`
   (roughly 15 to 30 times), pseudonormal.
   - The estimate includes the extra candidates. Recomputing all 502,681
     vertices instead of today's 261,621 adds about 241k points. About 15k of
     the 262k candidates are already out of range today, and the extra ones
     are out of range too.
   - Each out-of-range point still costs a bounded closest-point query. If
     that is as cheap as an in-range one (1.6 µs, an upper bound), the
     closest-point phase grows from 0.41 s to about 0.8 s.
   - If the query stops at B's root box, which is likely for most of them,
     it adds far less, so the phase stays near 0.45 s.
   - The sign tests (40 ns each) run only for in-range points and do not
     grow.
   - The simplest exact form is to recompute all of A's vertices, as the full
     path does, when the work estimate (`known` plus the near bound) is a
     large share of A. The 230k vertices beyond reach cost little each,
     because the bounded closest-point query stops at B's root box.
   - A finer form answers "is this A triangle within the margin of B?"
     without listing pairs. It stops at the first B triangle in reach, or
     accepts a whole A node at once when its box is within the margin of a B
     triangle with a rounding pad. That keeps the candidate set as small as
     today's.
   - Either form must keep the near path for the contact range, where it
     wins (1.4 ms against recomputing 500k vertices).
2. **Warm-start the closest-point query.** Expected: 1.5 to 3 times on the
   closest-point phase, which is about 85% of what is left after item 1
   (pseudonormal).
   - Cache each vertex's last closest triangle. At the new pose, seed the
     bound with that triangle's distance before the traversal.
   - The result is the lexicographic minimum of (distance², id) within
     `max_distance` either way, because boxes at exactly the bound are still
     visited and ties go to the smaller id. So it equals the cold result.
   - A cached triangle that is dead or moved must not be read: check
     `is_triangle_alive`, or drop the cache on a topology change of B.
   - Covers review focus 1 and 2.
3. **Winding mode: walk points in spatially coherent batches.** Expected
   (unmeasured, a guess from cache behaviour only): about 1.3 to 2 times on
   the winding sign phase (32% of the winding total today, about 90% after
   item 1).
   - Points near each other open the same BVH nodes. Visiting a batch of
     neighbouring points per node keeps the moments and the triangles in
     cache.
   - It stays exact only if each point keeps its own traversal order and
     summation order. Only the interleaving between points changes.
   - Not needed for the bench, which runs in pseudonormal (`auto` on an open
     terrain).
4. **Cheaper candidate assembly.** About 56 ms per field per pose (0.4% today,
   about 10% after item 1). Build the candidate list as a bitmap over vertex
   slots instead of extend, sort and dedup (or reuse it across poses).
5. **Hoist the per-point checks** in `signed_distances`. `closest_point`
   re-runs `validate` and `check` for every point; an internal unchecked
   entry would skip them. This is a few ns per point, under 1%.

Apply and recolour (4 ms) are not worth touching.
