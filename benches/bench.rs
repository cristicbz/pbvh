#![feature(test)]

extern crate test;
extern crate pbvh;
extern crate cgmath;

use test::{Bencher, black_box};
use pbvh::{Bvh, BinnedSahPartition, Four, Six, Eight, Sixteen, TotalAabbLimit, CentroidAabbLimit,
           Aabb, PartitionHeuristic, SmallSphereProbe, LargeSphereProbe};
use cgmath::vec3;


fn bench_build<H: PartitionHeuristic>(bencher: &mut Bencher, size: u32, min_leaves: usize) {
    let mut boxes = Vec::with_capacity(size as usize);
    let mut bvh = Bvh::<H>::with_capacity(size as usize);
    let mut r = 0u32;
    boxes.clear();
    for _ in 0..size {
        let mut floats = [0f32; 6];
        for float in &mut floats {
            r = r.wrapping_mul(1664525).wrapping_add(1013904223);
            *float = (r as f32) / (!0u32 as f32) * 2.0 - 1.0;
        }
        let centre = vec3(floats[0], floats[1], floats[2]) * 100.0;
        let half_size = vec3(floats[0], floats[1], floats[2]) * 2.0;
        boxes.push(Aabb::of_points(&[centre + half_size, centre - half_size]));
    }
    bencher.iter(|| {
        bvh.rebuild(min_leaves, boxes.iter().cloned());
        bvh.intersect(0, &SmallSphereProbe::new(vec3(0.0, 0.0, 0.0), 1.0), |i| {
            black_box(i);
        });
    });
}

fn bench_lookup<H: PartitionHeuristic>(bencher: &mut Bencher, size: u32, min_leaves: usize) {
    let mut boxes = Vec::with_capacity(size as usize);
    let mut bvh = Bvh::<H>::with_capacity(size as usize);
    let mut r = 0u32;
    boxes.clear();
    for _ in 0..size {
        let mut floats = [0f32; 6];
        for float in &mut floats {
            r = r.wrapping_mul(1664525).wrapping_add(1013904223);
            *float = (r as f32) / (!0u32 as f32) * 2.0 - 1.0;
        }
        let centre = vec3(floats[0], floats[1], floats[2]) * 100.0;
        let half_size = vec3(floats[0], floats[1], floats[2]) * 2.0;
        boxes.push(Aabb::of_points(&[centre + half_size, centre - half_size]));
    }
    bvh.rebuild(min_leaves, boxes.iter().cloned());
    bencher.iter(|| {
        bvh.intersect(0, &LargeSphereProbe::new(vec3(0.0, 0.0, 0.0), 30.0), |i| {
            black_box(i);
        });
    });
}


#[bench]
fn build_100_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn build_1000_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn build_10000_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn build_100000_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn build_100_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn build_1000_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn build_10000_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn build_100000_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn build_100_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100, 20);
}

#[bench]
fn build_1000_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 1000, 20);
}

#[bench]
fn build_10000_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 10000, 20);
}

#[bench]
fn build_100000_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100000, 20);
}


#[bench]
fn build_100_total_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn build_1000_total_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn build_10000_total_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn build_100000_total_bins8_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn build_100_total_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn build_1000_total_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn build_10000_total_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn build_100000_total_bins16_min2(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn build_100_total_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100, 20);
}

#[bench]
fn build_1000_total_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 1000, 20);
}

#[bench]
fn build_10000_total_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 10000, 20);
}

#[bench]
fn build_100000_total_bins8_min20(bencher: &mut Bencher) {
    bench_build::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100000, 20);
}


#[bench]
fn lookup_100_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn lookup_1000_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn lookup_10000_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn lookup_100000_centroid_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn lookup_100_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn lookup_1000_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn lookup_10000_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn lookup_100000_centroid_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, CentroidAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn lookup_100_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100, 20);
}

#[bench]
fn lookup_1000_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 1000, 20);
}

#[bench]
fn lookup_10000_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 10000, 20);
}

#[bench]
fn lookup_100000_centroid_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, CentroidAabbLimit>>(bencher, 100000, 20);
}


#[bench]
fn lookup_100_total_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn lookup_1000_total_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn lookup_10000_total_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn lookup_100000_total_bins8_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn lookup_100_total_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 100, 2);
}

#[bench]
fn lookup_1000_total_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 1000, 2);
}

#[bench]
fn lookup_10000_total_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 10000, 2);
}

#[bench]
fn lookup_100000_total_bins16_min2(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Sixteen, TotalAabbLimit>>(bencher, 100000, 2);
}

#[bench]
fn lookup_100_total_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100, 20);
}

#[bench]
fn lookup_1000_total_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 1000, 20);
}

#[bench]
fn lookup_10000_total_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 10000, 20);
}

#[bench]
fn lookup_100000_total_bins8_min20(bencher: &mut Bencher) {
    bench_lookup::<BinnedSahPartition<Eight, TotalAabbLimit>>(bencher, 100000, 20);
}
