use super::aabb::Aabb;
use cgmath::Vector3;
use rayon;
use std::f32;
use std::marker::PhantomData;
use std::mem;
use sync_splitter::SyncSplitter;

pub struct Bvh<O: PartitionHeuristic> {
    nodes: Vec<Node>,
    leaves: Vec<u32>,
    centroids: Vec<Vector3<f32>>,
    bbs: Vec<Aabb>,
    new_bbs: Vec<Aabb>,
    _phantom: PhantomData<O>,
}

impl<O: PartitionHeuristic> Bvh<O> {
    pub fn new() -> Self {
        Bvh {
            nodes: Vec::new(),
            leaves: Vec::new(),
            bbs: Vec::new(),
            new_bbs: Vec::new(),
            centroids: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn rebuild<I: IntoIterator<Item = Aabb>>(&mut self, min_leaves: usize, bbs_iter: I) {
        let Bvh {
            ref mut nodes,
            ref mut leaves,
            ref mut bbs,
            ref mut new_bbs,
            ref mut centroids,
            ..
        } = *self;
        assert!(min_leaves >= 2, "min_leaves must be at least 2");

        new_bbs.clear();
        new_bbs.extend(bbs_iter);
        let num_bbs = new_bbs.len();
        assert!(num_bbs <= (MAX_CAPACITY as usize));
        if num_bbs == 0 {
            bbs.clear();
            new_bbs.clear();
            nodes.clear();
            leaves.clear();
            centroids.clear();
            return;
        }

        if num_bbs != leaves.len() {
            mem::swap(bbs, new_bbs);
            leaves.clear();
            leaves.extend(0..num_bbs as u32);
        } else {
            bbs.clear();
            bbs.extend(leaves.iter().map(|&index| new_bbs[index as usize]));
        }
        centroids.clear();
        centroids.extend(bbs.iter().map(|bb| bb.centroid()));

        nodes.clear();
        if num_bbs <= min_leaves {
            nodes.push(Node {
                aabb: Aabb::union(&bbs[..]),
                child: 0,
                max_index: num_bbs as u32 - 1,
                leaf_end: num_bbs as u32,
            });
            return;
        }
        nodes.resize((num_bbs + 1) * 2, Node::new());
        let num_nodes = {
            let splitter = SyncSplitter::new(nodes);
            {
                let (root, root_index) = splitter.pop().unwrap();
                assert_eq!(root_index, 0);
                let root_aabb = Aabb::union(&bbs[..]);
                root.aabb = root_aabb;
                root.max_index = num_bbs as u32 - 1;
                let root_expansion = NodeExpansion::<O> {
                    node: root,
                    bbs: bbs,
                    centroids: centroids,
                    leaves: leaves,
                    offset: 0,
                    _phantom: PhantomData,
                };
                root_expansion.parallel_expand(min_leaves, &splitter);
            }
            splitter.done()
        };
        nodes.truncate(num_nodes);
    }

    #[inline]
    pub fn on_sphere_intersection<F>(
        &self,
        position: Vector3<f32>,
        radius: f32,
        min_index: usize,
        mut handler: F,
    ) where
        F: FnMut(usize),
    {
        if self.nodes.is_empty() || min_index >= self.leaves.len() ||
            !self.nodes[0].aabb.intersects_sphere(position, radius)
        {
            return;
        }
        self.sphere_intersector(
            &mut handler,
            position,
            radius,
            min_index as u32,
            &self.nodes[0],
        );
    }

    fn sphere_intersector<'a, F>(
        &'a self,
        handler: &mut F,
        position: Vector3<f32>,
        radius: f32,
        min_index: u32,
        mut node: &'a Node,
    ) where
        F: FnMut(usize),
    {
        loop {
            if node.leaf_end != INVALID_ID {
                let (start, end) = (node.child as usize, node.leaf_end as usize);
                assert!(end <= self.leaves.len());
                assert!(end > start);
                for &leaf in &self.leaves[start..end] {
                    if leaf >= min_index {
                        handler(leaf as usize);
                    }
                }
                return;
            } else {
                let child_index = node.child as usize;
                let child2 = &self.nodes[child_index + 1];
                let child1 = &self.nodes[child_index];
                let intersect1 = child1.max_index >= min_index &&
                    child1.aabb.intersects_sphere(position, radius);
                let intersect2 = child2.max_index >= min_index &&
                    child2.aabb.intersects_sphere(position, radius);

                if intersect2 {
                    if intersect1 {
                        self.sphere_intersector(handler, position, radius, min_index, child1);
                    }
                    node = child2;
                } else if intersect1 {
                    node = child1;
                } else {
                    return;
                }
            }
        }
    }
}


pub trait Number: Send {
    fn specify_number() -> usize;
}

pub trait Array<T: Copy + Sync + Send>: AsRef<[T]> + AsMut<[T]> + Sync + Send {
    fn of(value: T) -> Self;
}

pub trait ArraySize<T: Copy + Sync + Send>: Send {
    type ArrayOfSize: Array<T>;
}

macro_rules! impl_numbers {
    ($(#[number_value($value:expr)] pub enum $name:ident {})+) => {
        $(
            #[allow(unused)]
            pub enum $name {}

            impl Number for $name {
                #[inline]
                fn specify_number() -> usize {
                    $value
                }
            }

            impl<T: Copy + Sync + Send> Array<T> for [T; $value] {
                #[inline]
                fn of(value: T) -> Self {
                    [value; $value]
                }
            }

            impl<T: Copy + Sync + Send> ArraySize<T> for $name {
                type ArrayOfSize = [T; $value];
            }
            )+
    }
}

impl_numbers! {
    #[number_value(2)]
    pub enum Two {}

    #[number_value(4)]
    pub enum Four {}

    #[number_value(6)]
    pub enum Six {}

    #[number_value(8)]
    pub enum Eight {}

    #[number_value(16)]
    pub enum Sixteen {}
}


pub trait PartitionHeuristic: Send {
    fn partition(
        aabb: &Aabb,
        bbs: &mut [Aabb],
        centroids: &mut [Vector3<f32>],
        leaves: &mut [u32],
    ) -> Option<(usize, Aabb, Aabb)>;
}

pub trait SahBinLimits: Send {
    fn sah_bin_limits(
        aabb: &Aabb,
        bbs: &[Aabb],
        centroids: &[Vector3<f32>],
        leaves: &[u32],
    ) -> (usize, f32, f32);
}

pub enum CentroidAabbLimit {}
impl SahBinLimits for CentroidAabbLimit {
    #[inline]
    fn sah_bin_limits(
        _aabb: &Aabb,
        _bbs: &[Aabb],
        centroids: &[Vector3<f32>],
        _leaves: &[u32],
    ) -> (usize, f32, f32) {
        let centroid_bb = Aabb::of_points(centroids);
        let axis = centroid_bb.longest_axis();
        (axis, centroid_bb.min()[axis], centroid_bb.max()[axis])
    }
}
pub enum TotalAabbLimit {}
impl SahBinLimits for TotalAabbLimit {
    #[inline]
    fn sah_bin_limits(
        aabb: &Aabb,
        _bbs: &[Aabb],
        _centroids: &[Vector3<f32>],
        _leaves: &[u32],
    ) -> (usize, f32, f32) {
        let axis = aabb.longest_axis();
        (axis, aabb.min()[axis], aabb.max()[axis])
    }
}

pub trait SpecifyBinCount
    : Send + Number + ArraySize<f32> + ArraySize<u32> + ArraySize<Aabb> {
    type CostArray: Array<f32>;
    type CountArray: Array<u32>;
    type AabbArray: Array<Aabb>;
}
impl<N> SpecifyBinCount for N
where
    N: Number + ArraySize<f32> + ArraySize<u32> + ArraySize<Aabb>,
{
    type CostArray = <N as ArraySize<f32>>::ArrayOfSize;
    type CountArray = <N as ArraySize<u32>>::ArrayOfSize;
    type AabbArray = <N as ArraySize<Aabb>>::ArrayOfSize;
}

pub struct BinnedSahPartition<N: SpecifyBinCount, Limits: SahBinLimits> {
    _phantom: PhantomData<(N, Limits)>,
}

const INVALID_ID: u32 = 0xff_ff_ff_ff;
const MAX_CAPACITY: u32 = INVALID_ID;

#[derive(Clone, Debug)]
struct Node {
    aabb: Aabb,
    max_index: u32,
    child: u32,
    leaf_end: u32,
}

impl Node {
    fn new() -> Self {
        Node {
            aabb: Aabb::negative(),
            child: INVALID_ID,
            leaf_end: INVALID_ID,
            max_index: INVALID_ID,
        }
    }
}

const SEQUENTIAL_EXPANSION_THRESHOLD: usize = 512;

struct NodeExpansion<'a, H: PartitionHeuristic> {
    node: &'a mut Node,
    bbs: &'a mut [Aabb],
    centroids: &'a mut [Vector3<f32>],
    leaves: &'a mut [u32],
    offset: u32,
    _phantom: PhantomData<H>,
}

impl<'a, H: PartitionHeuristic> NodeExpansion<'a, H> {
    #[inline]
    fn parallel_expand(mut self, min_leaves: usize, splitter: &'a SyncSplitter<Node>) {
        while self.bbs.len() > SEQUENTIAL_EXPANSION_THRESHOLD {
            match self.expand_node(min_leaves, splitter) {
                (Some(e1), Some(e2)) => {
                    rayon::join(|| e1.parallel_expand(min_leaves, splitter), || {
                        e2.parallel_expand(min_leaves, splitter)
                    });
                    return;
                }
                (Some(e), None) | (None, Some(e)) => self = e,
                (None, None) => return,
            }
        }
        self.sequential_expand(min_leaves, splitter);
    }

    fn sequential_expand(mut self, min_leaves: usize, splitter: &'a SyncSplitter<Node>) {
        loop {
            match self.expand_node(min_leaves, splitter) {
                (Some(e1), Some(e2)) => {
                    e1.sequential_expand(min_leaves, splitter);
                    self = e2;
                }
                (Some(e), None) | (None, Some(e)) => self = e,
                (None, None) => return,
            }
        }
    }

    fn expand_node(
        self,
        min_leaves: usize,
        splitter: &'a SyncSplitter<Node>,
    ) -> (Option<Self>, Option<Self>) {
        let NodeExpansion {
            node,
            bbs,
            centroids,
            leaves,
            offset,
            ..
        } = self;

        let len = bbs.len();
        assert!(len > min_leaves);
        assert_eq!(leaves.len(), len);
        assert_eq!(centroids.len(), len);

        let (split, left_bb, right_bb) = match H::partition(&node.aabb, bbs, centroids, leaves) {
            Some(partition) => partition,
            None => {
                node.child = offset;
                node.leaf_end = offset + len as u32;
                return (None, None);
            }
        };

        assert!(split > 0 && split < len);
        let (left_bbs, right_bbs) = bbs.split_at_mut(split);
        let (left_centroids, right_centroids) = centroids.split_at_mut(split);
        let (left_leaves, right_leaves) = leaves.split_at_mut(split);
        let (len, split) = (len as u32, split as u32);
        let ((child1, child2), index1) = splitter.pop_two().expect("not enough preallocated nodes");

        node.child = index1 as u32;
        node.leaf_end = INVALID_ID;

        child1.aabb = left_bb;
        child1.max_index = *left_leaves.iter().max().expect(
            "left_leaves shouldn't be empty",
        );
        let left = if left_bbs.len() <= min_leaves {
            child1.child = offset;
            child1.leaf_end = offset + split;
            None
        } else {
            Some(NodeExpansion {
                node: child1,
                bbs: left_bbs,
                centroids: left_centroids,
                leaves: left_leaves,
                offset: offset,
                _phantom: PhantomData,
            })
        };

        child2.aabb = right_bb;
        child2.max_index = *right_leaves.iter().max().expect(
            "right_leaves shouldn't be empty",
        );
        let right = if right_bbs.len() <= min_leaves {
            child2.child = offset + split;
            child2.leaf_end = offset + len;
            None
        } else {
            Some(NodeExpansion {
                node: child2,
                bbs: right_bbs,
                centroids: right_centroids,
                leaves: right_leaves,
                offset: offset + split,
                _phantom: PhantomData,
            })
        };

        (left, right)
    }
}


#[derive(Copy, Clone)]
struct Bins<N: SpecifyBinCount> {
    bbs: N::AabbArray,
    counts: N::CountArray,
}

impl<N: SpecifyBinCount> Bins<N> {
    fn identity() -> Self {
        Bins {
            bbs: N::AabbArray::of(Aabb::negative()),
            counts: N::CountArray::of(0),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for ((count, bb), (other_count, other_bb)) in
            self.counts.as_mut().iter_mut().zip(self.bbs.as_mut()).zip(
                other.counts.as_ref().iter().zip(other.bbs.as_ref()),
            )
        {

            *count += *other_count;
            bb.add_aabb(other_bb);
        }
        self
    }

    fn create(
        binning_const: f32,
        min_limit: f32,
        axis: usize,
        bbs: &[Aabb],
        centroids: &[Vector3<f32>],
    ) -> Self {
        let mut bins = Bins::<N>::identity();
        {
            let bin_counts = bins.counts.as_mut();
            let bin_bbs = bins.bbs.as_mut();
            for (bb, centroid) in bbs.iter().zip(centroids) {
                let bin_index = (binning_const * (centroid[axis] - min_limit)) as usize;
                bin_counts[bin_index] += 1;
                bin_bbs[bin_index].add_aabb(bb);
            }
        }
        bins
    }

    fn par_create(
        binning_const: f32,
        min_limit: f32,
        axis: usize,
        bbs: &[Aabb],
        centroids: &[Vector3<f32>],
    ) -> Self {
        const CHUNK_SIZE: usize = 4096;

        let len = bbs.len();
        assert!(centroids.len() == len);
        if len < CHUNK_SIZE {
            Self::create(binning_const, min_limit, axis, bbs, centroids)
        } else {
            let (left_bbs, right_bbs) = bbs.split_at(len / 2);
            let (left_centroids, right_centroids) = centroids.split_at(len / 2);
            let (left_bins, right_bins) = rayon::join(
                || {
                    Self::create(binning_const, min_limit, axis, left_bbs, left_centroids)
                },
                || {
                    Self::create(binning_const, min_limit, axis, right_bbs, right_centroids)
                },
            );
            Self::merge(left_bins, right_bins)
        }
    }
}

impl<N: SpecifyBinCount, Limits: SahBinLimits> PartitionHeuristic
    for BinnedSahPartition<N, Limits> {
    fn partition(
        aabb: &Aabb,
        bbs: &mut [Aabb],
        centroids: &mut [Vector3<f32>],
        leaves: &mut [u32],
    ) -> Option<(usize, Aabb, Aabb)> {
        let len = bbs.len();
        assert!(len >= 2);
        assert_eq!(centroids.len(), len);
        assert_eq!(leaves.len(), len);

        let (axis, min_limit, max_limit) = Limits::sah_bin_limits(aabb, bbs, centroids, leaves);
        if max_limit - min_limit <= 1e-5 {
            return None;
        }

        let binning_const = (1.0 - 1e-5) * N::specify_number() as f32 / (max_limit - min_limit);
        let bins = Bins::<N>::par_create(binning_const, min_limit, axis, bbs, centroids);
        let bin_counts = bins.counts.as_ref();
        let bin_bbs = bins.bbs.as_ref();

        let num_bins = N::specify_number();
        let mut a_left_bbs = N::AabbArray::of(Aabb::negative());
        let mut a_left_costs = N::CostArray::of(0.0);

        {
            let left_bbs = a_left_bbs.as_mut();
            let left_costs = a_left_costs.as_mut();
            let mut left_bb = Aabb::negative();
            let mut left_count = 0u32;
            for bin_index in 0..num_bins - 1 {
                left_bb.add_aabb(&bin_bbs[bin_index]);
                left_count += bin_counts[bin_index];

                left_bbs[bin_index] = left_bb;
                left_costs[bin_index] = left_bb.area() * (left_count as f32);
            }
        }

        let left_bbs = a_left_bbs.as_ref();
        let left_costs = a_left_costs.as_ref();

        let mut best_bin_cost = f32::INFINITY;
        let mut best_bin_index = N::specify_number() + 1;
        let mut best_right_bb = Aabb::negative();
        {
            let mut right_bb = Aabb::negative();
            let mut right_count = 0u32;
            for bin_index in (0..num_bins - 1).rev() {
                right_bb.add_aabb(&bin_bbs[bin_index + 1]);
                right_count += bin_counts[bin_index + 1];
                let cost = left_costs[bin_index] + right_bb.area() * (right_count as f32);

                if cost < best_bin_cost {
                    best_bin_cost = cost;
                    best_bin_index = bin_index;
                    best_right_bb = right_bb;
                }
            }
        }

        if best_bin_cost >= len as f32 * aabb.area() {
            return None;
        }

        let len = bbs.len();
        assert!(axis < 3);
        assert!(len >= 2);
        assert_eq!(centroids.len(), len);
        assert_eq!(leaves.len(), len);
        let limit = (best_bin_index + 1) as f32;
        let mut split = 0;
        for i_leaf in 0..len {
            if binning_const * (centroids[i_leaf][axis] - min_limit) < limit {
                bbs.swap(split, i_leaf);
                centroids.swap(split, i_leaf);
                leaves.swap(split, i_leaf);
                split += 1;
            }
        }
        Some((split, left_bbs[best_bin_index], best_right_bb))
    }
}

#[cfg(test)]
mod tests {
    use super::{Bvh, TotalAabbLimit, BinnedSahPartition, Four};
    use super::super::aabb::Aabb;
    use cgmath::{Vector3, vec3};
    use fnv::FnvHashSet;

    #[derive(Copy, Clone, Debug)]
    struct Sphere {
        center: Vector3<f32>,
        radius: f32,
    }

    type ArbitraryFloat = i32;
    type ArbitraryVec3 = (ArbitraryFloat, ArbitraryFloat, ArbitraryFloat);
    type ArbitraryAabb = (ArbitraryVec3, ArbitraryVec3);
    type ArbitraryAabbVec = Vec<ArbitraryAabb>;
    type ArbitrarySphere = (ArbitraryVec3, ArbitraryFloat);

    fn make_signed_float(int: ArbitraryFloat) -> f32 {
        100.0 * (int as f32) / (i32::max_value() as f32)
    }

    fn make_positive_float(int: ArbitraryFloat) -> f32 {
        make_signed_float(int) * 0.5 + 50.1
    }

    fn make_vec3(vec: ArbitraryVec3) -> Vector3<f32> {
        vec3(
            make_signed_float(vec.0),
            make_signed_float(vec.1),
            make_signed_float(vec.2),
        )
    }

    fn make_aabb(vecs: ArbitraryAabb) -> Aabb {
        Aabb::of_points(
            &[
                make_vec3(vecs.0) - vec3(1e-1, 1e-1, 1e-1),
                make_vec3(vecs.1) + vec3(1e-1, 1e-1, 1e-1),
            ],
        )
    }

    fn make_sphere(sphere: ArbitrarySphere) -> Sphere {
        Sphere {
            center: make_vec3(sphere.0),
            radius: make_positive_float(sphere.1),
        }
    }

    type TotalBvh = Bvh<BinnedSahPartition<Four, TotalAabbLimit>>;

    quickcheck! {
        fn total_intersects_sphere(
            boxes: ArbitraryAabbVec,
            spheres: Vec<(ArbitrarySphere, usize)>,
            min_leaves: usize
            ) -> bool {
            let min_leaves  = 2 + (min_leaves % (boxes.len() + 1));
            let boxes: Vec<_> = boxes.iter().cloned().map(make_aabb).collect();
            let spheres: Vec<_> = spheres
                .iter()
                .map(|arbitrary| (make_sphere(arbitrary.0), arbitrary.1))
                .collect();
            let mut total_bvh = TotalBvh::new();

            let mut expected =
                FnvHashSet::with_capacity_and_hasher(boxes.len(), Default::default());
            let mut actual = FnvHashSet::with_capacity_and_hasher(boxes.len(), Default::default());
            total_bvh.rebuild(min_leaves, boxes.iter().cloned());
            for &(sphere, min_index) in &spheres {
                let min_index = min_index % (boxes.len() + 1);
                total_bvh.on_sphere_intersection(
                    sphere.center,
                    sphere.radius,
                    min_index,
                    |index| { actual.insert(index); },
                    );
                for (index, bb) in boxes.iter().enumerate().skip(min_index) {
                    if bb.intersects_sphere(sphere.center, sphere.radius) {
                        expected.insert(index);
                    }
                }
                if expected != actual {
                    error!(
                        "boxes={:?} sphere={:?} expected={:?} actual={:?} min_index={:?}",
                        boxes,
                        sphere,
                        expected,
                        actual,
                        min_index
                        );
                    return false;
                }

                expected.clear();
                actual.clear();
            }
            true
        }
    }
}
