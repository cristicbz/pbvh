use super::aabb::Aabb;
use cgmath::{Vector3, vec3, ElementWise};
use rayon;
use std::f32;
use std::marker::PhantomData;
use std::mem;
use sync_splitter::SyncSplitter;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Intersection {
    None,
    Partial,
    Full,
}

pub trait BvhProbe {
    fn intersect(&self, bb: Aabb) -> Intersection;
    fn intersect_bool(&self, bb: Aabb) -> bool {
        self.intersect(bb) != Intersection::None
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SmallSphereProbe {
    smin: Vector3<f32>,
    smax: Vector3<f32>,
}

impl SmallSphereProbe {
    pub fn new(center: Vector3<f32>, radius: f32) -> Self {
        SmallSphereProbe {
            smin: vec3(center.x - radius, center.y - radius, center.z - radius),
            smax: vec3(center.x + radius, center.y + radius, center.z + radius),
        }
    }
}

impl BvhProbe for SmallSphereProbe {
    #[inline]
    fn intersect(&self, bb: Aabb) -> Intersection {
        let a = bb.min() - self.smax;
        let b = self.smin - bb.max();
        let bits = a.x.to_bits() & a.y.to_bits() & a.z.to_bits() & b.x.to_bits() & b.y.to_bits() &
            b.z.to_bits();
        if bits >> 31 == 0 {
            Intersection::None
        } else {
            Intersection::Partial
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LargeSphereProbe {
    center: Vector3<f32>,
    squared_radius: f32,
}

impl LargeSphereProbe {
    pub fn new(center: Vector3<f32>, radius: f32) -> Self {
        LargeSphereProbe {
            center,
            squared_radius: radius * radius,
        }
    }
}

impl BvhProbe for LargeSphereProbe {
    #[inline]
    fn intersect(&self, bb: Aabb) -> Intersection {
        let mut d2 = self.squared_radius;
        let a = self.center - bb.min();
        let b = bb.max() - self.center;

        let a2 = a.mul_element_wise(a);
        let b2 = b.mul_element_wise(b);

        if a.x < 0.0 {
            d2 -= a2.x
        } else if b.x < 0.0 {
            d2 -= b2.x
        }

        if a.y < 0.0 {
            d2 -= a2.y
        } else if b.y < 0.0 {
            d2 -= b2.y
        }

        if a.z < 0.0 {
            d2 -= a2.z;
        } else if b.z < 0.0 {
            d2 -= b2.z;
        }

        if d2 > 0.0 {
            if a2.x + a2.y + a2.z - self.squared_radius < 0.0 &&
                b2.x + b2.y + b2.z - self.squared_radius < 0.0
            {
                Intersection::Full
            } else {
                Intersection::Partial
            }
        } else {
            Intersection::None
        }
    }
}

pub struct Bvh<O: PartitionHeuristic> {
    nodes: Vec<Node>,
    leaves: Vec<u32>,
    centroids: Vec<Vector3<f32>>,
    bbs: Vec<Aabb>,
    new_bbs: Vec<Aabb>,
    _phantom: PhantomData<O>,
}

impl<O: PartitionHeuristic> Bvh<O> {
    pub fn with_capacity(capacity: usize) -> Self {
        Bvh {
            nodes: Vec::with_capacity(capacity),
            leaves: Vec::with_capacity(capacity),
            bbs: Vec::with_capacity(capacity),
            new_bbs: Vec::with_capacity(capacity),
            centroids: Vec::with_capacity(capacity),
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
        let min_leaves = min_leaves as u32;
        assert!(min_leaves >= 2, "min_leaves must be at least 2");
        assert!(
            min_leaves <= MAX_CAPACITY,
            "min_leaves must be less than MAX_CAPACITY"
        );
        let min_leaves = min_leaves as u32;

        new_bbs.clear();
        new_bbs.extend(bbs_iter);
        let num_bbs = new_bbs.len() as u32;
        assert!(num_bbs <= MAX_CAPACITY);
        if num_bbs == 0 {
            bbs.clear();
            new_bbs.clear();
            nodes.clear();
            leaves.clear();
            centroids.clear();
            return;
        }

        if num_bbs != leaves.len() as u32 {
            mem::swap(bbs, new_bbs);
            leaves.clear();
            leaves.extend(0..num_bbs);
        } else {
            bbs.clear();
            bbs.extend(leaves.iter().map(|&index| new_bbs[index as usize]));
        }
        centroids.clear();
        centroids.extend(bbs.iter().map(|bb| bb.centroid()));

        nodes.clear();
        let root_aabb = Aabb::union(bbs.iter().cloned());
        if num_bbs <= min_leaves {
            nodes.push(Node {
                aabb: root_aabb,
                child: INVALID_ID,
                max_index: num_bbs - 1,
                leaf_start: 0,
                leaf_end: num_bbs,
            });
            return;
        }
        nodes.resize((num_bbs as usize + 1) * 2, Node::new());
        let num_nodes = {
            let splitter = SyncSplitter::new(nodes);
            {
                let (root, root_index) = splitter.pop().unwrap();
                debug_assert_eq!(root_index, 0);
                root.aabb = root_aabb;
                root.leaf_start = 0;
                root.leaf_end = num_bbs;
                root.max_index = num_bbs as u32 - 1;
                let root_expansion = NodeExpansion::<O> {
                    node: root,
                    bbs,
                    centroids,
                    leaves,
                    _phantom: PhantomData,
                };
                root_expansion.parallel_expand(min_leaves, &splitter);
            }
            splitter.done()
        };
        nodes.truncate(num_nodes);
    }

    #[inline]
    pub fn intersect<F, P>(&self, min_index: usize, probe: &P, mut handler: F)
    where
        F: FnMut(usize),
        P: BvhProbe,
    {
        if self.nodes.is_empty() || min_index >= self.leaves.len() {
            return;
        }
        let root = &self.nodes[0];
        let intersect = probe.intersect(root.aabb);
        let min_index = min_index as u32;
        if intersect == Intersection::None {
            return;
        }
        if root.child == INVALID_ID || intersect == Intersection::Full {
            self.handle_all(&mut handler, min_index, &self.nodes[0]);
        } else {
            self.intersect_impl(&mut handler, probe, min_index as u32, &self.nodes[0]);
        }
    }

    #[inline]
    fn handle_all<'a, F>(&'a self, handler: &mut F, min_index: u32, node: &'a Node)
    where
        F: FnMut(usize),
    {
        let (start, end) = (node.leaf_start as usize, node.leaf_end as usize);
        debug_assert!(end <= self.leaves.len());
        debug_assert!(end > start);
        for &leaf in &self.leaves[start..end] {
            if leaf >= min_index {
                handler(leaf as usize);
            }
        }
    }

    fn intersect_impl<'a, F, P>(
        &'a self,
        handler: &mut F,
        probe: &P,
        min_index: u32,
        mut node: &'a Node,
    ) where
        F: FnMut(usize),
        P: BvhProbe,
    {
        loop {
            let child_index = node.child as usize;
            let child2 = &self.nodes[child_index + 1];
            let child1 = &self.nodes[child_index];
            let expand1 = if child1.max_index < min_index {
                false
            } else {
                let intersect = probe.intersect(child1.aabb);
                if intersect == Intersection::None {
                    false
                } else if child1.child == INVALID_ID || intersect == Intersection::Full {
                    self.handle_all(handler, min_index, child1);
                    false
                } else {
                    true
                }
            };
            let expand2 = if child2.max_index < min_index {
                false
            } else {
                let intersect = probe.intersect(child2.aabb);
                if intersect == Intersection::None {
                    false
                } else if child2.child == INVALID_ID || intersect == Intersection::Full {
                    self.handle_all(handler, min_index, child2);
                    false
                } else {
                    true
                }
            };

            if expand2 {
                if expand1 {
                    self.intersect_impl(handler, probe, min_index, child1);
                }
                node = child2;
            } else if expand1 {
                node = child1;
            } else {
                return;
            }
        }
    }
}

pub trait BinCount: Send + Sync + 'static {
    type CostArray: Array<f32>;
    type CountArray: Array<u32>;
    type AabbArray: Array<Aabb>;

    fn bin_count() -> usize;
    fn new_cost_array() -> Self::CostArray;
    fn new_count_array() -> Self::CountArray;
    fn new_aabb_array() -> Self::AabbArray;
}


pub trait Array<T: Copy + Sync + Send>: AsRef<[T]> + AsMut<[T]> + Sync + Send {}

macro_rules! impl_bin_count {
    ($(#[count($value:expr)] pub enum $name:ident {})+) => {
        $(
            pub enum $name {}

            impl<T: Copy + Sync + Send> Array<T> for [T; $value] {}

            impl BinCount for $name {
                type CostArray = [f32; $value];
                type CountArray = [u32; $value];
                type AabbArray = [Aabb; $value];

                #[inline]
                fn bin_count() -> usize { $value }

                #[inline]
                fn new_cost_array() -> Self::CostArray { [0.0; $value] }

                #[inline]
                fn new_count_array() -> Self::CountArray { [0; $value] }

                #[inline]
                fn new_aabb_array() -> Self::AabbArray { [Aabb::negative(); $value] }
            }
            )+
    }
}

impl_bin_count! {
    #[count(2)]
    pub enum Two {}

    #[count(4)]
    pub enum Four {}

    #[count(6)]
    pub enum Six {}

    #[count(8)]
    pub enum Eight {}

    #[count(16)]
    pub enum Sixteen {}
}


pub trait PartitionHeuristic: Send {
    fn partition(
        aabb: &Aabb,
        bbs: &mut [Aabb],
        centroids: &mut [Vector3<f32>],
        leaves: &mut [u32],
    ) -> Option<(u32, Aabb, Aabb)>;
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

pub struct BinnedSahPartition<N: BinCount, Limits: SahBinLimits> {
    _phantom: PhantomData<(N, Limits)>,
}

const INVALID_ID: u32 = 0xff_ff_ff_ff;
const MAX_CAPACITY: u32 = INVALID_ID;

#[derive(Clone, Debug)]
struct Node {
    aabb: Aabb,
    max_index: u32,
    child: u32,
    leaf_start: u32,
    leaf_end: u32,
}

impl Node {
    #[inline]
    fn new() -> Self {
        Node {
            aabb: Aabb::negative(),
            child: INVALID_ID,
            leaf_start: INVALID_ID,
            leaf_end: INVALID_ID,
            max_index: INVALID_ID,
        }
    }
}

const SEQUENTIAL_EXPANSION_THRESHOLD: usize = 128;

struct NodeExpansion<'a, H: PartitionHeuristic> {
    node: &'a mut Node,
    bbs: &'a mut [Aabb],
    centroids: &'a mut [Vector3<f32>],
    leaves: &'a mut [u32],
    _phantom: PhantomData<H>,
}

impl<'a, H: PartitionHeuristic> NodeExpansion<'a, H> {
    #[inline]
    fn parallel_expand(mut self, min_leaves: u32, splitter: &'a SyncSplitter<Node>) {
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

    fn sequential_expand(mut self, min_leaves: u32, splitter: &'a SyncSplitter<Node>) {
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
        min_leaves: u32,
        splitter: &'a SyncSplitter<Node>,
    ) -> (Option<Self>, Option<Self>) {
        let NodeExpansion {
            node,
            bbs,
            centroids,
            leaves,
            ..
        } = self;
        let len = bbs.len() as u32;
        debug_assert!(len > min_leaves);
        debug_assert_eq!(leaves.len() as u32, len);
        debug_assert_eq!(centroids.len() as u32, len);

        let (split, left_bb, right_bb) = match H::partition(&node.aabb, bbs, centroids, leaves) {
            Some(partition) => partition,
            None => {
                node.child = INVALID_ID;
                return (None, None);
            }
        };

        debug_assert!(split > 0 && split < len);
        let (left_bbs, right_bbs) = bbs.split_at_mut(split as usize);
        let (left_centroids, right_centroids) = centroids.split_at_mut(split as usize);
        let (left_leaves, right_leaves) = leaves.split_at_mut(split as usize);
        let split_offset = node.leaf_start + split;
        let ((child1, child2), index1) = splitter.pop_two().expect("not enough preallocated nodes");

        node.child = index1 as u32;
        child1.aabb = left_bb;
        child1.max_index = *left_leaves.iter().max().expect(
            "left_leaves shouldn't be empty",
        );
        child1.leaf_start = node.leaf_start;
        child1.leaf_end = split_offset;
        let left = if left_bbs.len() as u32 <= min_leaves {
            None
        } else {
            Some(NodeExpansion {
                node: child1,
                bbs: left_bbs,
                centroids: left_centroids,
                leaves: left_leaves,
                _phantom: PhantomData,
            })
        };

        child2.aabb = right_bb;
        child2.max_index = *right_leaves.iter().max().expect(
            "right_leaves shouldn't be empty",
        );
        child2.leaf_start = split_offset;
        child2.leaf_end = node.leaf_end;
        let right = if right_bbs.len() as u32 <= min_leaves {
            None
        } else {
            Some(NodeExpansion {
                node: child2,
                bbs: right_bbs,
                centroids: right_centroids,
                leaves: right_leaves,
                _phantom: PhantomData,
            })
        };

        (left, right)
    }
}


#[derive(Copy, Clone)]
struct Bins<N: BinCount> {
    bbs: N::AabbArray,
    counts: N::CountArray,
}

impl<N: BinCount> Bins<N> {
    fn identity() -> Self {
        Bins {
            bbs: N::new_aabb_array(),
            counts: N::new_count_array(),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for ((count, bb), (&other_count, &other_bb)) in
            self.counts.as_mut().iter_mut().zip(self.bbs.as_mut()).zip(
                other.counts.as_ref().iter().zip(other.bbs.as_ref()),
            )
        {

            *count += other_count;
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
            for (&bb, centroid) in bbs.iter().zip(centroids) {
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
        const CHUNK_SIZE: usize = 32768;

        let len = bbs.len();
        debug_assert!(centroids.len() == len);
        if len < CHUNK_SIZE {
            Self::create(binning_const, min_limit, axis, bbs, centroids)
        } else {
            let (left_bbs, right_bbs) = bbs.split_at(len / 2);
            let (left_centroids, right_centroids) = centroids.split_at(len / 2);
            let (left_bins, right_bins) = rayon::join(
                || {
                    Self::par_create(binning_const, min_limit, axis, left_bbs, left_centroids)
                },
                || {
                    Self::par_create(binning_const, min_limit, axis, right_bbs, right_centroids)
                },
            );
            Self::merge(left_bins, right_bins)
        }
    }
}

impl<N: BinCount, Limits: SahBinLimits> PartitionHeuristic for BinnedSahPartition<N, Limits> {
    fn partition(
        aabb: &Aabb,
        bbs: &mut [Aabb],
        centroids: &mut [Vector3<f32>],
        leaves: &mut [u32],
    ) -> Option<(u32, Aabb, Aabb)> {
        let len = bbs.len();
        debug_assert!(len >= 2);
        debug_assert_eq!(centroids.len(), len);
        debug_assert_eq!(leaves.len(), len);

        let (axis, min_limit, max_limit) = Limits::sah_bin_limits(aabb, bbs, centroids, leaves);
        if max_limit - min_limit <= 1e-5 {
            return None;
        }

        let binning_const = (1.0 - 1e-5) * N::bin_count() as f32 / (max_limit - min_limit);
        let bins = Bins::<N>::par_create(binning_const, min_limit, axis, bbs, centroids);
        let bin_bbs = bins.bbs.as_ref();
        let bin_counts = bins.counts.as_ref();

        let num_bins = N::bin_count();
        let mut a_left_bbs = N::new_aabb_array();
        let mut a_left_costs = N::new_cost_array();

        {
            let left_bbs = a_left_bbs.as_mut();
            let left_costs = a_left_costs.as_mut();
            let mut left_bb = Aabb::negative();
            let mut left_count = 0;
            for bin_index in 0..num_bins - 1 {
                left_bb.add_aabb(bin_bbs[bin_index]);
                left_count += bin_counts[bin_index];

                left_bbs[bin_index] = left_bb;
                left_costs[bin_index] = left_bb.area() * left_count as f32;
            }
        }

        let left_bbs = a_left_bbs.as_ref();
        let left_costs = a_left_costs.as_ref();

        let mut best_bin_cost = f32::INFINITY;
        let mut best_bin_index = N::bin_count() + 1;
        let mut best_right_bb = Aabb::negative();
        {
            let mut right_bb = Aabb::negative();
            let mut right_count = 0;
            for bin_index in (0..num_bins - 1).rev() {
                right_bb.add_aabb(bin_bbs[bin_index + 1]);
                right_count += bin_counts[bin_index + 1];
                let cost = left_costs[bin_index] + right_bb.area() * right_count as f32;

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

        let len = bbs.len() as u32;
        debug_assert!(axis < 3);
        debug_assert!(len >= 2);
        debug_assert_eq!(centroids.len() as u32, len);
        debug_assert_eq!(leaves.len() as u32, len);
        let limit = (best_bin_index + 1) as f32;
        let mut split = 0u32;
        for i_leaf in 0..len {
            if binning_const * (centroids[i_leaf as usize][axis] - min_limit) < limit {
                bbs.swap(split as usize, i_leaf as usize);
                centroids.swap(split as usize, i_leaf as usize);
                leaves.swap(split as usize, i_leaf as usize);
                split += 1;
            }
        }
        Some((split, left_bbs[best_bin_index], best_right_bb))
    }
}

#[cfg(test)]
mod tests {
    use super::{Bvh, TotalAabbLimit, CentroidAabbLimit, Two, BinnedSahPartition, Six,
                PartitionHeuristic, SmallSphereProbe, LargeSphereProbe, BvhProbe};
    use super::super::aabb::Aabb;
    use cgmath::{Vector3, vec3};
    use fnv::FnvHashSet;
    use quickcheck::quickcheck;

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

    fn make_sphere(sphere: ArbitrarySphere) -> LargeSphereProbe {
        LargeSphereProbe::new(make_vec3(sphere.0), make_positive_float(sphere.1))
    }

    fn intersect_sphere_helper<H: PartitionHeuristic>(
        first: ArbitraryAabbVec,
        second: ArbitraryAabbVec,
        spheres: Vec<(ArbitrarySphere, usize)>,
        min_leaves: usize,
    ) -> bool {
        let max_len = first.len().max(second.len());
        let min_leaves = 2 + (min_leaves % (max_len + 1));
        let spheres: Vec<_> = spheres
            .iter()
            .map(|arbitrary| (make_sphere(arbitrary.0), arbitrary.1))
            .collect();
        let mut total_bvh = Bvh::<H>::with_capacity(max_len);

        let mut expected = FnvHashSet::with_capacity_and_hasher(max_len, Default::default());
        let mut actual = FnvHashSet::with_capacity_and_hasher(max_len, Default::default());
        for boxes in &[&first, &first, &second, &first] {
            let boxes: Vec<_> = boxes.iter().cloned().map(make_aabb).collect();
            total_bvh.rebuild(min_leaves, boxes.iter().cloned());
            for &(sphere, min_index) in &spheres {
                let min_index = min_index % (boxes.len() + 1);
                total_bvh.intersect(min_index, &sphere, |index| { actual.insert(index); });
                for (index, &bb) in boxes.iter().enumerate().skip(min_index) {
                    if sphere.intersect_bool(bb) {
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
        }
        true
    }

    #[test]
    fn total_intersect_sphere() {
        quickcheck(
            intersect_sphere_helper::<BinnedSahPartition<Six, TotalAabbLimit>> as
                fn(ArbitraryAabbVec,
                   ArbitraryAabbVec,
                   Vec<(ArbitrarySphere, usize)>,
                   usize)
                   -> bool,
        );
    }

    #[test]
    fn centroid_intersect_sphere() {
        quickcheck(
            intersect_sphere_helper::<BinnedSahPartition<Two, CentroidAabbLimit>> as
                fn(ArbitraryAabbVec,
                   ArbitraryAabbVec,
                   Vec<(ArbitrarySphere, usize)>,
                   usize)
                   -> bool,
        );
    }


    #[test]
    fn small_sphere_probe() {
        let bb = Aabb::of_points(&[vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0)]);

        assert!(SmallSphereProbe::new(vec3(0.0, 0.0, 0.0), 1.0).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, 0.0, 0.0), 0.1).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, 0.0, 0.0), 10.0).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(2.0, 0.0, 0.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, 2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, 0.0, 2.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(2.0, 2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, 2.0, 2.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(2.0, 0.0, 2.0), 1.5).intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(2.0, 0.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(0.0, 2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(0.0, 0.0, 2.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(2.0, 2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(0.0, 2.0, 2.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(2.0, 0.0, 2.0), 0.9)
            .intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(-2.0, 0.0, 0.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, -2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, 0.0, -2.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(-2.0, -2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(0.0, -2.0, -2.0), 1.5).intersect_bool(bb));
        assert!(SmallSphereProbe::new(vec3(-2.0, 0.0, -2.0), 1.5).intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(-2.0, 0.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(0.0, -2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(0.0, 0.0, -2.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(-2.0, -2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(0.0, -2.0, -2.0), 0.9)
            .intersect_bool(bb));
        assert!(!SmallSphereProbe::new(vec3(-2.0, 0.0, -2.0), 0.9)
            .intersect_bool(bb));
    }

    #[test]
    fn large_sphere_probe() {
        let bb = Aabb::of_points(&[vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0)]);

        assert!(LargeSphereProbe::new(vec3(0.0, 0.0, 0.0), 1.0).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, 0.0, 0.0), 0.1).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, 0.0, 0.0), 10.0).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(2.0, 0.0, 0.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, 2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, 0.0, 2.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(2.0, 2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, 2.0, 2.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(2.0, 0.0, 2.0), 1.5).intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(2.0, 0.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(0.0, 2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(0.0, 0.0, 2.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(2.0, 2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(0.0, 2.0, 2.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(2.0, 0.0, 2.0), 0.9)
            .intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(-2.0, 0.0, 0.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, -2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, 0.0, -2.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(-2.0, -2.0, 0.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(0.0, -2.0, -2.0), 1.5).intersect_bool(bb));
        assert!(LargeSphereProbe::new(vec3(-2.0, 0.0, -2.0), 1.5).intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(-2.0, 0.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(0.0, -2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(0.0, 0.0, -2.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(-2.0, -2.0, 0.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(0.0, -2.0, -2.0), 0.9)
            .intersect_bool(bb));
        assert!(!LargeSphereProbe::new(vec3(-2.0, 0.0, -2.0), 0.9)
            .intersect_bool(bb));
    }
}
