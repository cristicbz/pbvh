use cgmath::{Vector3, vec3};
use std::f32;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Aabb {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

impl Aabb {
    #[inline]
    pub fn from_unchecked_min_max(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        Aabb { min, max }
    }

    #[inline]
    pub fn clip_axis_min(&self, axis: u32, limit: f32) -> Self {
        let mut clipped = *self;
        clipped.min[axis as usize] = limit;
        clipped
    }

    #[inline]
    pub fn clip_axis_max(&self, axis: u32, limit: f32) -> Self {
        let mut clipped = *self;
        clipped.max[axis as usize] = limit;
        clipped
    }

    #[inline]
    pub fn negative() -> Self {
        Aabb {
            min: vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: vec3(-f32::INFINITY, -f32::INFINITY, -f32::INFINITY),
        }
    }

    #[inline]
    pub fn of_sphere(position: Vector3<f32>, radius: f32) -> Self {
        assert!(radius > 0.0);
        let radius = vec3(radius, radius, radius);
        Aabb {
            min: position - radius,
            max: position + radius,
        }
    }

    pub fn of_points(points: &[Vector3<f32>]) -> Self {
        let mut aabb = Aabb::negative();
        for point in points {
            aabb.min[0] = aabb.min[0].min(point[0]);
            aabb.min[1] = aabb.min[1].min(point[1]);
            aabb.min[2] = aabb.min[2].min(point[2]);

            aabb.max[0] = aabb.max[0].max(point[0]);
            aabb.max[1] = aabb.max[1].max(point[1]);
            aabb.max[2] = aabb.max[2].max(point[2]);
        }
        aabb
    }

    #[inline]
    pub fn union<I: IntoIterator<Item = Aabb>>(bbs: I) -> Self {
        bbs.into_iter().fold(Aabb::negative(), Aabb::merge)
    }

    #[inline]
    pub fn area(&self) -> f32 {
        let edges = self.max - self.min;
        2.0 * (edges[0] * edges[1] + edges[1] * edges[2] + edges[0] * edges[2])
    }

    #[inline]
    pub fn intersects_sphere(&self, position: Vector3<f32>, radius: f32) -> bool {
        self.min[0] - position[0] <= radius && self.min[1] - position[1] <= radius &&
            self.min[2] - position[2] <= radius &&
            position[0] - self.max[0] <= radius && position[1] - self.max[1] <= radius &&
            position[2] - self.max[2] <= radius
    }

    #[inline]
    pub fn longest_axis(&self) -> usize {
        let diagonal = self.max - self.min;
        if diagonal[0] > diagonal[1] {
            if diagonal[0] > diagonal[2] { 0 } else { 2 }
        } else {
            if diagonal[1] > diagonal[2] { 1 } else { 2 }
        }
    }

    #[inline]
    pub fn centroid(&self) -> Vector3<f32> {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn min(&self) -> Vector3<f32> {
        self.min
    }

    #[inline]
    pub fn max(&self) -> Vector3<f32> {
        self.max
    }

    #[inline]
    pub fn merge(mut self, other: Aabb) -> Self {
        self.add_aabb(other);
        self
    }

    #[inline]
    pub fn add_aabb(&mut self, other: Aabb) {
        let Aabb {
            ref mut min,
            ref mut max,
        } = *self;
        let Aabb {
            min: other_min,
            max: other_max,
        } = other;
        min.x = min.x.min(other_min.x);
        min.y = min.y.min(other_min.y);
        min.z = min.z.min(other_min.z);

        max.x = max.x.max(other_max.x);
        max.y = max.y.max(other_max.y);
        max.z = max.z.max(other_max.z);
    }
}

#[cfg(test)]
mod tests {
    use super::Aabb;
    use cgmath::{Vector3, vec3, Zero};

    #[test]
    fn aabb_union() {
        let aabb = Aabb::union(
            [
                Aabb::of_sphere(Vector3::zero(), 1.0),
                Aabb::of_sphere(Vector3::zero(), 2.0),
                Aabb::of_sphere(vec3(1.0, 0.0, 0.0), 1.0),
                Aabb::of_sphere(vec3(-1.0, 0.0, 0.0), 2.0),
                Aabb::of_sphere(vec3(1.0, 2.0, 0.0), 1.0),
            ].iter()
                .cloned(),
        );

        assert_eq!(aabb.min(), vec3(-3.0, -2.0, -2.0));
        assert_eq!(aabb.max(), vec3(2.0, 3.0, 2.0));
    }
}
