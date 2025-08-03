use core::panic;
use std::{
    cmp::max,
    f64::consts::PI,
    ops::{Deref, Index, IndexMut, Mul, Neg},
    path::Path,
};

use cgmath::{
    BaseFloat, BaseNum, Basis2, Basis3, InnerSpace, Matrix, Matrix2, Matrix3, Matrix4, Rad,
    Rotation, Rotation2, Rotation3, SquareMatrix, Vector2, Vector3, Vector4, VectorSpace, Zero,
    num_traits::{Float, clamp, clamp_max},
    vec2, vec3, vec4,
};
use image::{ImageError, Pixel, Rgba, RgbaImage};

#[derive(Debug, Clone, Copy)]
struct HalfEdge<T: Float> {
    index: usize,
    prev: usize,
    next: usize,
    twin: usize,
    vertex: Vector3<T>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HalfEdgeIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq)]
struct HalfThroatIndex(usize);

impl Deref for HalfEdgeIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for HalfThroatIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Mesh<T: Float> {
    half_edges: Vec<HalfEdge<T>>,
}

impl<T: Float> Index<HalfEdgeIndex> for Mesh<T> {
    type Output = HalfEdge<T>;

    fn index(&self, index: HalfEdgeIndex) -> &Self::Output {
        &self.half_edges[index.0]
    }
}

struct Ambient {
    half_throat_indices: Vec<usize>,
    background: RgbaSkybox,
}

struct Environment<T: Float> {
    meshes: Vec<Mesh<T>>,
    half_throats: Vec<HalfThroat<T>>,
    ambients: Vec<Ambient>,
}

struct HalfThroatIntersection<T: Float> {
    half_throat_index: usize,
    mesh_intersection: MeshIntersection<T>,
}

impl<T: BaseFloat> Environment<T> {
    fn find_closest_half_throat_intersection(
        &self,
        ray: SituatedQV3<T>,
    ) -> Option<HalfThroatIntersection<T>> {
        let ChartIndex::Ambient(ambient_index) = ray.chart_index else {
            panic!()
        };
        let ambient = &self.ambients[ambient_index];

        let mut closest_intersection: Option<HalfThroatIntersection<T>> = None;

        for &half_throat_index in &ambient.half_throat_indices {
            let ht = &self.half_throats[half_throat_index];
            let throat_local_pos = (ht.gtl * ray.pos.extend(T::one())).truncate();
            let throat_local_vel = (ht.gtl * ray.vel.extend(T::zero())).truncate();

            let mesh = &self.meshes[ht.mesh_index];
            let Some(mi) = mesh.intersect_ray(throat_local_pos, throat_local_vel) else {
                continue;
            };

            let smaller_t = match closest_intersection {
                Some(ref hti) => mi.tuv.x < hti.mesh_intersection.tuv.x,
                None => true,
            };

            if smaller_t {
                closest_intersection = Some(HalfThroatIntersection {
                    half_throat_index,
                    mesh_intersection: mi,
                })
            }
        }
        closest_intersection
    }

    fn process_ambient_ray(&self, ray: SituatedQV3<T>) -> SituatedQV3<T> {
        // unchecked
        let intersection = self.find_closest_half_throat_intersection(ray);
        match intersection {
            Some(hti) => self.half_throat_entry(ray, hti),
            None => ray,
        }
    }

    fn half_throat_entry(
        &self,
        global_ray: SituatedQV3<T>,
        hti: HalfThroatIntersection<T>,
    ) -> SituatedQV3<T> {
        let ht = &self.half_throats[hti.half_throat_index];
        let throat_local_4vel = ht.gtl * global_ray.vel.extend(T::zero());
        let mesh = &self.meshes[ht.mesh_index];
        let he = mesh.half_edges[hti.mesh_intersection.half_edge_index];
        let local_verts = mesh.half_edge_to_local_vertices(he);
        let tuv = hti.mesh_intersection.tuv;
        let local_pos = (local_verts[1] * tuv.y + local_verts[2] * tuv.z).extend(T::zero());
        let local_vel = mesh.jac_pinv(he, local_pos) * throat_local_4vel;
        SituatedQV3 {
            chart_index: ChartIndex::HalfThroat(
                HalfThroatIndex(hti.half_throat_index),
                HalfEdgeIndex(he.index),
            ),
            pos: local_pos,
            vel: local_vel,
        }
    }

    fn half_throat_exit(&self, throat_patch_ray: SituatedQV3<T>) -> SituatedQV3<T> {
        let ChartIndex::HalfThroat(HalfThroatIndex(hti), HalfEdgeIndex(hei)) =
            throat_patch_ray.chart_index
        else {
            panic!()
        };
        let ht = &self.half_throats[hti];
        let mesh = &self.meshes[ht.mesh_index];
        let he = mesh.half_edges[hei];
        let throat_local_4pos = mesh.uvt_to_4pos(he, throat_patch_ray.pos);
        let throat_local_4vel = mesh.jac(he, throat_patch_ray.pos) * throat_patch_ray.vel;
        let throat_local_pos = throat_local_4pos.truncate().extend(T::one()); // we assume throat_local_4pos.w is 0, i.e. we're in the flat region
        let throat_local_vel = throat_local_4vel.truncate().extend(T::zero()); // same as above
        let global_pos = (ht.ltg * throat_local_pos).truncate();
        let global_vel = (ht.ltg * throat_local_vel).truncate();
        SituatedQV3 {
            chart_index: ChartIndex::Ambient(ht.ambient_index),
            pos: global_pos,
            vel: global_vel,
        }
    }
    fn mid_throat_entry(&self, throat_patch_ray: SituatedQV3<T>) -> SituatedQV3<T> {
        let ChartIndex::HalfThroat(hti, hei) = throat_patch_ray.chart_index else {
            panic!()
        };
        SituatedQV3 {
            chart_index: ChartIndex::MidThroat(hti, hei),
            ..throat_patch_ray
        }
    }
    fn mid_throat_exit(&self, throat_patch_ray: SituatedQV3<T>) -> SituatedQV3<T> {
        let ChartIndex::MidThroat(hti, hei) = throat_patch_ray.chart_index else {
            panic!()
        };
        SituatedQV3 {
            chart_index: ChartIndex::HalfThroat(hti, hei),
            ..throat_patch_ray
        }
    }
    fn mid_throat_transition(&self, throat_patch_ray: SituatedQV3<T>) -> SituatedQV3<T> {
        let ChartIndex::MidThroat(hti, hei) = throat_patch_ray.chart_index else {
            panic!()
        };
        let ht = &self.half_throats[hti.0];
        let pos = throat_patch_ray
            .pos
            .truncate()
            .extend(T::from(2).unwrap() * ht.mid_t - throat_patch_ray.pos.z);
        let mut vel = throat_patch_ray.vel;
        vel.z *= -T::one();
        SituatedQV3 {
            chart_index: ChartIndex::MidThroat(HalfThroatIndex(ht.twin_index), hei),
            pos,
            vel,
        }
    }
    fn midthroat_traverse(
        &self,
        throat_patch_ray: SituatedQV3<T>,
        time_bound: T,
    ) -> Option<(SituatedQV3<T>, T)> {
        let ChartIndex::MidThroat(hti, hei) = throat_patch_ray.chart_index else {
            panic!()
        };
        let ht = &self.half_throats[hti.0];
        let hi_target_t = T::from(2).unwrap() * ht.mid_t - T::one();
        let lo_target_t = T::one();
        if throat_patch_ray.vel.z.is_zero() {
            return None;
        }
        let target_t = if throat_patch_ray.vel.z.is_sign_positive() {
            hi_target_t
        } else {
            lo_target_t
        };
        let dt = (target_t - throat_patch_ray.pos.z) / throat_patch_ray.vel.z;
        if dt.abs() > time_bound {
            return None;
        }
        let mesh = &self.meshes[ht.mesh_index];
        let mesh_qv0 = mesh.project_uvt_to_mesh(throat_patch_ray);
        let mesh_qv1 = mesh.exp(mesh_qv0, dt);
        let q1 = mesh_qv1.pos.extend(target_t);
        let v1 = mesh_qv1.vel.extend(throat_patch_ray.vel.z);
        let (mesh_qv1_hti, mesh_qv1_hei) = match mesh_qv1.chart_index {
            ChartIndex::MeshLocal2DTriangle(hti, hei) => (hti, hei),
            _ => panic!(),
        };
        let qv1 = SituatedQV3 {
            chart_index: ChartIndex::MidThroat(mesh_qv1_hti, mesh_qv1_hei),
            pos: q1,
            vel: v1,
        };
        let correct_side_qv1 = if qv1.vel.z.is_sign_positive() {
            self.mid_throat_transition(qv1)
        } else {
            qv1
        };
        Some((self.mid_throat_exit(correct_side_qv1), time_bound - dt))
    }
    fn halfthroat_step(&self, throat_patch_ray: SituatedQV3<T>, dt: T) -> SituatedQV3<T> {
        let (hti, _) = throat_patch_ray.chart_index.throat_indices().unwrap();
        let ht = &self.half_throats[hti.0];
        let mesh = &self.meshes[ht.mesh_index];
        let k1 = mesh.phase_vel(throat_patch_ray);
        let quasi_qv0 = SituatedQV3 {
            pos: throat_patch_ray.pos,
            vel: k1.pos,
            ..throat_patch_ray
        };
        let mesh_quasi_qv0 = mesh.project_uvt_to_mesh(quasi_qv0);
        let mesh_quasi_qv1 = mesh.exp(mesh_quasi_qv0, dt);
        let rot = Basis2::from_angle(mesh_quasi_qv0.vel.angle(mesh_quasi_qv1.vel));
        let q1 = mesh_quasi_qv1
            .pos
            .extend(throat_patch_ray.pos.z + k1.pos.z * dt);
        let v1_old_tri = throat_patch_ray.vel + k1.vel * dt;
        let v1_uv_rotated = rot.rotate_vector(v1_old_tri.truncate());
        let v1 = v1_uv_rotated.extend(v1_old_tri.z);
        let (_, new_hei) = mesh_quasi_qv1.chart_index.throat_indices().unwrap();
        let prelim = SituatedQV3 {
            chart_index: ChartIndex::HalfThroat(hti, new_hei),
            pos: q1,
            vel: v1,
        };
        if prelim.pos.z > ht.throat_to_mid_t {
            self.mid_throat_entry(prelim)
        } else if prelim.pos.z.is_sign_negative() {
            self.half_throat_exit(prelim)
        } else {
            prelim
        }
    }
    fn push_ray_step(
        &self,
        ray: SituatedQV3<T>,
        time_bound: T,
        dt: T,
    ) -> Option<(SituatedQV3<T>, T)> {
        match ray.chart_index {
            ChartIndex::Ambient(_) => Some((self.process_ambient_ray(ray), time_bound)),
            ChartIndex::HalfThroat(_, _) => {
                if dt >= time_bound {
                    None
                } else {
                    Some((self.halfthroat_step(ray, dt), time_bound - dt))
                }
            }
            ChartIndex::MidThroat(_, _) => self.midthroat_traverse(ray, time_bound),
            _ => panic!(),
        }
    }
    fn push_ray(
        &self,
        ray: SituatedQV3<T>,
        time_bound: T,
        iter_bound: u32,
        dt: T,
    ) -> Option<SituatedQV3<T>> {
        let mut cur_ray = ray;
        let mut cur_time_remaining = time_bound;
        for _ in 0..iter_bound {
            let (new_ray, new_time_remaining) =
                self.push_ray_step(cur_ray, cur_time_remaining, dt)?;
            if cur_ray == new_ray {
                return Some(cur_ray);
            }
            cur_time_remaining = new_time_remaining;
            cur_ray = new_ray;
        }
        Some(cur_ray)
    }
    fn ray_color(&self, ray: SituatedQV3<T>) -> Option<Rgba<u8>> {
        let ChartIndex::Ambient(ai) = ray.chart_index else {
            return None;
        };
        Some(self.ambients[ai].background.sample(ray.vel))
    }
}

// Each of these should be a separate type instead, ChartIndex should be a trait,
// and being throat-bound should also be a trait
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChartIndex {
    Ambient(usize),
    HalfThroat(HalfThroatIndex, HalfEdgeIndex),
    MidThroat(HalfThroatIndex, HalfEdgeIndex),
    MeshLocal2DTriangle(HalfThroatIndex, HalfEdgeIndex),
}

impl ChartIndex {
    fn throat_indices(self) -> Option<(HalfThroatIndex, HalfEdgeIndex)> {
        match self {
            ChartIndex::Ambient(_) => None,
            ChartIndex::HalfThroat(half_throat_index, half_edge_index) => {
                Some((half_throat_index, half_edge_index))
            }
            ChartIndex::MidThroat(half_throat_index, half_edge_index) => {
                Some((half_throat_index, half_edge_index))
            }
            ChartIndex::MeshLocal2DTriangle(half_throat_index, half_edge_index) => {
                Some((half_throat_index, half_edge_index))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SituatedQV2<T: Float> {
    chart_index: ChartIndex,
    pos: Vector2<T>,
    vel: Vector2<T>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SituatedQV3<T: Float> {
    chart_index: ChartIndex,
    pos: Vector3<T>,
    vel: Vector3<T>,
}

impl<T: BaseFloat> Mul<T> for SituatedQV3<T> {
    type Output = SituatedQV3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        SituatedQV3 {
            chart_index: self.chart_index,
            pos: self.pos * rhs,
            vel: self.vel * rhs,
        }
    }
}

struct SituatedQV4<T: Float> {
    chart_index: ChartIndex,
    pos: Vector4<T>,
    vel: Vector4<T>,
}

struct HalfThroat<T: Float> {
    index: usize,
    twin_index: usize,
    mesh_index: usize,
    ambient_index: usize,
    gtl: Matrix4<T>,
    ltg: Matrix4<T>,
    mid_t: T,           // other side must have same mid_t
    throat_to_mid_t: T, // mid_t > throat_to_mid_t > 1
}

#[derive(Debug, Clone, Copy)]
struct Affine2<T: Float> {
    a: Matrix2<T>,
    b: Vector2<T>,
}

impl<T: BaseFloat> Mul for Affine2<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Affine2 {
            a: self.a * rhs.a,
            b: self.a * rhs.b + self.b,
        }
    }
}

impl<T: BaseFloat> Mul<Vector2<T>> for Affine2<T> {
    type Output = Vector2<T>;

    fn mul(self, rhs: Vector2<T>) -> Self::Output {
        self.a * rhs + self.b
    }
}

impl<T: BaseFloat> Mul<SituatedQV2<T>> for Affine2<T> {
    type Output = SituatedQV2<T>;

    fn mul(self, rhs: SituatedQV2<T>) -> Self::Output {
        SituatedQV2 {
            chart_index: rhs.chart_index,
            pos: self * rhs.pos,
            vel: self.a * rhs.vel,
        }
    }
}

fn ray_segment_intersection<T: BaseFloat>(
    q: Vector2<T>,
    v: Vector2<T>,
    a: Vector2<T>,
    b: Vector2<T>,
) -> Option<(T, T)> {
    let qt = q - a;
    let bt = b - a;
    let lambda_denom = v.perp_dot(b);
    if lambda_denom.is_zero() {
        return None;
    }
    let lambda = -qt.perp_dot(bt) / lambda_denom;
    if lambda.is_sign_negative() {
        return None;
    }
    let l = (qt.dot(bt) + lambda * v.dot(bt)) / bt.dot(bt);
    if l.is_sign_negative() || (T::one() - l).is_sign_negative() {
        return None;
    }
    // q + lambda * v = (1-l) * a + l * b
    Some((lambda, l))
}

fn cubic<T: BaseFloat>(points: [Vector2<T>; 4], t: T) -> Vector2<T> {
    let mut a = points[0].lerp(points[1], t);
    let mut b = points[1].lerp(points[2], t);
    let c = points[2].lerp(points[3], t);
    a = a.lerp(b, t);
    b = b.lerp(c, t);
    a = a.lerp(b, t);
    a
}

fn cubic_velocity<T: BaseFloat>(points: [Vector2<T>; 4], t: T) -> Vector2<T> {
    let mut a = points[1] - points[0];
    let mut b = points[2] - points[1];
    let c = points[3] - points[2];
    a = a.lerp(b, t);
    b = b.lerp(c, t);
    a = a.lerp(b, t);
    a * T::from(3).unwrap()
}

fn cubic_accel<T: BaseFloat>(points: [Vector2<T>; 4], t: T) -> Vector2<T> {
    let mut a = points[2] - points[1] * T::from(2).unwrap() + points[0];
    let b = points[3] - points[2] * T::from(2).unwrap() + points[1];
    a = a.lerp(b, t);
    a * T::from(6).unwrap()
}

fn simple_cubic_points<T: BaseFloat>() -> [Vector2<T>; 4] {
    let half = T::from(0.5).unwrap();
    [
        vec2(T::one(), T::zero()),
        vec2(half, T::zero()),
        vec2(half, T::zero()),
        vec2(half, half),
    ]
}

fn simple_cubic<T: BaseFloat>(t: T) -> Vector2<T> {
    cubic(simple_cubic_points(), t)
}

fn simple_cubic_velocity<T: BaseFloat>(t: T) -> Vector2<T> {
    cubic_velocity(simple_cubic_points(), t)
}

fn simple_cubic_accel<T: BaseFloat>(t: T) -> Vector2<T> {
    cubic_accel(simple_cubic_points(), t)
}

fn extended_cubic<T: BaseFloat>(t: T) -> Vector2<T> {
    if t.is_sign_negative() {
        simple_cubic_velocity(T::zero()) * t + simple_cubic(T::zero())
    } else if t > T::one() {
        simple_cubic_velocity(T::one()) * t + simple_cubic(T::one())
    } else {
        simple_cubic(t)
    }
}

fn extended_cubic_velocity<T: BaseFloat>(t: T) -> Vector2<T> {
    simple_cubic_velocity(clamp(t, T::zero(), T::one()))
}

fn extended_cubic_accel<T: BaseFloat>(t: T) -> Vector2<T> {
    if t < T::zero() || t > T::one() {
        Vector2::zero()
    } else {
        simple_cubic_accel(t)
    }
}

#[derive(Debug, Clone, Copy)]
struct Matrix4x3<T> {
    x: Vector4<T>,
    y: Vector4<T>,
    z: Vector4<T>,
}

#[derive(Debug, Clone, Copy)]
struct Matrix3x4<T> {
    x: Vector3<T>,
    y: Vector3<T>,
    z: Vector3<T>,
    w: Vector3<T>,
}

impl<T: Copy> Matrix3x4<T> {
    fn transpose(&self) -> Matrix4x3<T> {
        Matrix4x3 {
            x: vec4(self.x.x, self.y.x, self.z.x, self.w.x),
            y: vec4(self.x.y, self.y.y, self.z.y, self.w.y),
            z: vec4(self.x.z, self.y.z, self.z.z, self.w.z),
        }
    }
}

impl<T: Copy> Matrix4x3<T> {
    fn transpose(&self) -> Matrix3x4<T> {
        Matrix3x4 {
            x: vec3(self.x.x, self.y.x, self.z.x),
            y: vec3(self.x.y, self.y.y, self.z.y),
            z: vec3(self.x.z, self.y.z, self.z.z),
            w: vec3(self.x.w, self.y.w, self.z.w),
        }
    }
}

impl<T: BaseFloat> Mul<Matrix3x4<T>> for Matrix3<T> {
    type Output = Matrix3x4<T>;

    fn mul(self, rhs: Matrix3x4<T>) -> Self::Output {
        Matrix3x4 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            w: self * rhs.w,
        }
    }
}

impl<T: BaseFloat> Mul<Vector4<T>> for Matrix3x4<T> {
    type Output = Vector3<T>;

    fn mul(self, rhs: Vector4<T>) -> Self::Output {
        vec3(
            self.x.x * rhs.x + self.y.x * rhs.y + self.z.x * rhs.z + self.w.x * rhs.w,
            self.x.y * rhs.x + self.y.y * rhs.y + self.z.y * rhs.z + self.w.y * rhs.w,
            self.x.z * rhs.x + self.y.z * rhs.y + self.z.z * rhs.z + self.w.z * rhs.w,
        )
    }
}

impl<T: BaseFloat> Neg for Matrix3x4<T> {
    type Output = Matrix3x4<T>;

    fn neg(self) -> Self::Output {
        Matrix3x4 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl<T: BaseFloat> Mul<Vector3<T>> for Matrix4x3<T> {
    type Output = Vector4<T>;

    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        vec4(
            self.x.x * rhs.x + self.y.x * rhs.y + self.z.x * rhs.z,
            self.x.y * rhs.x + self.y.y * rhs.y + self.z.y * rhs.z,
            self.x.z * rhs.x + self.y.z * rhs.y + self.z.z * rhs.z,
            self.x.w * rhs.x + self.y.w * rhs.y + self.z.w * rhs.z,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct MeshIntersection<T: Float> {
    tuv: Vector3<T>,
    half_edge_index: usize,
}

fn ray_triangle_intersect<T: BaseFloat>(
    vertices: [Vector3<T>; 3],
    pos: Vector3<T>,
    vel: Vector3<T>,
) -> Option<Vector3<T>> {
    // tuv
    let e1 = vertices[1] - vertices[0];
    let e2 = vertices[2] - vertices[0];

    let h = vel.cross(e2);
    let a = e1.dot(h);

    let small: T = T::from(1e-8).unwrap();

    // Ray is parallel to triangle
    if a.abs() < small {
        return None;
    }

    let f = a.recip();
    let s = pos - vertices[0];
    let u = f * s.dot(h);

    if u.is_sign_negative() || u > T::one() {
        return None;
    }

    let q = s.cross(e1);
    let v = f * vel.dot(q);

    if v.is_sign_negative() || u + v > T::one() {
        return None;
    }

    let t = f * e2.dot(q);
    if t > small { Some(vec3(t, u, v)) } else { None }
}

impl<T: BaseFloat> Mesh<T> {
    fn next_half_edge(&self, he: HalfEdge<T>) -> HalfEdge<T> {
        self.half_edges[he.next]
    }
    fn prev_half_edge(&self, he: HalfEdge<T>) -> HalfEdge<T> {
        self.half_edges[he.prev]
    }
    fn twin_half_edge(&self, he: HalfEdge<T>) -> HalfEdge<T> {
        self.half_edges[he.twin]
    }
    fn half_edge_to_local_vertices(&self, he: HalfEdge<T>) -> [Vector2<T>; 3] {
        let next = self.next_half_edge(he);
        let prev = self.prev_half_edge(he);
        let e1 = next.vertex - he.vertex;
        let e2 = prev.vertex - he.vertex;
        let base_length = e1.magnitude();
        let p1 = Vector2::new(base_length, T::zero());
        let p2 = Vector2::new(e1.cross(e2).z, e1.dot(e2)) / base_length;
        [Vector2::zero(), p1, p2]
    }
    fn half_edge_to_vertices(&self, he: HalfEdge<T>) -> [Vector3<T>; 3] {
        [
            he.vertex,
            self.next_half_edge(he).vertex,
            self.prev_half_edge(he).vertex,
        ]
    }
    fn half_edge_to_embedded_basis(&self, he: HalfEdge<T>) -> Matrix4<T> {
        // maybe should be Affine4 instead
        let next = self.next_half_edge(he);
        let prev = self.prev_half_edge(he);
        let e1 = (next.vertex - he.vertex).normalize();
        let d2 = prev.vertex - he.vertex;
        let e2 = (d2 - d2.project_on(e1)).normalize();
        let e3 = e1.cross(e2);
        Matrix4 {
            x: Vector4 {
                x: e1.x,
                y: e1.y,
                z: e1.z,
                w: T::zero(),
            },
            y: Vector4 {
                x: e2.x,
                y: e2.y,
                z: e2.z,
                w: T::zero(),
            },
            z: Vector4 {
                x: e3.x,
                y: e3.y,
                z: e3.z,
                w: T::zero(),
            },
            w: Vector4 {
                x: he.vertex.x,
                y: he.vertex.y,
                z: he.vertex.z,
                w: T::one(),
            },
        }
    }
    fn exp(&self, qv: SituatedQV2<T>, dt: T) -> SituatedQV2<T> {
        let ChartIndex::MeshLocal2DTriangle(half_throat_index, half_edge_index) = qv.chart_index
        else {
            panic!("called exp on mesh with wrongly indexed qv2")
        };
        if dt.is_zero() {
            return qv;
        }
        let half_edge = self[half_edge_index];
        let verts = self.half_edge_to_local_vertices(half_edge);
        let twins = [
            self.twin_half_edge(half_edge),
            self.twin_half_edge(self.next_half_edge(half_edge)),
            self.twin_half_edge(self.prev_half_edge(half_edge)),
        ];
        let (q, v) = (qv.pos, qv.vel);
        for i in 0..3 {
            let (a, b) = (verts[i], verts[(i + 1) % 3]);
            let sect = ray_segment_intersection(q, v, a, b);
            let Some((lambda, l)) = sect else { continue };
            if lambda > dt {
                continue;
            };
            let nx = (a - b).normalize();
            let new_v = Vector2::new(nx.dot(v), nx.perp_dot(v));
            let new_q = Vector2::new((T::one() - l) * nx.dot(a - b), T::zero());
            let new_index =
                ChartIndex::MeshLocal2DTriangle(half_throat_index, HalfEdgeIndex(twins[i].index));
            let new_sqv = SituatedQV2 {
                chart_index: new_index,
                pos: new_q,
                vel: new_v,
            };
            return self.exp(new_sqv, dt - lambda);
        }
        let new_q = q + v * dt;
        SituatedQV2 {
            chart_index: qv.chart_index,
            pos: new_q,
            vel: qv.vel,
        }
    }
    fn intersect_ray(&self, pos: Vector3<T>, vel: Vector3<T>) -> Option<MeshIntersection<T>> {
        let mut closest_intersection = None;
        let mut closest_t: Option<T> = None;

        for he in self.half_edges.iter().step_by(3) {
            let verts = self.half_edge_to_vertices(*he);
            let Some(tuv) = ray_triangle_intersect(verts, pos, vel) else {
                continue;
            };

            let smaller_t = match closest_t {
                Some(v) => tuv.x < v,
                None => true,
            };
            if smaller_t {
                closest_t = Some(tuv.x);
                closest_intersection = Some(MeshIntersection {
                    tuv,
                    half_edge_index: he.index,
                });
            }
        }
        closest_intersection
    }
    fn jac_tilde(&self, he: HalfEdge<T>, t: T) -> Matrix4x3<T> {
        let eb = self.half_edge_to_embedded_basis(he);
        let vel = extended_cubic_velocity(t);
        let p = eb.z * eb.z.dot(eb.w);
        let c2 = p * vel.x + vec4(T::zero(), T::zero(), T::zero(), vel.y);
        Matrix4x3 {
            x: eb.x,
            y: eb.y,
            z: c2,
        }
    }
    fn jac_tilde_pinv(&self, he: HalfEdge<T>, t: T) -> Matrix3x4<T> {
        let mut jt = self.jac_tilde(he, t);
        jt.z /= jt.z.magnitude2();
        jt.transpose()
    }
    fn ortho_to_uvt(&self, he: HalfEdge<T>, uvt: Vector3<T>) -> Matrix3<T> {
        let eb = self.half_edge_to_embedded_basis(he);
        let sc = extended_cubic(uvt.z);
        let vel = extended_cubic_velocity(uvt.z);
        let coef = -vel.x / sc.x;
        let inv_scale = T::one() / sc.x;
        let a = eb.w.dot(eb.x) + uvt.x;
        let b = eb.w.dot(eb.y) + uvt.y;
        Matrix3 {
            x: vec3(inv_scale, T::zero(), T::zero()),
            y: vec3(T::zero(), inv_scale, T::zero()),
            z: vec3(a * coef, b * coef, T::one()),
        }
    }
    fn jac(&self, he: HalfEdge<T>, uvt: Vector3<T>) -> Matrix4x3<T> {
        let eb = self.half_edge_to_embedded_basis(he);
        let sc = extended_cubic(uvt.z);
        let vel = extended_cubic_velocity(uvt.z);
        let mesh_pos = eb.w + eb.x * uvt.x + eb.y * uvt.y;
        let col3_part = mesh_pos.truncate() * vel.x;
        Matrix4x3 {
            x: eb.x * sc.x,
            y: eb.y * sc.x,
            z: col3_part.extend(vel.y),
        }
    }
    fn jac_pinv(&self, he: HalfEdge<T>, uvt: Vector3<T>) -> Matrix3x4<T> {
        self.ortho_to_uvt(he, uvt) * self.jac_tilde_pinv(he, uvt.z)
    }
    fn djac_dt(&self, he: HalfEdge<T>, uvt: Vector3<T>, uvt_vel: Vector3<T>) -> Matrix4x3<T> {
        let vel = extended_cubic_velocity(uvt.z);
        let acc = extended_cubic_accel(uvt.z);
        let eb = self.half_edge_to_embedded_basis(he);
        let mesh_pos = (eb.w + eb.x * uvt.x + eb.y * uvt.y).truncate();
        Matrix4x3 {
            x: eb.x * vel.x * uvt_vel.z,
            y: eb.y * vel.x * uvt_vel.z,
            z: (mesh_pos * acc.x * uvt_vel.z
                + eb.x.truncate() * vel.x * uvt_vel.x
                + eb.y.truncate() * vel.x * uvt_vel.y)
                .extend(acc.y * uvt_vel.z),
        }
    }
    fn uvt_to_4pos(&self, he: HalfEdge<T>, uvt: Vector3<T>) -> Vector4<T> {
        let eb = self.half_edge_to_embedded_basis(he);
        let sc = extended_cubic(uvt.z);
        let mesh_pos = eb.w + eb.x * uvt.x + eb.y * uvt.y;
        (mesh_pos.truncate() * sc.x).extend(sc.y)
    }
    fn project_uvt_to_mesh(&self, throat_patch_ray: SituatedQV3<T>) -> SituatedQV2<T> {
        let (hti, hei) = match throat_patch_ray.chart_index {
            ChartIndex::HalfThroat(hti, hei) => (hti, hei),
            ChartIndex::MidThroat(hti, hei) => (hti, hei),
            _ => panic!(),
        };
        let chart_index = ChartIndex::MeshLocal2DTriangle(hti, hei);
        SituatedQV2 {
            chart_index,
            pos: throat_patch_ray.pos.truncate(),
            vel: throat_patch_ray.vel.truncate(),
        }
    }
    fn phase_vel(&self, ray: SituatedQV3<T>) -> SituatedQV3<T> {
        let (_, hei) = ray.chart_index.throat_indices().unwrap();
        let he = self.half_edges[hei.0];
        let djdt = self.djac_dt(he, ray.pos, ray.vel);
        let jpinv = self.jac_pinv(he, ray.pos);
        SituatedQV3 {
            chart_index: ray.chart_index,
            pos: ray.vel,
            vel: -(jpinv * (djdt * ray.vel)),
        }
    }
}

// Has to have equal resolution on all faces
#[derive(Debug, Clone)]
struct RgbaSkybox {
    px: RgbaImage,
    nx: RgbaImage,
    py: RgbaImage,
    ny: RgbaImage,
    pz: RgbaImage,
    nz: RgbaImage,
}

impl RgbaSkybox {
    fn load_from_path(bg_path: &Path) -> Result<Self, ImageError> {
        let [px, nx, py, ny, pz, nz]: [Result<_, ImageError>; 6] =
            ["right", "left", "bottom", "top", "front", "back"].map(|x| {
                let mut im = image::open(bg_path.join(format!("{}.png", x)))?.into_rgba8();
                // image::imageops::flip_vertical_in_place(&mut im);
                Ok(im)
            });
        Ok(RgbaSkybox {
            px: px?,
            nx: nx?,
            py: py?,
            ny: ny?,
            pz: pz?,
            nz: nz?,
        })
    }
    fn sample<T: BaseFloat>(&self, ray: Vector3<T>) -> Rgba<u8> {
        let abs_ray = ray.map(|x| x.abs());
        let inf_mag = abs_ray.x.max(abs_ray.y.max(abs_ray.z));
        let n = ray / inf_mag;
        let (x, y, image) = if abs_ray.x == inf_mag {
            if n.x.is_sign_positive() {
                (-n.z, n.y, &self.px)
            } else {
                (n.z, n.y, &self.nx)
            }
        } else if abs_ray.y == inf_mag {
            if n.y.is_sign_positive() {
                (n.x, -n.z, &self.py)
            } else {
                (n.x, n.z, &self.ny)
            }
        } else {
            if n.z.is_sign_positive() {
                (n.x, n.y, &self.pz)
            } else {
                (-n.x, n.y, &self.nz)
            }
        };
        let half = T::from(0.5).unwrap();
        let pixel_x = ((x + T::one()) * half * T::from(image.width()).unwrap())
            .floor()
            .to_u32()
            .unwrap()
            .clamp(0, image.width() - 1);
        let pixel_y = ((y + T::one()) * half * T::from(image.height()).unwrap())
            .floor()
            .to_u32()
            .unwrap()
            .clamp(0, image.height() - 1);
        let pixel = image.get_pixel(pixel_x, pixel_y);
        *pixel
    }
}

fn tetrahedron() -> Vec<HalfEdge<f64>> {
    let verts = [
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(1.0, -1.0, -1.0),
        Vector3::new(-1.0, 1.0, -1.0),
        Vector3::new(-1.0, -1.0, 1.0),
    ];
    let faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];

    let mut half_edges = Vec::with_capacity(12);

    // Create half-edges for each face
    for face_idx in 0..4 {
        let face = faces[face_idx];
        let base_he_idx = face_idx * 3;

        for i in 0..3 {
            let he_idx = base_he_idx + i;
            let next_idx = base_he_idx + ((i + 1) % 3);
            let prev_idx = base_he_idx + ((i + 2) % 3);

            half_edges.push(HalfEdge {
                index: he_idx,
                prev: prev_idx,
                next: next_idx,
                twin: 0, // Will be set below
                vertex: verts[face[i]],
            });
        }
    }

    // Set up twin relationships
    let twins = [
        (0, 5),
        (1, 11),
        (2, 6),
        (3, 8),
        (4, 9),
        (5, 0),
        (6, 2),
        (7, 10),
        (8, 3),
        (9, 4),
        (10, 7),
        (11, 1),
    ];

    for (he_idx, twin_idx) in twins {
        half_edges[he_idx].twin = twin_idx;
    }

    half_edges
}

struct Camera<T: Float> {
    width: T,
    height: T,
    frame: Matrix3<T>,
    frame_inv: Matrix3<T>,
    centre: Vector3<T>,
    yfov: T,
    chart_index: ChartIndex,
}

impl<T: BaseFloat> Camera<T> {
    fn fragpos_to_ray(&self, pos: Vector2<T>) -> SituatedQV3<T> {
        let half = T::from(0.5).unwrap();
        let ray_coords = vec3(
            (pos.x / self.width - half) * (self.width / self.height),
            pos.y / self.height - half,
            half / (half * self.yfov).tan(),
        )
        .normalize();
        let ray = self.frame * ray_coords;
        SituatedQV3 {
            chart_index: self.chart_index,
            pos: self.centre,
            vel: ray,
        }
    }
    fn fragrays(&self) -> impl Iterator<Item = SituatedQV3<T>> {
        let iw = self.width.to_u32().unwrap();
        let ih = self.height.to_u32().unwrap();
        let w = (0..iw)
            .flat_map(move |x| (0..ih).map(move |y| (x, y)))
            .map(|(x, y)| vec2(T::from(x).unwrap(), T::from(y).unwrap()))
            .map(|fragpos| self.fragpos_to_ray(fragpos));
        w
    }
}

fn main() {
    let bg0 = RgbaSkybox::load_from_path(Path::new("textures/bg_debug")).unwrap();
    let bg1 = RgbaSkybox::load_from_path(Path::new("textures/bg0")).unwrap();
    let tet_edges = tetrahedron();
    let tet_mesh = Mesh {
        half_edges: tet_edges,
    };
    let ht0 = HalfThroat::<f64> {
        index: 0,
        twin_index: 1,
        mesh_index: 0,
        ambient_index: 0,
        gtl: Matrix4::identity(),
        ltg: Matrix4::identity(),
        mid_t: 2.0,
        throat_to_mid_t: 1.1,
    };
    let ht1 = HalfThroat::<f64> {
        index: 1,
        twin_index: 0,
        mesh_index: 0,
        ambient_index: 1,
        gtl: Matrix4::identity(),
        ltg: Matrix4::identity(),
        mid_t: 2.0,
        throat_to_mid_t: 1.1,
    };
    let a0 = Ambient {
        half_throat_indices: vec![0],
        background: bg0.clone(),
    };
    let a1 = Ambient {
        half_throat_indices: vec![1],
        background: bg1,
    };
    let tet_env = Environment {
        meshes: vec![tet_mesh],
        half_throats: vec![ht0, ht1],
        ambients: vec![a0, a1],
    };
    let empty_env = Environment::<f64> {
        meshes: vec![],
        half_throats: vec![],
        ambients: vec![Ambient {
            background: bg0.clone(),
            half_throat_indices: vec![],
        }],
    };
    let (width, height) = (768, 768);
    let rot = Basis3::from_axis_angle(vec3(1.0, 0.0, 0.0), Rad(PI / 2.0));
    let mrot = Matrix3::from(rot);
    let view_dir = vec3(-1.0, -1.0, 1.0).normalize();
    let persp = Matrix3::look_to_lh(view_dir, vec3(0.0, 1.0, 0.0)).transpose();
    let camera = Camera::<f64> {
        width: width as f64,
        height: height as f64,
        frame: persp,
        frame_inv: persp.invert().unwrap(),
        centre: vec3(2.0, 2.0, -2.0),
        yfov: PI / 3.0,
        chart_index: ChartIndex::Ambient(0),
    };
    let mut res_image = RgbaImage::new(width, height);
    println!("Hello, world!");
    for x in 0..width {
        for y in 0..height {
            let fragpos = vec2(x as f64 + 0.5, y as f64 + 0.5);
            let fragray = camera.fragpos_to_ray(fragpos);
            let pushed_ray = tet_env.push_ray(fragray, 50.0, 300, 0.01);
            let color = match pushed_ray {
                Some(qv) => tet_env.ray_color(qv).unwrap_or(Rgba([0u8, 0, 0, 255])),
                None => Rgba([0u8, 0, 0, 255]),
            };
            *res_image.get_pixel_mut(x, y) = color;
        }
    }
    res_image.save("wowee.png").unwrap();
}
