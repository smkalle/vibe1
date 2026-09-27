//! Just enough 3D math for a wireframe renderer.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

impl V3 {
    pub const UP: V3 = v3(0.0, 1.0, 0.0);

    pub fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: V3) -> V3 {
        v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn norm(self) -> V3 {
        let l = self.len();
        if l > 1e-9 { self * (1.0 / l) } else { self }
    }
    pub fn lerp(self, o: V3, t: f32) -> V3 {
        self + (o - self) * t
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        v3(-self.x, -self.y, -self.z)
    }
}

/// Rotation as an orthonormal basis: columns are the local x (forward), y (up), z (right) axes.
#[derive(Clone, Copy, Debug)]
pub struct Basis {
    pub f: V3,
    pub u: V3,
    pub r: V3,
}

impl Basis {
    /// Yaw about world up, then pitch (nose up positive), then roll about forward.
    pub fn from_euler(yaw: f32, pitch: f32, roll: f32) -> Basis {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();
        let f = v3(cy * cp, sp, sy * cp);
        let r0 = v3(-sy, 0.0, cy);
        let u0 = r0.cross(f);
        // roll rotates up/right about forward
        let u = u0 * cr + r0 * sr;
        let r = r0 * cr - u0 * sr;
        Basis { f, u, r }
    }
    pub fn apply(&self, p: V3) -> V3 {
        self.f * p.x + self.u * p.y + self.r * p.z
    }
}

pub fn rot_y(p: V3, a: f32) -> V3 {
    let (s, c) = a.sin_cos();
    v3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z)
}

pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn smoothstep(x: f32) -> f32 {
    let t = clamp01(x);
    t * t * (3.0 - 2.0 * t)
}

/// Shortest signed angle from a to b.
pub fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = (b - a) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// Frame-rate independent exponential approach.
pub fn damp(cur: f32, target: f32, rate: f32, dt: f32) -> f32 {
    cur + (target - cur) * (1.0 - (-rate * dt).exp())
}

pub fn damp3(cur: V3, target: V3, rate: f32, dt: f32) -> V3 {
    cur.lerp(target, 1.0 - (-rate * dt).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_orthonormal() {
        let b = Basis::from_euler(0.7, 0.3, -0.4);
        for v in [b.f, b.u, b.r] {
            assert!((v.len() - 1.0).abs() < 1e-5);
        }
        assert!(b.f.dot(b.u).abs() < 1e-5);
        assert!(b.f.dot(b.r).abs() < 1e-5);
        assert!(b.u.dot(b.r).abs() < 1e-5);
    }

    #[test]
    fn level_basis_points_along_yaw() {
        let b = Basis::from_euler(0.0, 0.0, 0.0);
        assert!((b.f - v3(1.0, 0.0, 0.0)).len() < 1e-6);
        assert!((b.u - V3::UP).len() < 1e-6);
    }

    #[test]
    fn angle_diff_wraps() {
        assert!((angle_diff(3.0, -3.0) - (std::f32::consts::TAU - 6.0)).abs() < 1e-5);
        assert!((angle_diff(0.1, 0.3) - 0.2).abs() < 1e-6);
    }
}
