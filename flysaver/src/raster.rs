//! Perspective projection and line/point rasterisation into braille sub-pixels.

use crate::fb::Frame;
use crate::math::{V3, v3};
use crate::theme::Rgb;

const NEAR: f32 = 0.02;

#[derive(Clone, Copy, Debug)]
pub struct Cam {
    pub pos: V3,
    fwd: V3,
    right: V3,
    up: V3,
    focal: f32,
    cx: f32,
    cy: f32,
    /// Horizontal stretch for non-square sub-pixels.
    aspect: f32,
    pub fog_near: f32,
    pub fog_far: f32,
    /// Distant lines never dim below this.
    pub fog_floor: f32,
}

impl Cam {
    /// `fov_y` in radians; `aspect` = sub-pixel height / width.
    pub fn look_at(pos: V3, target: V3, fov_y: f32, sub_w: usize, sub_h: usize, aspect: f32) -> Cam {
        let fwd = (target - pos).norm();
        let mut right = fwd.cross(V3::UP);
        if right.len() < 1e-4 {
            right = v3(1.0, 0.0, 0.0);
        }
        let right = right.norm();
        let up = right.cross(fwd);
        Cam {
            pos,
            fwd,
            right,
            up,
            focal: sub_h as f32 * 0.5 / (fov_y * 0.5).tan(),
            cx: sub_w as f32 * 0.5,
            cy: sub_h as f32 * 0.5,
            aspect,
            fog_near: 0.5,
            fog_far: 6.0,
            fog_floor: 0.12,
        }
    }

    fn to_view(self, p: V3) -> V3 {
        let d = p - self.pos;
        v3(d.dot(self.right), d.dot(self.up), d.dot(self.fwd))
    }

    fn view_to_screen(&self, v: V3) -> (f32, f32) {
        (self.cx + v.x / v.z * self.focal * self.aspect, self.cy - v.y / v.z * self.focal)
    }

    /// Screen position (sub-pixels) and depth, or None behind the camera.
    pub fn project(&self, p: V3) -> Option<(f32, f32, f32)> {
        let v = self.to_view(p);
        if v.z < NEAR {
            return None;
        }
        let (x, y) = self.view_to_screen(v);
        Some((x, y, v.z))
    }

    /// Sub-pixels per world unit at depth z (for sizing).
    pub fn scale_at(&self, z: f32) -> f32 {
        self.focal / z.max(NEAR)
    }

    pub fn fog(&self, z: f32) -> f32 {
        let t = (z - self.fog_near) / (self.fog_far - self.fog_near);
        (1.0 - t).clamp(self.fog_floor, 1.0)
    }
}

/// Draw a 3D segment with depth cueing.
pub fn line3(f: &mut Frame, cam: &Cam, a: V3, b: V3, i: f32, c: Rgb) {
    let mut va = cam.to_view(a);
    let mut vb = cam.to_view(b);
    if va.z < NEAR && vb.z < NEAR {
        return;
    }
    // Clip against the near plane.
    if va.z < NEAR {
        let t = (NEAR - va.z) / (vb.z - va.z);
        va = va.lerp(vb, t);
    } else if vb.z < NEAR {
        let t = (NEAR - vb.z) / (va.z - vb.z);
        vb = vb.lerp(va, t);
    }
    let (x0, y0) = cam.view_to_screen(va);
    let (x1, y1) = cam.view_to_screen(vb);
    let (ia, ib) = (i * cam.fog(va.z), i * cam.fog(vb.z));
    line2(f, x0, y0, x1, y1, ia, ib, c);
}

/// DDA line in sub-pixel space, clipped to the screen, intensity interpolated.
pub fn line2(f: &mut Frame, x0: f32, y0: f32, x1: f32, y1: f32, i0: f32, i1: f32, c: Rgb) {
    let (w, h) = (f.sub_w() as f32, f.sub_h() as f32);
    let Some((t0, t1)) = clip(x0, y0, x1, y1, -1.0, -1.0, w + 1.0, h + 1.0) else { return };
    let (dx, dy) = (x1 - x0, y1 - y0);
    let (sx, sy) = (x0 + dx * t0, y0 + dy * t0);
    let (ex, ey) = (x0 + dx * t1, y0 + dy * t1);
    let steps = (ex - sx).abs().max((ey - sy).abs()).ceil().max(1.0) as usize;
    for k in 0..=steps {
        let t = k as f32 / steps as f32;
        let tt = t0 + (t1 - t0) * t;
        let inten = i0 + (i1 - i0) * tt;
        f.plot((sx + (ex - sx) * t).round() as i32, (sy + (ey - sy) * t).round() as i32, inten, c);
    }
}

/// Liang-Barsky: parameter range of the segment inside the rectangle.
fn clip(x0: f32, y0: f32, x1: f32, y1: f32, xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Option<(f32, f32)> {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    for (p, q) in [(-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

pub fn point3(f: &mut Frame, cam: &Cam, p: V3, i: f32, c: Rgb) {
    if let Some((x, y, z)) = cam.project(p) {
        f.plot(x.round() as i32, y.round() as i32, i * cam.fog(z), c);
    }
}

/// Polyline through 3D points.
pub fn poly3(f: &mut Frame, cam: &Cam, pts: &[V3], closed: bool, i: f32, c: Rgb) {
    for w in pts.windows(2) {
        line3(f, cam, w[0], w[1], i, c);
    }
    if closed && pts.len() > 2 {
        line3(f, cam, pts[pts.len() - 1], pts[0], i, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_projects_to_centre() {
        let cam = Cam::look_at(v3(0.0, 0.0, -5.0), v3(0.0, 0.0, 0.0), 1.0, 200, 100, 1.0);
        let (x, y, z) = cam.project(v3(0.0, 0.0, 0.0)).unwrap();
        assert!((x - 100.0).abs() < 1e-3 && (y - 50.0).abs() < 1e-3 && (z - 5.0).abs() < 1e-3);
        assert!(cam.project(v3(0.0, 0.0, -6.0)).is_none());
    }

    #[test]
    fn up_is_up_and_right_is_right() {
        let cam = Cam::look_at(v3(0.0, 0.0, -5.0), v3(0.0, 0.0, 0.0), 1.0, 200, 100, 1.0);
        let (_, y, _) = cam.project(v3(0.0, 1.0, 0.0)).unwrap();
        assert!(y < 50.0);
        // Looking down +z with y up, the right-hand side is -x... or +x: check consistency with cross().
        let right = v3(0.0, 0.0, 1.0).cross(V3::UP);
        let (x, _, _) = cam.project(right).unwrap();
        assert!(x > 100.0);
    }

    #[test]
    fn clipped_line_stays_on_screen() {
        let mut f = Frame::new(10, 5);
        line2(&mut f, -1000.0, 10.0, 1000.0, 10.0, 1.0, 1.0, Rgb(0, 255, 0));
        let lit = f.dots.iter().filter(|d| **d > 0.0).count();
        assert_eq!(lit, f.sub_w());
    }
}
