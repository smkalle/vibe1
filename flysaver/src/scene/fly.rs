//! The fly as a wireframe: head with big compound eyes, thorax, striped abdomen,
//! six jointed legs, antennae, proboscis, and wings drawn as a stroke blur in
//! flight or folded over the back at rest.

use crate::fb::Frame;
use crate::math::{v3, Basis, V3};
use crate::raster::{line3, poly3, Cam};
use crate::sim::{Fly, FLY_LEN};
use crate::theme::{Rgb, Theme};
use std::f32::consts::{PI, TAU};

struct Pose {
    at: V3,
    b: Basis,
    s: f32,
}

impl Pose {
    /// Local body coordinates (x forward, y up, z right; body length 1) to world.
    fn w(&self, p: V3) -> V3 {
        self.at + self.b.apply(p * self.s)
    }
}

pub fn draw(f: &mut Frame, cam: &Cam, theme: &Theme, fly: &Fly) {
    let pose = Pose { at: fly.pos, b: Basis::from_euler(fly.yaw, fly.pitch, fly.roll), s: FLY_LEN };
    let body = theme.fly;
    let eye = theme.fire;

    // Far away the wireframe collapses to a smear; give it a glowing core instead.
    let px = cam.project(fly.pos).map(|(_, _, z)| cam.scale_at(z) * FLY_LEN).unwrap_or(0.0);
    if px < 16.0 {
        if let Some((x, y, _)) = cam.project(fly.pos) {
            let (x, y) = (x.round() as i32, y.round() as i32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                f.plot(x + dx, y + dy, 1.0, body);
            }
            if fly.airborne() {
                let on = fly.wing_phase < 0.5;
                f.plot(x - 1, y - 1 + on as i32, 0.6, body);
                f.plot(x + 2, y - 1 + on as i32, 0.6, body);
            }
        }
        if px < 6.0 {
            return;
        }
    }

    let detail = if px > 60.0 { 2 } else { 1 };
    ellipsoid(f, cam, &pose, v3(0.15, 0.02, 0.0), v3(0.19, 0.15, 0.15), detail, 0.9, body); // thorax
    ellipsoid(f, cam, &pose, v3(-0.22, -0.01, 0.0), v3(0.27, 0.13, 0.14), detail, 0.8, body); // abdomen
    for k in 0..4 {
        let x = -0.1 - 0.09 * k as f32;
        let r = (1.0 - ((x + 0.22) / 0.27).powi(2)).max(0.0).sqrt();
        ring(f, cam, &pose, v3(x, -0.01, 0.0), 0.13 * r, 0.14 * r, 0.6, body);
    }
    ellipsoid(f, cam, &pose, v3(0.4, 0.04, 0.0), v3(0.08, 0.1, 0.12), detail, 0.9, body); // head
    for side in [-1.0f32, 1.0] {
        ellipsoid(f, cam, &pose, v3(0.41, 0.05, side * 0.09), v3(0.07, 0.09, 0.05), detail, 1.0, eye); // compound eyes
        // Antenna with arista.
        let base = v3(0.47, 0.08, side * 0.03);
        let tip = v3(0.53, 0.05, side * 0.05);
        line3(f, cam, pose.w(base), pose.w(tip), 0.8, body);
        line3(f, cam, pose.w(tip), pose.w(v3(0.56, 0.12, side * 0.08)), 0.5, body);
    }

    if fly.feeding {
        let ext = 0.6 + 0.4 * (fly.time * 5.0).sin().abs();
        line3(f, cam, pose.w(v3(0.42, -0.04, 0.0)), pose.w(v3(0.46, -0.04 - 0.2 * ext, 0.0)), 1.0, eye);
    }

    legs(f, cam, &pose, fly, body);
    wings(f, cam, &pose, fly, body);
}

/// Wire ellipsoid: latitude rings along the body axis plus two meridians.
fn ellipsoid(f: &mut Frame, cam: &Cam, pose: &Pose, c: V3, r: V3, detail: usize, i: f32, col: Rgb) {
    let n = 12 * detail;
    let rings = 2 + detail;
    for k in 1..=rings {
        let u = -1.0 + 2.0 * k as f32 / (rings + 1) as f32;
        let s = (1.0 - u * u).sqrt();
        ring(f, cam, pose, c + v3(u * r.x, 0.0, 0.0), r.y * s, r.z * s, i * 0.8, col);
    }
    let meridian = |plane: usize| -> Vec<V3> {
        (0..=n).map(|j| {
            let t = j as f32 / n as f32 * TAU;
            let p = if plane == 0 { v3(t.cos() * r.x, t.sin() * r.y, 0.0) } else { v3(t.cos() * r.x, 0.0, t.sin() * r.z) };
            pose.w(c + p)
        }).collect()
    };
    poly3(f, cam, &meridian(0), false, i, col);
    poly3(f, cam, &meridian(1), false, i, col);
}

/// Ring around the body axis at local centre c.
fn ring(f: &mut Frame, cam: &Cam, pose: &Pose, c: V3, ry: f32, rz: f32, i: f32, col: Rgb) {
    let pts: Vec<V3> = (0..12).map(|j| {
        let t = j as f32 / 12.0 * TAU;
        pose.w(c + v3(0.0, t.cos() * ry, t.sin() * rz))
    }).collect();
    poly3(f, cam, &pts, true, i, col);
}

fn legs(f: &mut Frame, cam: &Cam, pose: &Pose, fly: &Fly, col: Rgb) {
    let t = fly.time;
    for (pair, x) in [0.26f32, 0.15, 0.04].iter().enumerate() {
        for side in [-1.0f32, 1.0] {
            let hip = v3(*x, -0.08, side * 0.06);
            let splay = [0.5f32, 0.0, -0.55][pair];
            let (knee, ankle, foot) = if fly.airborne() {
                // Tucked under the body in flight.
                (
                    hip + v3(0.05, -0.08, side * 0.1),
                    hip + v3(-0.08, -0.12, side * 0.08),
                    hip + v3(-0.14, -0.1, side * 0.05),
                )
            } else if pair == 0 && fly.grooming > 0.5 {
                // Forelegs up, rubbing each other in front of the head.
                let rub = (t * 14.0).sin() * 0.03 * side;
                (
                    hip + v3(0.12, 0.02, side * 0.12),
                    v3(0.5, -0.05 + rub, side * 0.04),
                    v3(0.47, -0.1 - rub, side * 0.01),
                )
            } else {
                let fidget = if pair == 2 && fly.grooming > 0.5 { (t * 9.0).sin() * 0.04 } else { 0.0 };
                (
                    hip + v3(splay * 0.12, 0.06, side * 0.2),
                    hip + v3(splay * 0.22 + fidget, -0.1, side * 0.32),
                    hip + v3(splay * 0.26 + fidget, -0.2, side * 0.36),
                )
            };
            poly3(f, cam, &[pose.w(hip), pose.w(knee), pose.w(ankle), pose.w(foot)], false, 0.75, col);
        }
    }
}

fn wing_outline(pose: &Pose, hinge: V3, dir: V3, side_axis: V3) -> Vec<V3> {
    let (len, wid) = (0.72, 0.2);
    (0..=16).map(|j| {
        let t = j as f32 / 16.0 * TAU;
        let along = (1.0 - t.cos()) * 0.5 * len;
        let across = t.sin() * wid * (0.55 + 0.45 * (along / len));
        pose.w(hinge + dir * along + side_axis * across)
    }).collect()
}

fn wings(f: &mut Frame, cam: &Cam, pose: &Pose, fly: &Fly, col: Rgb) {
    for side in [-1.0f32, 1.0] {
        let hinge = v3(0.2, 0.12, side * 0.06);
        if fly.airborne() {
            // Stroke plane roughly horizontal; draw the sweep as a blur of ghosts.
            let ghosts = 5;
            for g in 0..ghosts {
                let phase = (g as f32 / ghosts as f32 + fly.wing_phase) % 1.0;
                let sweep = -0.9 + 1.9 * (0.5 - 0.5 * (phase * TAU).cos());
                let dir = v3(sweep.sin(), 0.15, side * sweep.cos()).norm();
                let across = v3(-sweep.cos(), 0.0, side * sweep.sin()).norm();
                let i = if g == 0 { 0.55 } else { 0.22 };
                poly3(f, cam, &wing_outline(pose, hinge, dir, across), false, i, col);
            }
        } else {
            // Folded back over the abdomen, slightly splayed.
            let a = PI - 0.22;
            let dir = v3(a.cos(), 0.05, side * a.sin()).norm();
            let across = v3(0.0, 0.2, side).norm();
            poly3(f, cam, &wing_outline(pose, hinge, dir, across * 0.9), false, 0.5, col);
        }
    }
}
