//! The wireframe room: floor grid, walls, the table, a banana and a piece of bread.

use crate::fb::Frame;
use crate::math::{v3, V3};
use crate::raster::{line3, point3, poly3, Cam};
use crate::rng::hash01;
use crate::sim::{BANANA, BREAD, ROOM, TABLE_CENTER, TABLE_SIZE};
use crate::theme::Theme;
use std::f32::consts::{PI, TAU};

pub fn draw(f: &mut Frame, cam: &Cam, theme: &Theme, t: f32) {
    let c = theme.wire;
    let (w, h, d) = (ROOM.x, ROOM.y, ROOM.z);

    // Floor grid every half metre, back walls every metre.
    let mut g = 0.0;
    while g <= w + 1e-3 {
        line3(f, cam, v3(g, 0.0, 0.0), v3(g, 0.0, d), 0.32, c);
        line3(f, cam, v3(g, 0.0, 0.0), v3(g, h, 0.0), 0.2, c);
        g += 0.5;
    }
    let mut g = 0.0;
    while g <= d + 1e-3 {
        line3(f, cam, v3(0.0, 0.0, g), v3(w, 0.0, g), 0.32, c);
        line3(f, cam, v3(0.0, 0.0, g), v3(0.0, h, g), 0.2, c);
        g += 0.5;
    }
    let mut y = 0.5;
    while y < h {
        line3(f, cam, v3(0.0, y, 0.0), v3(w, y, 0.0), 0.2, c);
        line3(f, cam, v3(0.0, y, 0.0), v3(0.0, y, d), 0.2, c);
        y += 0.5;
    }
    // Room edges.
    for (a, b) in [
        (v3(0.0, h, 0.0), v3(w, h, 0.0)),
        (v3(0.0, h, 0.0), v3(0.0, h, d)),
        (v3(w, 0.0, 0.0), v3(w, h, 0.0)),
        (v3(0.0, 0.0, d), v3(0.0, h, d)),
        (v3(w, 0.0, d), v3(w, h, d)),
        (v3(w, h, 0.0), v3(w, h, d)),
        (v3(0.0, h, d), v3(w, h, d)),
    ] {
        line3(f, cam, a, b, 0.45, c);
    }

    table(f, cam, theme);
    banana(f, cam, theme);
    bread(f, cam, theme);

    // Dust motes drifting in the light.
    for i in 0..60u32 {
        let (a, b, e) = (hash01(i), hash01(i + 1000), hash01(i + 2000));
        let p = v3(
            (a * w + (t * 0.02 + a * TAU).sin() * 0.1).rem_euclid(w),
            (b * h + t * 0.01 * (e - 0.5)).rem_euclid(h),
            (e * d + (t * 0.015 + b * TAU).cos() * 0.1).rem_euclid(d),
        );
        let tw = 0.3 + 0.3 * (t * (1.0 + a * 2.0) + e * TAU).sin();
        point3(f, cam, p, tw, theme.rain_head);
    }
}

fn boxe(f: &mut Frame, cam: &Cam, lo: V3, hi: V3, i: f32, c: crate::theme::Rgb) {
    let p = |x: bool, y: bool, z: bool| v3(if x { hi.x } else { lo.x }, if y { hi.y } else { lo.y }, if z { hi.z } else { lo.z });
    for &(a, b) in &[
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)), ((0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
    ] {
        line3(f, cam, p(a.0 == 1, a.1 == 1, a.2 == 1), p(b.0 == 1, b.1 == 1, b.2 == 1), i, c);
    }
}

fn table(f: &mut Frame, cam: &Cam, theme: &Theme) {
    let c = theme.wire;
    let (cx, cz) = (TABLE_CENTER.x, TABLE_CENTER.z);
    let (hw, hd, top) = (TABLE_SIZE.x * 0.5, TABLE_SIZE.z * 0.5, TABLE_SIZE.y);
    boxe(f, cam, v3(cx - hw, top - 0.04, cz - hd), v3(cx + hw, top, cz + hd), 0.7, c);
    for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let x = cx + sx * (hw - 0.06);
        let z = cz + sz * (hd - 0.06);
        boxe(f, cam, v3(x - 0.025, 0.0, z - 0.025), v3(x + 0.025, top - 0.04, z + 0.025), 0.55, c);
    }
}

fn banana(f: &mut Frame, cam: &Cam, theme: &Theme) {
    // A curved tube: rings along an arc, joined by four seams.
    let col = theme.fire;
    let n = 12;
    let mut rings: Vec<Vec<V3>> = Vec::new();
    for k in 0..=n {
        let s = k as f32 / n as f32;
        let a = (s - 0.5) * 1.6;
        let centre = BANANA + v3(a.sin() * 0.14, 0.035 + (1.0 - a.cos()) * 0.12, (1.0 - a.cos()) * -0.03);
        let r = 0.03 * (1.0 - (2.0 * s - 1.0).powi(4)).max(0.15);
        let tangent = v3(a.cos(), a.sin(), 0.0);
        let side = v3(0.0, 0.0, 1.0);
        let up = side.cross(tangent);
        let ring: Vec<V3> = (0..6).map(|j| {
            let th = j as f32 / 6.0 * TAU;
            centre + up * (th.cos() * r) + side * (th.sin() * r)
        }).collect();
        rings.push(ring);
    }
    for (k, ring) in rings.iter().enumerate() {
        if k % 3 == 0 {
            poly3(f, cam, ring, true, 0.55, col);
        }
    }
    for j in 0..6 {
        let seam: Vec<V3> = rings.iter().map(|r| r[j]).collect();
        poly3(f, cam, &seam, false, 0.7, col);
    }
}

fn bread(f: &mut Frame, cam: &Cam, theme: &Theme) {
    // A loaf: a box body with a domed crust made of arcs.
    let c = theme.rain_head;
    let (hx, hz, h) = (0.11, 0.07, 0.05);
    boxe(f, cam, BREAD + v3(-hx, 0.0, -hz), BREAD + v3(hx, h, hz), 0.5, c);
    for k in 0..=4 {
        let x = -hx + 2.0 * hx * k as f32 / 4.0;
        let arc: Vec<V3> = (0..=8).map(|j| {
            let a = j as f32 / 8.0 * PI;
            BREAD + v3(x, h + a.sin() * 0.035, -a.cos() * hz)
        }).collect();
        poly3(f, cam, &arc, false, 0.55, c);
    }
    let ridge: Vec<V3> = (0..=8).map(|j| BREAD + v3(-hx + 2.0 * hx * j as f32 / 8.0, h + 0.035, 0.0)).collect();
    poly3(f, cam, &ridge, false, 0.55, c);
}

/// Sugar on a fruit: a few grains twinkling just above it.
pub fn sugar(f: &mut Frame, cam: &Cam, theme: &Theme, at: V3, t: f32) {
    for i in 0..14u32 {
        let (a, b, e) = (hash01(i + 5000), hash01(i + 6000), hash01(i + 7000));
        let p = at + v3((a - 0.5) * 0.08, 0.01 + b * 0.03, (e - 0.5) * 0.06);
        let tw = 0.5 + 0.5 * (t * (3.0 + 4.0 * a) + e * TAU).sin();
        point3(f, cam, p, 0.4 + 0.6 * tw, theme.rain_head);
    }
}
