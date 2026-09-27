//! The connectome: every sampled neuron of the BANC 888 fly drawn where it sits,
//! turning slowly like a hologram. Firing is decorative: region pulses tied to
//! what the fly is doing, plus a wave running from the brain down the nerve cord.

use crate::fb::Frame;
use crate::math::{rot_y, v3, V3};
use crate::raster::{line3, Cam};
use crate::rng::{hash01, Rng};
use crate::sim::{Event, Fly, BRAIN_CENTER, BRAIN_SCALE};
use crate::theme::Theme;
use std::f32::consts::TAU;

static ASSET: &[u8] = include_bytes!("../../assets/connectome.bin");

// Region indices in the atlas (see tools/build_connectome.py).
const OPTIC: [u8; 2] = [0, 1];
const CENTRAL: [u8; 2] = [2, 3];
const ANTENNA: u8 = 6;
const EYE: u8 = 7;
const HALTERES: u8 = 9;
const WINGS: u8 = 10;
const LEGS: u8 = 11;
const OTHER_SENSORY: u8 = 12;
const DESCENDING: u8 = 13;
const WING_MOTOR: u8 = 15;
const LEG_MOTOR: u8 = 16;

pub struct Connectome {
    pub pos: Vec<V3>,
    pub region: Vec<u8>,
    pub regions: Vec<String>,
}

pub fn load(max_points: usize) -> Connectome {
    parse(ASSET, max_points).expect("embedded connectome asset is malformed")
}

pub fn parse(b: &[u8], max_points: usize) -> Option<Connectome> {
    if b.get(0..4)? != b"FLYC" {
        return None;
    }
    let u32at = |i: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?)) };
    if u32at(4)? != 1 {
        return None;
    }
    let count = u32at(8)? as usize;
    let nreg = *b.get(12)? as usize;
    let mut i = 13;
    let mut regions = Vec::with_capacity(nreg);
    for _ in 0..nreg {
        let l = *b.get(i)? as usize;
        regions.push(String::from_utf8_lossy(b.get(i + 1..i + 1 + l)?).into_owned());
        i += 1 + l;
    }
    let n = count.min(max_points);
    let mut pos = Vec::with_capacity(n);
    let mut region = Vec::with_capacity(n);
    let q = |k: usize| -> Option<f32> { Some(u16::from_le_bytes(b.get(k..k + 2)?.try_into().ok()?) as f32 / 65535.0 * 2.0 - 1.0) };
    for k in 0..n {
        let o = i + k * 7;
        // Atlas frame is frontal with y up the body axis; the room is y-up too.
        pos.push(v3(q(o)?, q(o + 2)?, q(o + 4)?));
        region.push(*b.get(o + 6)?);
    }
    Some(Connectome { pos, region, regions })
}

pub struct Brain {
    net: Connectome,
    act: Vec<f32>,
    wave: f32,
    angle: f32,
    bucket: u32,
    bucket_t: f32,
}

impl Brain {
    pub fn new(points: usize) -> Brain {
        let net = load(points);
        let n = net.regions.len().max(18);
        Brain { net, act: vec![0.0; n], wave: 2.0, angle: 0.0, bucket: 0, bucket_t: 0.0 }
    }

    pub fn len(&self) -> usize {
        self.net.pos.len()
    }

    fn kick(&mut self, r: u8, v: f32) {
        if let Some(a) = self.act.get_mut(r as usize) {
            *a = a.max(v);
        }
    }

    pub fn step(&mut self, dt: f32, fly: &Fly, rng: &mut Rng) {
        self.angle = (self.angle + dt * 0.22) % TAU;
        for a in &mut self.act {
            *a *= (-dt * 1.8).exp();
        }
        for e in &fly.events {
            match e {
                Event::Saccade => {
                    OPTIC.iter().for_each(|r| self.kick(*r, 0.9));
                    self.kick(EYE, 0.8);
                    self.kick(DESCENDING, 0.7);
                }
                Event::TakeOff => {
                    self.kick(WING_MOTOR, 1.0);
                    self.kick(DESCENDING, 1.0);
                    self.kick(LEG_MOTOR, 0.8);
                }
                Event::Land => {
                    self.kick(LEGS, 0.9);
                    self.kick(LEG_MOTOR, 0.8);
                }
            }
        }
        if fly.airborne() {
            self.kick(WING_MOTOR, 0.35);
            self.kick(HALTERES, 0.4);
            self.kick(WINGS, 0.3);
        }
        if fly.feeding {
            self.kick(ANTENNA, 0.6);
            self.kick(OTHER_SENSORY, 0.5);
            CENTRAL.iter().for_each(|r| self.kick(*r, 0.45));
        }
        if fly.grooming > 0.5 {
            self.kick(LEG_MOTOR, 0.6);
            self.kick(LEGS, 0.5);
        }
        // Spontaneous activity somewhere, now and then.
        if rng.chance(dt * 0.8) {
            let r = rng.below(self.act.len()) as u8;
            self.kick(r, rng.range(0.3, 0.7));
        }
        // A descending wave, head to nerve cord, every few seconds.
        self.wave -= dt * 0.9;
        if self.wave < -1.6 && rng.chance(dt * 0.5) {
            self.wave = 1.3;
        }
        // Resample who fires ~12 times a second so sparks flicker instead of strobing.
        self.bucket_t += dt;
        if self.bucket_t > 0.083 {
            self.bucket_t = 0.0;
            self.bucket = self.bucket.wrapping_add(1);
        }
    }

    pub fn draw(&self, f: &mut Frame, cam: &Cam, theme: &Theme) {
        let s = BRAIN_SCALE;
        let a = self.angle;
        let bucket = self.bucket.wrapping_mul(7919);
        // Keep the cloud airy: at rest, light only as many neurons as fill ~35%
        // of its silhouette at this distance. Firing neurons always show.
        let keep = match cam.project(BRAIN_CENTER) {
            Some((_, _, z)) => {
                let r = cam.scale_at(z) * s;
                (0.35 * 1.3 * r * r / self.net.pos.len() as f32).min(1.0)
            }
            None => 1.0,
        };
        for (k, p) in self.net.pos.iter().enumerate() {
            let w = BRAIN_CENTER + rot_y(*p * s, a);
            let Some((x, y, z)) = cam.project(w) else { continue };
            let h = hash01(k as u32);
            let r = self.net.region[k] as usize;
            let act = self.act.get(r).copied().unwrap_or(0.0);
            let wave = (-((p.y - self.wave).powi(2)) / 0.004).exp() * if h < 0.5 { 0.8 } else { 0.0 };
            let spark = if hash01(k as u32 ^ bucket) < act * 0.35 { act } else { 0.0 };
            let fire = spark.max(wave).min(1.0);
            // Most neurons sit in the optic lobes and central brain; thin those
            // harder so the nerve cord still reads.
            let keep_here = if r <= 3 { keep * 0.45 } else { (keep * 2.5).min(1.0) };
            if fire < 0.05 && h >= keep_here {
                continue;
            }
            let base = 0.2 + 0.25 * (h / keep_here.max(1e-3)).min(1.0);
            let i = (base + 0.8 * fire) * cam.fog(z).max(0.4);
            let c = theme.wire.mix(theme.fire, fire);
            f.plot(x.round() as i32, y.round() as i32, i, c);
        }
        // The projector: a faint ring below the hologram.
        let base = BRAIN_CENTER - v3(0.0, s * 1.15, 0.0);
        let ring: Vec<V3> = (0..=24).map(|j| {
            let t = j as f32 / 24.0 * TAU;
            base + v3(t.cos() * 0.28, 0.0, t.sin() * 0.28)
        }).collect();
        for w in ring.windows(2) {
            line3(f, cam, w[0], w[1], 0.3, theme.wire);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_parses_with_all_regions() {
        let c = load(usize::MAX);
        assert!(c.pos.len() >= 20_000);
        assert_eq!(c.regions.len(), 18);
        assert_eq!(c.regions[0], "optic lobe, left");
        assert!(c.region.iter().all(|r| (*r as usize) < c.regions.len()));
        assert!(c.pos.iter().all(|p| p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && p.z.abs() <= 1.0));
    }

    #[test]
    fn prefix_subsample_keeps_every_big_region() {
        let c = load(8000);
        for r in [0u8, 1, 2, 3, 4, 5, 11] {
            assert!(c.region.iter().any(|x| *x == r), "region {r} missing");
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(b"nope", 10).is_none());
    }
}
