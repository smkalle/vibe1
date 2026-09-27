//! The connectome: every sampled neuron of the BANC 888 fly drawn where it sits,
//! turning slowly like a hologram. In live mode (the default) the 60,000-neuron
//! sub-net runs the Cadence rate model (crate::neuro), fed by the fly's senses
//! (crate::senses), and every active neuron is drawn lit by its real activity over
//! a dim whole-brain silhouette. In decorative mode the firing is region pulses
//! tied to what the fly does, plus a wave down the nerve cord.

use crate::fb::Frame;
use crate::math::{rot_y, v3, V3};
use crate::learner::{self, Learner};
use crate::memory::Memory;
use crate::neuro;
use crate::senses::{self, Motion};
use crate::raster::{line3, Cam};
use crate::rng::{hash01, Rng};
use crate::sim::{Command, Event, Fly, BRAIN_CENTER, BRAIN_SCALE, SWATTER_R};
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

/// Steps run before the first frame so the net starts settled under the senses
/// (about 8 time constants of the rate model).
const WARM_UP: usize = 40;
/// Activity below this is not drawn.
const SHOW_LEVEL: f64 = 0.02;

// The brain layer's readouts and constants, as the original's web/life.js declares them.
const DNA02_L: &str = "dn:DNa02:left";
const DNA02_R: &str = "dn:DNa02:right";
const DNP09: &str = "dn:DNp09";
const LANDING: &str = "dn:landing";
const GIANT_FIBRE: &str = "gf";
const MN9: &str = "mn9";
const GROOMING: &str = "dn:grooming";
const MBON_APPROACH: &str = "mbon:MBON11:right";
const MBON_AVOID: &str = "mbon:MBON05:left";
const READOUTS: [&str; 9] = [DNA02_L, DNA02_R, DNP09, LANDING, GIANT_FIBRE, MN9, GROOMING, MBON_APPROACH, MBON_AVOID];
const K_TURN: f64 = 6.0;
const TURN_DEADBAND: f64 = 0.02;
const K_SPEED: f64 = 0.6;
const LAND_LEVEL: f64 = 0.15;
const ESCAPE_LEVEL: f64 = 0.5;
/// MN9 above this starts feeding. Without sugar MN9 is 0 (at most 1e-5); tasting sugar, it
/// peaks 0.02-0.3 about 1.3 s after landing (0.3-0.48 only on a brain's very first
/// taste), and bitter suppresses that peak. The original's 0.2 is reached only on that
/// first taste, so the threshold is set against the zero baseline (specs/flysaver-taste.md).
const FEED_LEVEL: f64 = 0.005;
const GROOM_LEVEL: f64 = 0.15;
/// The page's softmax temperature for the mushroom body's choice.
const MB_TEMPERATURE: f64 = 0.3;

/// The live rate model and what it needs between frames.
pub struct Live {
    pub net: neuro::Brain,
    motion: Motion,
    steps_per_frame: usize,
    warm: bool,
    /// Smoothed wall-clock cost of one model step, in ms.
    pub ms_per_step: f32,
    /// Neurons at or above half activation (the original's "active" count).
    pub active: usize,
    /// Readouts (means of READOUTS) now, and at level-flight rest after warm-up.
    pub read: [f64; 9],
    baseline: Option<[f64; 9]>,
    /// The mushroom body's actor-critic (Phase 3); None keeps the measured seam fixed.
    pub learner: Option<Learner>,
}

impl Live {
    fn dev(&self, k: usize) -> f64 {
        self.read[k] - self.baseline.map_or(0.0, |b| b[k])
    }

    /// What the output neurons ask the body to do (life.js "the brain layer").
    pub fn command(&self) -> Command {
        let asym = self.dev(1) - self.dev(0);
        Command {
            turn_rate: if asym.abs() > TURN_DEADBAND { (K_TURN * asym) as f32 } else { 0.0 },
            speed_factor: (1.0 + K_SPEED * self.dev(2)).clamp(0.3, 1.5) as f32,
            land: self.dev(3) > LAND_LEVEL,
            escape: self.read[4] > ESCAPE_LEVEL,
            feed: self.read[5] > FEED_LEVEL,
            groom: self.dev(6) > GROOM_LEVEL,
            mn9: self.read[5] as f32,
        }
    }

    /// The mushroom body's probability of approaching: a softmax over MBON11
    /// (approach, GABAergic) and MBON05 (avoid, glutamatergic), Aso et al. 2014.
    pub fn p_approach(&self) -> f64 {
        let (a, v) = (self.read[7] / MB_TEMPERATURE, self.read[8] / MB_TEMPERATURE);
        let m = a.max(v);
        let (ea, ev) = ((a - m).exp(), (v - m).exp());
        ea / (ea + ev)
    }

    /// The mushroom body decides approach (true) or avoid with uniform draw `u`. With a
    /// learner this is its act (and the decision's eligibility is kept for the outcome).
    pub fn decide(&mut self, u: f64) -> (bool, f64) {
        match &mut self.learner {
            Some(l) => {
                let d = l.act(&self.net, u);
                (d.choice == 0, d.p_approach)
            }
            None => {
                let p = self.p_approach();
                (u < p, p)
            }
        }
    }

    /// The outcome of the last decision: the learner's lesson, if it is learning.
    pub fn outcome(&mut self, reward: f64) -> Option<learner::Lesson> {
        let l = self.learner.as_mut()?;
        l.learn(&mut self.net, reward, true)
    }

    pub fn remember(&mut self, m: &Memory) {
        if let Some(l) = &mut self.learner {
            l.efficacy = m.efficacy.clone();
            l.w_critic = m.w_critic.clone();
            l.b_critic = m.b_critic;
            l.apply(&mut self.net);
        }
    }

    pub fn giant_fibre(&self) -> f64 {
        self.read[4]
    }

    pub fn turn_asym(&self) -> f64 {
        self.dev(1) - self.dev(0)
    }
}

pub struct Brain {
    net: Connectome,
    act: Vec<f32>,
    wave: f32,
    angle: f32,
    bucket: u32,
    bucket_t: f32,
    pub live: Option<Live>,
}

impl Brain {
    pub fn new(points: usize, live: bool, steps_per_frame: usize, learning: bool) -> Brain {
        let net = load(points);
        let n = net.regions.len().max(18);
        let live = live.then(|| {
            let brain = neuro::Brain::load();
            let learner = learning.then(|| Learner::new(&brain));
            Live {
            net: brain,
            learner,
            motion: Motion::default(),
            steps_per_frame: steps_per_frame.max(1),
            warm: false,
            ms_per_step: 0.0,
            active: 0,
            read: [0.0; 9],
            baseline: None,
        }});
        Brain { net, act: vec![0.0; n], wave: 2.0, angle: 0.0, bucket: 0, bucket_t: 0.0, live }
    }

    pub fn len(&self) -> usize {
        self.net.pos.len()
    }

    fn step_live(live: &mut Live, dt: f32, fly: &Fly, threat: Option<crate::math::V3>) {
        let rates = live.motion.rates(fly, dt);
        let loom = live.motion.looming(fly, threat, SWATTER_R, dt);
        live.net.clear_stimuli();
        for (name, level) in senses::sense(fly, rates, loom) {
            live.net.stimulate(name, level);
        }
        let steps = if live.warm { live.steps_per_frame } else { WARM_UP };
        live.warm = true;
        let t = std::time::Instant::now();
        for _ in 0..steps {
            live.net.step();
        }
        let ms = t.elapsed().as_secs_f32() * 1e3 / steps as f32;
        live.ms_per_step = if live.ms_per_step == 0.0 { ms } else { live.ms_per_step * 0.9 + ms * 0.1 };
        live.active = live.net.active_count(0.5);
        for (k, name) in READOUTS.iter().enumerate() {
            live.read[k] = live.net.mean(name);
        }
        if live.baseline.is_none() {
            live.baseline = Some(live.read); // the level-flight rest, right after warm-up
        }
    }

    fn kick(&mut self, r: u8, v: f32) {
        if let Some(a) = self.act.get_mut(r as usize) {
            *a = a.max(v);
        }
    }

    pub fn step(&mut self, dt: f32, fly: &Fly, threat: Option<crate::math::V3>, rng: &mut Rng) {
        self.angle = (self.angle + dt * 0.22) % TAU;
        if let Some(live) = &mut self.live {
            Brain::step_live(live, dt, fly, threat);
            return;
        }
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
                Event::Escape => {
                    self.kick(DESCENDING, 1.0);
                    self.kick(WING_MOTOR, 1.0);
                    self.kick(LEG_MOTOR, 1.0);
                }
                Event::Hit => {
                    for r in 0..18 {
                        self.kick(r, 0.8);
                    }
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
            // Live mode: the silhouette stays dark; the model's neurons light up on top.
            let fire = if self.live.is_some() { 0.0 } else { spark.max(wave).min(1.0) };
            // Most neurons sit in the optic lobes and central brain; thin those
            // harder so the nerve cord still reads.
            let keep_here = if r <= 3 { keep * 0.45 } else { (keep * 2.5).min(1.0) };
            if fire < 0.05 && h >= keep_here {
                continue;
            }
            let base = if self.live.is_some() { 0.1 + 0.12 * h } else { 0.2 + 0.25 * (h / keep_here.max(1e-3)).min(1.0) };
            let i = (base + 0.8 * fire) * cam.fog(z).max(0.4);
            let c = theme.wire.mix(theme.fire, fire);
            f.plot(x.round() as i32, y.round() as i32, i, c);
        }
        if let Some(live) = &self.live {
            for (k, act) in live.net.s.iter().enumerate() {
                if *act < SHOW_LEVEL {
                    continue;
                }
                let w = BRAIN_CENTER + rot_y(live.net.pos[k] * s, a);
                let Some((x, y, z)) = cam.project(w) else { continue };
                let act = *act as f32;
                let i = (0.45 + 0.55 * act) * cam.fog(z).max(0.6);
                f.plot(x.round() as i32, y.round() as i32, i, theme.heat(act));
            }
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
