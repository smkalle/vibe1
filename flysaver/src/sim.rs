//! The fly's life and the camera director. Procedural, no neural simulation:
//! rates follow the original page's ethogram (flight bouts of ~5-15 s, about
//! 0.45 saccades per second of flight, sits of 3-8 s, grooming, feeding).

use crate::config::CameraMode;
use crate::math::{angle_diff, damp, damp3, smoothstep, v3, V3};
use crate::rng::Rng;
use std::f32::consts::{PI, TAU};

// The room, in metres. x across, y up, z deep.
pub const ROOM: V3 = v3(3.0, 2.2, 3.0);
pub const TABLE_CENTER: V3 = v3(1.55, 0.0, 1.6);
pub const TABLE_SIZE: V3 = v3(1.3, 0.75, 0.8);
pub const BANANA: V3 = v3(1.25, 0.75, 1.52);
pub const BREAD: V3 = v3(1.9, 0.75, 1.72);
pub const BRAIN_CENTER: V3 = v3(0.62, 1.25, 2.35);
pub const BRAIN_SCALE: f32 = 0.5;
/// Drawn body length; a real fly is 2.5 mm, this one is a hologram-sized 5 cm.
pub const FLY_LEN: f32 = 0.05;

const FLIGHT_MIN: V3 = v3(0.35, 0.35, 0.35);
const FLIGHT_MAX: V3 = v3(2.65, 1.95, 2.65);
const SACCADE_RATE: f32 = 0.45;
const SACCADE_TIME: f32 = 0.16;
const CRUISE: f32 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Flying,
    Approach,
    Landing,
    Sitting,
    TakeOff,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spot {
    Table(V3),
    Banana,
    Bread,
}

impl Spot {
    pub fn pos(self) -> V3 {
        match self {
            Spot::Table(p) => p,
            Spot::Banana => BANANA + v3(0.0, 0.035, 0.0),
            Spot::Bread => BREAD + v3(0.0, 0.07, 0.0),
        }
    }
    pub fn is_food(self) -> bool {
        !matches!(self, Spot::Table(_))
    }
    pub fn name(self) -> &'static str {
        match self {
            Spot::Table(_) => "table",
            Spot::Banana => "banana",
            Spot::Bread => "bread",
        }
    }
}

/// Things the brain cloud lights up for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    Saccade,
    TakeOff,
    Land,
}

#[derive(Clone)]
pub struct Fly {
    pub pos: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub speed: f32,
    pub vy: f32,
    pub mode: Mode,
    pub mode_t: f32,
    mode_len: f32,
    alt_target: f32,
    saccade: Option<(f32, f32, f32)>, // from, to, elapsed
    burst: f32,
    pub spot: Option<Spot>,
    pub feeding: bool,
    pub grooming: f32,
    groom_t: f32,
    pub wing_phase: f32,
    pub saccade_log: Vec<f32>,
    pub hunger: f32,
    pub events: Vec<Event>,
    pub time: f32,
}

impl Fly {
    pub fn new(rng: &mut Rng) -> Fly {
        Fly {
            pos: v3(rng.range(0.8, 2.2), rng.range(0.9, 1.6), rng.range(0.8, 2.2)),
            yaw: rng.range(0.0, TAU),
            pitch: 0.0,
            roll: 0.0,
            speed: CRUISE,
            vy: 0.0,
            mode: Mode::Flying,
            mode_t: 0.0,
            mode_len: rng.range(5.0, 15.0),
            alt_target: rng.range(0.8, 1.8),
            saccade: None,
            burst: 0.0,
            spot: None,
            feeding: false,
            grooming: 0.0,
            groom_t: 0.0,
            wing_phase: 0.0,
            saccade_log: Vec::new(),
            hunger: rng.range(0.4, 0.8),
            events: Vec::new(),
            time: 0.0,
        }
    }

    pub fn airborne(&self) -> bool {
        !matches!(self.mode, Mode::Sitting)
    }

    pub fn heading(&self) -> V3 {
        v3(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    /// Wingbeat shown on the HUD: the thorax resonance, ~200 Hz, in flight.
    pub fn wingbeat_hz(&self) -> f32 {
        if self.airborne() { 196.0 + 14.0 * (self.speed / 0.6).min(1.0) } else { 0.0 }
    }

    pub fn status(&self) -> &'static str {
        match self.mode {
            Mode::Flying => "flying",
            Mode::Approach => "smelling",
            Mode::Landing => "landing",
            Mode::TakeOff => "take-off",
            Mode::Sitting if self.feeding => "feeding",
            Mode::Sitting if self.grooming > 0.5 => "grooming",
            Mode::Sitting => "sitting",
        }
    }

    fn set_mode(&mut self, m: Mode, len: f32) {
        self.mode = m;
        self.mode_t = 0.0;
        self.mode_len = len;
    }

    pub fn step(&mut self, dt: f32, rng: &mut Rng) {
        self.events.clear();
        self.time += dt;
        self.mode_t += dt;
        if self.airborne() {
            self.wing_phase = (self.wing_phase + dt * 11.0) % 1.0; // visual flicker, not 200 Hz
        }
        self.hunger = (self.hunger + dt * if self.feeding { -0.05 } else { 0.004 }).clamp(0.0, 1.0);
        match self.mode {
            Mode::Flying => self.fly(dt, rng),
            Mode::Approach => self.approach(dt, rng),
            Mode::Landing => self.land(dt, rng),
            Mode::Sitting => self.sit(dt, rng),
            Mode::TakeOff => self.take_off(dt, rng),
        }
        // Clamp to the room whatever happens.
        self.pos = v3(
            self.pos.x.clamp(0.05, ROOM.x - 0.05),
            self.pos.y.clamp(0.02, ROOM.y - 0.05),
            self.pos.z.clamp(0.05, ROOM.z - 0.05),
        );
        let now = self.time;
        self.saccade_log.retain(|t| now - *t < 20.0);
    }

    fn start_saccade(&mut self, to: f32) {
        self.saccade = Some((self.yaw, to, 0.0));
        self.saccade_log.push(self.time);
        self.events.push(Event::Saccade);
    }

    /// Advance an ongoing saccade; returns true while turning.
    fn turn(&mut self, dt: f32) -> bool {
        let Some((from, to, t)) = self.saccade else { return false };
        let t = t + dt;
        let k = smoothstep(t / SACCADE_TIME);
        let d = angle_diff(from, to);
        self.yaw = from + d * k;
        self.roll = damp(self.roll, -d.signum() * 0.6 * (1.0 - (2.0 * k - 1.0).abs()), 30.0, dt);
        self.saccade = if t >= SACCADE_TIME { None } else { Some((from, to, t)) };
        true
    }

    fn fly(&mut self, dt: f32, rng: &mut Rng) {
        let turning = self.turn(dt);
        if !turning {
            self.roll = damp(self.roll, 0.0, 6.0, dt);
            // Walls ahead: turn away, toward the middle of the room.
            let ahead = self.pos + self.heading() * 0.45;
            let outside = ahead.x < FLIGHT_MIN.x || ahead.x > FLIGHT_MAX.x || ahead.z < FLIGHT_MIN.z || ahead.z > FLIGHT_MAX.z;
            if outside {
                let c = v3(ROOM.x * 0.5, 0.0, ROOM.z * 0.5) - v3(self.pos.x, 0.0, self.pos.z);
                let to = c.z.atan2(c.x) + rng.range(-0.6, 0.6);
                self.start_saccade(to);
            } else if rng.chance(SACCADE_RATE * dt) {
                let a = rng.range(0.7, 1.9) * if rng.chance(0.5) { 1.0 } else { -1.0 };
                self.start_saccade(self.yaw + a);
            }
        }
        if rng.chance(0.08 * dt) {
            self.alt_target = rng.range(FLIGHT_MIN.y + 0.3, FLIGHT_MAX.y);
        }
        if rng.chance(0.05 * dt) {
            self.burst = rng.range(1.5, 3.0);
        }
        self.burst = (self.burst - dt).max(0.0);
        let want = if self.burst > 0.0 { 0.6 } else { CRUISE };
        self.speed = damp(self.speed, want, 2.0, dt);
        self.climb(self.alt_target, dt);
        self.pos = self.pos + self.heading() * (self.speed * dt);
        if self.mode_t > self.mode_len && self.saccade.is_none() {
            self.spot = Some(match rng.below(if self.hunger > 0.5 { 5 } else { 3 }) {
                0 => Spot::Table(v3(
                    TABLE_CENTER.x + rng.range(-0.5, 0.5),
                    TABLE_SIZE.y,
                    TABLE_CENTER.z + rng.range(-0.3, 0.3),
                )),
                1 | 3 => Spot::Banana,
                _ => Spot::Bread,
            });
            self.set_mode(Mode::Approach, 20.0);
        }
    }

    fn climb(&mut self, target_y: f32, dt: f32) {
        let want_vy = ((target_y - self.pos.y) * 1.2).clamp(-0.25, 0.25);
        self.vy = damp(self.vy, want_vy, 3.0, dt);
        self.pos.y += self.vy * dt;
        self.pitch = damp(self.pitch, (self.vy * 1.5).clamp(-0.4, 0.4), 5.0, dt);
    }

    fn approach(&mut self, dt: f32, rng: &mut Rng) {
        let spot = self.spot.expect("approach without a spot");
        let target = spot.pos() + v3(0.0, 0.12, 0.0);
        let to = target - self.pos;
        let flat = v3(to.x, 0.0, to.z);
        let dist = flat.len();
        if !self.turn(dt) {
            let want = to.z.atan2(to.x);
            let d = angle_diff(self.yaw, want);
            // Odour-guided casting: big errors get a saccade, small ones a smooth turn.
            if d.abs() > 1.2 && dist > 0.3 {
                self.start_saccade(want + rng.range(-0.2, 0.2));
            } else {
                let rate = 3.0;
                self.yaw += d.clamp(-rate * dt, rate * dt);
                self.roll = damp(self.roll, -d.clamp(-0.5, 0.5), 6.0, dt);
            }
        }
        self.speed = damp(self.speed, (dist * 0.8).clamp(0.05, CRUISE), 3.0, dt);
        self.climb(target.y, dt);
        self.pos = self.pos + self.heading() * (self.speed * dt).min(dist);
        if dist < 0.03 && (to.y).abs() < 0.03 {
            self.set_mode(Mode::Landing, 2.0);
        } else if self.mode_t > self.mode_len {
            self.spot = None;
            self.set_mode(Mode::Flying, rng.range(5.0, 12.0));
        }
    }

    fn land(&mut self, dt: f32, rng: &mut Rng) {
        let spot = self.spot.expect("landing without a spot");
        let rest = spot.pos() + v3(0.0, FLY_LEN * 0.28, 0.0);
        self.speed = damp(self.speed, 0.0, 5.0, dt);
        self.pos = damp3(self.pos, rest, 3.0, dt);
        self.pitch = damp(self.pitch, 0.25 * (1.0 - self.mode_t / self.mode_len), 6.0, dt);
        self.roll = damp(self.roll, 0.0, 6.0, dt);
        if (self.pos - rest).len() < 0.004 || self.mode_t > self.mode_len {
            self.pos = rest;
            self.feeding = spot.is_food() && self.hunger > 0.3 && rng.chance(0.75);
            self.events.push(Event::Land);
            self.set_mode(Mode::Sitting, rng.range(3.0, 8.0) + if self.feeding { 3.0 } else { 0.0 });
            self.groom_t = rng.range(0.5, 2.0);
        }
    }

    fn sit(&mut self, dt: f32, rng: &mut Rng) {
        self.pitch = damp(self.pitch, 0.0, 4.0, dt);
        self.groom_t -= dt;
        if self.groom_t <= 0.0 {
            let grooming = self.grooming < 0.5 && !self.feeding && rng.chance(0.6);
            self.grooming = if grooming { 1.0 } else { 0.0 };
            self.groom_t = rng.range(1.0, 2.5);
            if !grooming && rng.chance(0.4) {
                self.yaw += rng.range(-0.6, 0.6); // a fidget
            }
        }
        if self.mode_t > self.mode_len {
            self.feeding = false;
            self.grooming = 0.0;
            self.events.push(Event::TakeOff);
            self.set_mode(Mode::TakeOff, 0.8);
            self.vy = 0.3;
        }
    }

    fn take_off(&mut self, dt: f32, rng: &mut Rng) {
        self.speed = damp(self.speed, CRUISE * 0.6, 3.0, dt);
        self.pos.y += 0.3 * dt;
        self.pos = self.pos + self.heading() * (self.speed * dt);
        self.pitch = damp(self.pitch, 0.35, 8.0, dt);
        if self.mode_t > self.mode_len {
            self.spot = None;
            self.alt_target = rng.range(0.9, 1.8);
            self.set_mode(Mode::Flying, rng.range(5.0, 15.0));
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub pos: V3,
    pub target: V3,
    pub fov: f32,
}

impl Shot {
    fn lerp(self, o: Shot, t: f32) -> Shot {
        Shot { pos: self.pos.lerp(o.pos, t), target: self.target.lerp(o.target, t), fov: self.fov + (o.fov - self.fov) * t }
    }
}

const BLEND: f32 = 3.0;

pub struct Director {
    pub mode: CameraMode,
    cycling: bool,
    prev: CameraMode,
    since: f32,
    hold: f32,
    follow_pos: V3,
    follow_tgt: V3,
    orbit: f32,
}

impl Director {
    pub fn new(mode: CameraMode, fly: &Fly, rng: &mut Rng) -> Director {
        let cycling = mode == CameraMode::Cycle;
        let start = if cycling { CameraMode::Follow } else { mode };
        Director {
            mode: start,
            cycling,
            prev: start,
            since: BLEND,
            hold: rng.range(30.0, 60.0),
            follow_pos: fly.pos - fly.heading() * 0.3 + v3(0.0, 0.08, 0.0),
            follow_tgt: fly.pos,
            orbit: rng.range(0.0, TAU),
        }
    }

    pub fn name(&self) -> &'static str {
        match self.mode {
            CameraMode::Follow => "follow",
            CameraMode::Room => "room",
            CameraMode::Brain => "brain",
            CameraMode::Cycle => "cycle",
        }
    }

    pub fn step(&mut self, dt: f32, fly: &Fly, rng: &mut Rng) {
        self.since += dt;
        self.orbit = (self.orbit + dt * 0.07) % TAU;
        // The follow rig always runs, so cutting to it is continuous.
        // A three-quarter view from behind and to one side, so the fly reads as a fly;
        // while it sits, the camera walks slowly around it.
        let back = if fly.airborne() { fly.heading() } else { v3((self.orbit * 3.0).cos(), 0.0, (self.orbit * 3.0).sin()) };
        let side = back.cross(V3::UP);
        let lift = if fly.airborne() { 0.07 } else { 0.12 };
        let want = fly.pos - back * 0.14 + side * 0.13 + v3(0.0, lift, 0.0);
        let want = v3(want.x.clamp(0.08, ROOM.x - 0.08), want.y.clamp(0.1, ROOM.y - 0.1), want.z.clamp(0.08, ROOM.z - 0.08));
        self.follow_pos = damp3(self.follow_pos, want, 2.2, dt);
        self.follow_tgt = damp3(self.follow_tgt, fly.pos, 6.0, dt);

        if self.cycling && self.since > self.hold {
            self.prev = self.mode;
            self.mode = match self.mode {
                CameraMode::Follow => CameraMode::Room,
                CameraMode::Room => CameraMode::Brain,
                _ => CameraMode::Follow,
            };
            self.since = 0.0;
            self.hold = rng.range(30.0, 60.0);
        }
    }

    fn shot_for(&self, mode: CameraMode, fly: &Fly) -> Shot {
        match mode {
            CameraMode::Room | CameraMode::Cycle => {
                let c = v3(ROOM.x * 0.5, 0.85, ROOM.z * 0.5);
                let a = self.orbit;
                Shot { pos: c + v3(a.cos() * 1.3, 0.75, a.sin() * 1.3), target: c.lerp(fly.pos, 0.3), fov: 75f32.to_radians() }
            }
            CameraMode::Brain => {
                let a = self.orbit * 1.4 + PI;
                let r = 1.5 + 0.15 * (self.orbit * 3.0).sin();
                Shot {
                    pos: BRAIN_CENTER + v3(a.cos() * r, 0.25, a.sin() * r),
                    target: BRAIN_CENTER + v3(0.0, 0.1, 0.0),
                    fov: 55f32.to_radians(),
                }
            }
            CameraMode::Follow => Shot { pos: self.follow_pos, target: self.follow_tgt, fov: 45f32.to_radians() },
        }
    }

    pub fn shot(&self, fly: &Fly) -> Shot {
        let now = self.shot_for(self.mode, fly);
        if self.since >= BLEND {
            return now;
        }
        self.shot_for(self.prev, fly).lerp(now, smoothstep(self.since / BLEND))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(seed: u64, secs: f32) -> (Fly, Vec<Mode>) {
        let mut rng = Rng::new(seed);
        let mut fly = Fly::new(&mut rng);
        let mut modes = Vec::new();
        let dt = 1.0 / 30.0;
        for _ in 0..(secs / dt) as usize {
            fly.step(dt, &mut rng);
            if modes.last() != Some(&fly.mode) {
                modes.push(fly.mode);
            }
            assert!(fly.pos.x > 0.0 && fly.pos.x < ROOM.x && fly.pos.y > 0.0 && fly.pos.y < ROOM.y);
            assert!(fly.pos.z > 0.0 && fly.pos.z < ROOM.z);
        }
        (fly, modes)
    }

    #[test]
    fn fly_goes_through_its_whole_life() {
        let (_, modes) = run(7, 300.0);
        for m in [Mode::Flying, Mode::Approach, Mode::Landing, Mode::Sitting, Mode::TakeOff] {
            assert!(modes.contains(&m), "never reached {m:?}: {modes:?}");
        }
    }

    #[test]
    fn saccade_rate_is_in_the_animal_band() {
        let mut rng = Rng::new(3);
        let mut fly = Fly::new(&mut rng);
        let dt = 1.0 / 30.0;
        let (mut flying, mut saccades) = (0.0, 0);
        for _ in 0..(600.0 / dt) as usize {
            fly.step(dt, &mut rng);
            if fly.mode == Mode::Flying {
                flying += dt;
                saccades += fly.events.iter().filter(|e| **e == Event::Saccade).count();
            }
        }
        let rate = saccades as f32 / flying;
        assert!((0.3..1.2).contains(&rate), "saccade rate {rate}");
    }

    #[test]
    fn director_cycles_and_blends() {
        let mut rng = Rng::new(1);
        let fly = Fly::new(&mut rng);
        let mut d = Director::new(CameraMode::Cycle, &fly, &mut rng);
        let mut seen = vec![d.mode];
        for _ in 0..(200.0 * 30.0) as usize {
            d.step(1.0 / 30.0, &fly, &mut rng);
            if *seen.last().unwrap() != d.mode {
                seen.push(d.mode);
            }
            let s = d.shot(&fly);
            assert!(s.pos.x.is_finite() && s.fov > 0.0);
        }
        assert!(seen.contains(&CameraMode::Room) && seen.contains(&CameraMode::Brain));
    }
}
