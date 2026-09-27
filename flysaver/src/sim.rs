//! The fly's life and the camera director.
//!
//! Two layers act on the fly, as in "A fly in the Matrix" (web/life.js). The
//! instinct layer is procedural and declared: flight bouts of ~5-15 s, about 0.45
//! saccades per second of flight, sits of 3-8 s, grooming, feeding, wall
//! avoidance. When `cmd` is set, the brain layer overrides it with what the live
//! model's output neurons say (see scene::brain::Live::command): turns, speed,
//! landing, escape, feeding, grooming, and the mushroom body's approach-or-avoid
//! decision over a fruit.

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
/// Horizontal distance over a fruit at which the mushroom body is asked to decide.
const DECISION_REACH: f32 = 0.12;
/// Hunger below which smells do not call (the original's APPETITE): a sated fly's
/// receptors are too weakly driven for the antennal lobe to answer.
const APPETITE: f32 = 0.2;
/// Hover this long over the fruit before asking, so the brain has settled into the
/// smell (about 30 model steps; the rate model's time constant is ~5 steps).
const DECISION_SETTLE: f32 = 1.0;
/// Without an answer within this long after asking, the instinct lands (as the original).
const DECISION_WAIT: f32 = 1.5;
/// A swatter blow: how long the fly tumbles, and how long punishment dopamine shows.
const STUN_S: f32 = 0.4;
const PUNISH_S: f32 = 0.8;

/// What the brain's output neurons ask for this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Command {
    /// DNa02 asymmetry turn in the original's convention (left turn positive), rad/s.
    pub turn_rate: f32,
    /// DNp09: fraction of cruise speed.
    pub speed_factor: f32,
    pub land: bool,
    pub escape: bool,
    pub feed: bool,
    pub groom: bool,
    /// The proboscis motor neuron's activity, which sets how far the proboscis extends.
    pub mn9: f32,
}

/// MN9 activity that fully extends the drawn proboscis, before feeding latches it out.
/// A refused, laced taste (MN9 below the feeding level) shows as a short flick.
const MN9_FULL: f32 = 0.05;

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
    Floor(V3),
    Banana,
    Bread,
}

impl Spot {
    pub fn pos(self) -> V3 {
        match self {
            Spot::Table(p) | Spot::Floor(p) => p,
            Spot::Banana => BANANA + v3(0.0, 0.035, 0.0),
            Spot::Bread => BREAD + v3(0.0, 0.07, 0.0),
        }
    }
    pub fn is_food(self) -> bool {
        matches!(self, Spot::Banana | Spot::Bread)
    }
    pub fn name(self) -> &'static str {
        match self {
            Spot::Table(_) => "table",
            Spot::Floor(_) => "floor",
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
    Escape,
    Hit,
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
    /// The brain layer's command; None flies on instinct alone.
    pub cmd: Option<Command>,
    /// Set while hovering over a fruit, waiting for the mushroom body.
    pub wants_decision: bool,
    decided: Option<bool>,
    decision_t: f32,
    /// The fruit the fly last chose to leave, skipped by the next approach.
    avoided: Option<Spot>,
    pub stun: f32,
    pub punish: f32,
    /// Where the current threat is, so an escape jumps away from it.
    pub threat_at: Option<V3>,
    /// The fruit with sugar on it: the only one the fly feeds on.
    pub sugar: Option<Spot>,
    /// How strongly the sugared fruit is laced with a bitter compound (0 = not at all).
    pub bitter: f32,
    /// Proboscis extension, 0 (retracted) to 1: MN9's reflex when brain-piloted.
    pub proboscis: f32,
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
            cmd: None,
            wants_decision: false,
            decided: None,
            decision_t: 0.0,
            avoided: None,
            stun: 0.0,
            punish: 0.0,
            threat_at: None,
            sugar: None,
            bitter: 0.0,
            proboscis: 0.0,
        }
    }

    /// What the labellum tastes: (sugar, bitter) while sitting on the sugared fruit.
    pub fn tasting(&self) -> Option<(f32, f32)> {
        (!self.airborne() && self.spot.is_some() && self.spot == self.sugar).then_some((1.0, self.bitter))
    }

    /// Put the fly down on `spot` for a long sit (tests and demos).
    pub fn perch(&mut self, spot: Spot) {
        self.spot = Some(spot);
        self.pos = spot.pos() + v3(0.0, FLY_LEN * 0.28, 0.0);
        self.feeding = false;
        self.set_mode(Mode::Sitting, 1e6);
        self.groom_t = 1e6;
    }

    /// The mushroom body's answer for the fruit being approached.
    pub fn decide(&mut self, approach: bool) {
        self.decided = Some(approach);
        self.wants_decision = false;
    }

    /// The giant fibre fired: jump off whatever it is doing, away from `from`.
    pub fn escape(&mut self, from: V3) {
        let away = self.pos - from;
        let yaw = away.z.atan2(away.x);
        self.events.push(Event::Escape);
        self.feeding = false;
        self.grooming = 0.0;
        self.wants_decision = false;
        self.decided = None;
        self.spot = None;
        self.burst = 1.5;
        if self.airborne() && self.mode != Mode::TakeOff {
            self.alt_target = (self.pos.y + 0.4).min(FLIGHT_MAX.y);
            self.set_mode(Mode::Flying, 5.0);
            self.start_saccade(yaw);
        } else {
            self.events.push(Event::TakeOff);
            self.yaw = yaw;
            self.speed = 0.5;
            self.vy = 0.9;
            self.set_mode(Mode::TakeOff, 0.6);
        }
    }

    /// Struck by the swatter: a tumble, and punishment dopamine for a moment.
    pub fn hit(&mut self) {
        self.events.push(Event::Hit);
        self.stun = STUN_S;
        self.punish = PUNISH_S;
        self.feeding = false;
        self.grooming = 0.0;
        if !self.airborne() {
            self.events.push(Event::TakeOff);
            self.set_mode(Mode::TakeOff, 0.8);
            self.vy = 0.4;
        }
    }

    /// Where to set down when the brain asks to land now: the table if over it, else the floor.
    fn landing_spot(&self) -> Spot {
        let (hx, hz) = (TABLE_SIZE.x * 0.5, TABLE_SIZE.z * 0.5);
        let over = (self.pos.x - TABLE_CENTER.x).abs() < hx + 0.1 && (self.pos.z - TABLE_CENTER.z).abs() < hz + 0.1 && self.pos.y > TABLE_SIZE.y;
        if over {
            Spot::Table(v3(
                self.pos.x.clamp(TABLE_CENTER.x - hx + 0.05, TABLE_CENTER.x + hx - 0.05),
                TABLE_SIZE.y,
                self.pos.z.clamp(TABLE_CENTER.z - hz + 0.05, TABLE_CENTER.z + hz - 0.05),
            ))
        } else {
            Spot::Floor(v3(self.pos.x.clamp(0.2, ROOM.x - 0.2), 0.0, self.pos.z.clamp(0.2, ROOM.z - 0.2)))
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
            Mode::Approach if self.wants_decision => "deciding",
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
        self.punish = (self.punish - dt).max(0.0);
        if self.stun > 0.0 {
            // Tumbling from a blow: spin, drop a little, nothing else.
            self.stun -= dt;
            self.roll += 14.0 * dt;
            self.yaw += 6.0 * dt;
            self.pos.y -= 0.15 * dt;
            if self.stun <= 0.0 {
                self.roll = 0.0;
            }
            self.clamp_to_room();
            return;
        }
        if let Some(cmd) = self.cmd {
            self.obey(cmd);
        }
        // The proboscis: MN9's reflex when brain-piloted, else out while feeding.
        let reach = match (self.cmd, self.tasting()) {
            (Some(_), Some(_)) if self.feeding => 1.0,
            (Some(cmd), Some(_)) => (cmd.mn9 / MN9_FULL).clamp(0.0, 1.0).sqrt(),
            (Some(_), None) => 0.0,
            (None, _) => if self.feeding { 1.0 } else { 0.0 },
        };
        self.proboscis = damp(self.proboscis, reach, 8.0, dt);
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
        self.clamp_to_room();
        let now = self.time;
        self.saccade_log.retain(|t| now - *t < 20.0);
    }

    /// Discrete brain commands: escape, land now, feed, groom.
    fn obey(&mut self, cmd: Command) {
        if cmd.escape && self.mode != Mode::TakeOff {
            let from = self.threat_at.unwrap_or(self.pos + self.heading() * 0.3);
            self.escape(from);
            return;
        }
        match self.mode {
            Mode::Flying if cmd.land && self.saccade.is_none() && self.mode_t > 1.0 => {
                self.spot = Some(self.landing_spot());
                self.decided = None;
                self.set_mode(Mode::Approach, 20.0);
            }
            Mode::Sitting => {
                // The proboscis extension reflex: MN9 firing on contact starts a feeding bout,
                // which lasts the sit (as the original's startFeeding).
                if self.tasting().is_some() && cmd.feed && self.hunger > 0.15 {
                    self.feeding = true;
                }
                if cmd.groom && !self.feeding {
                    self.grooming = 1.0;
                }
            }
            _ => {}
        }
    }

    fn clamp_to_room(&mut self) {
        self.pos = v3(
            self.pos.x.clamp(0.05, ROOM.x - 0.05),
            self.pos.y.clamp(0.02, ROOM.y - 0.05),
            self.pos.z.clamp(0.05, ROOM.z - 0.05),
        );
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
            // Brain: DNa02 asymmetry turns (the original's left-positive, our yaw grows rightward).
            if let Some(cmd) = self.cmd {
                self.yaw -= cmd.turn_rate * dt;
            }
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
        let factor = self.cmd.map_or(1.0, |c| c.speed_factor);
        let want = if self.burst > 0.0 { 0.6 } else { CRUISE * factor };
        self.speed = damp(self.speed, want, 2.0, dt);
        self.climb(self.alt_target, dt);
        self.pos = self.pos + self.heading() * (self.speed * dt);
        if self.mode_t > self.mode_len && self.saccade.is_none() {
            let choices = if self.hunger <= APPETITE { 1 } else if self.hunger > 0.5 { 5 } else { 3 };
            let mut spot = match rng.below(choices) {
                0 => Spot::Table(v3(
                    TABLE_CENTER.x + rng.range(-0.5, 0.5),
                    TABLE_SIZE.y,
                    TABLE_CENTER.z + rng.range(-0.3, 0.3),
                )),
                1 | 3 => Spot::Banana,
                _ => Spot::Bread,
            };
            if Some(spot) == self.avoided {
                spot = if spot == Spot::Banana { Spot::Bread } else { Spot::Banana };
            }
            self.spot = Some(spot);
            self.decided = None;
            self.decision_t = 0.0;
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
        // Over a fruit, the mushroom body decides once: land on it, or leave it.
        let mut hold = false;
        if spot.is_food() && dist < DECISION_REACH {
            match self.decided {
                Some(false) => {
                    self.avoided = Some(spot);
                    self.spot = None;
                    self.decided = None;
                    self.alt_target = (self.pos.y + 0.3).min(FLIGHT_MAX.y);
                    self.set_mode(Mode::Flying, rng.range(4.0, 10.0));
                    self.start_saccade(self.yaw + PI + rng.range(-0.5, 0.5));
                    return;
                }
                Some(true) => {}
                None if self.cmd.is_none() => self.decided = Some(true), // instincts approach every smell
                None => {
                    self.decision_t += dt;
                    self.wants_decision = self.decision_t >= DECISION_SETTLE;
                    if self.decision_t > DECISION_SETTLE + DECISION_WAIT {
                        self.decide(true); // no answer: the instinct lands
                    }
                    hold = true;
                }
            }
        }
        let cap = if hold { 0.03 } else { CRUISE };
        self.speed = damp(self.speed, (dist * 0.8).clamp(0.02, cap), 3.0, dt);
        self.climb(target.y, dt);
        self.pos = self.pos + self.heading() * (self.speed * dt).min(dist);
        if !hold && dist < 0.03 && (to.y).abs() < 0.03 {
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
            // Instincts feed on sugar by chance; a brain-piloted fly waits for its MN9.
            self.feeding = self.cmd.is_none() && Some(spot) == self.sugar && self.hunger > 0.15 && rng.chance(0.9);
            if spot.is_food() {
                self.avoided = None;
            }
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

/// Half-size of the swatter's head: the looming disc (the original's hand is 2.5 cm).
pub const SWATTER_R: f32 = 0.04;
const SWATTER_SPEED: f32 = 0.6;
const SWATTER_START: f32 = 0.8;

/// A swatter coming at a sitting fly: the threat the giant fibre exists for.
#[derive(Clone, Debug)]
pub struct Threat {
    pub pos: V3,
    pub vel: V3,
    travelled: f32,
    pub hit: bool,
}

impl Threat {
    /// Aimed at `target` from above and to one side.
    pub fn aimed_at(target: V3, rng: &mut Rng) -> Threat {
        let a = rng.range(0.0, TAU);
        let dir = (v3(a.cos(), -0.9, a.sin())).norm();
        Threat { pos: target - dir * SWATTER_START, vel: dir * SWATTER_SPEED, travelled: 0.0, hit: false }
    }

    /// Move; strike the fly if it is still there. Returns false once the swing is over.
    pub fn step(&mut self, dt: f32, fly: &mut Fly) -> bool {
        self.pos = self.pos + self.vel * dt;
        self.travelled += SWATTER_SPEED * dt;
        if !self.hit && (self.pos - fly.pos).len() < SWATTER_R + 0.02 {
            self.hit = true;
            fly.hit();
        }
        self.travelled < SWATTER_START + 0.25 && self.pos.y > 0.0
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

    fn cmd() -> Command {
        Command { speed_factor: 1.0, ..Command::default() }
    }

    #[test]
    fn brain_escape_jumps_off_away_from_the_threat() {
        let mut rng = Rng::new(2);
        let mut fly = Fly::new(&mut rng);
        fly.perch(Spot::Bread);
        let threat = fly.pos + v3(0.2, 0.2, 0.0);
        fly.threat_at = Some(threat);
        fly.cmd = Some(Command { escape: true, ..cmd() });
        fly.step(1.0 / 30.0, &mut rng);
        assert!(fly.events.contains(&Event::Escape));
        assert_eq!(fly.mode, Mode::TakeOff);
        assert!(fly.heading().dot(threat - fly.pos) < 0.0, "jumped toward the threat");
    }

    #[test]
    fn brain_landing_request_sets_down_on_the_nearest_surface() {
        let mut rng = Rng::new(4);
        let mut fly = Fly::new(&mut rng);
        fly.pos = v3(0.5, 1.2, 0.5); // not over the table
        fly.mode_t = 2.0;
        fly.cmd = Some(Command { land: true, ..cmd() });
        fly.step(1.0 / 30.0, &mut rng);
        assert_eq!(fly.mode, Mode::Approach);
        assert!(matches!(fly.spot, Some(Spot::Floor(_))));
        let mut fly = Fly::new(&mut rng);
        fly.pos = TABLE_CENTER + v3(0.1, 1.0, 0.1);
        fly.mode_t = 2.0;
        fly.cmd = Some(Command { land: true, ..cmd() });
        fly.step(1.0 / 30.0, &mut rng);
        assert!(matches!(fly.spot, Some(Spot::Table(_))));
    }

    #[test]
    fn dna02_asymmetry_turns_the_fly() {
        let mut rng = Rng::new(5);
        let mut fly = Fly::new(&mut rng);
        fly.pos = v3(1.5, 1.2, 1.5);
        let y0 = fly.yaw;
        fly.cmd = Some(Command { turn_rate: 1.0, ..cmd() }); // left turn, original's sign
        let mut quiet = Rng::new(99);
        for _ in 0..3 {
            // a turn only, no spontaneous saccade: step with a fixed draw
            fly.step(1.0 / 30.0, &mut quiet);
        }
        assert!(fly.saccade.is_none(), "seed drew a saccade; pick another");
        assert!(angle_diff(y0, fly.yaw) < -0.05, "a left-positive command must lower our yaw");
    }

    #[test]
    fn mushroom_body_avoid_leaves_the_fruit_and_the_next_approach_skips_it() {
        let mut rng = Rng::new(6);
        let mut fly = Fly::new(&mut rng);
        fly.cmd = Some(cmd());
        fly.spot = Some(Spot::Banana);
        fly.pos = Spot::Banana.pos() + v3(0.05, 0.12, 0.0);
        fly.mode = Mode::Approach;
        for _ in 0..40 {
            fly.step(1.0 / 30.0, &mut rng);
            if fly.wants_decision {
                break;
            }
        }
        assert!(fly.wants_decision, "hovering over the fruit should ask the mushroom body");
        fly.decide(false);
        fly.step(1.0 / 30.0, &mut rng);
        assert_eq!(fly.mode, Mode::Flying);
        assert_eq!(fly.avoided, Some(Spot::Banana));
    }

    #[test]
    fn instincts_approach_every_smell_without_asking() {
        let mut rng = Rng::new(6);
        let mut fly = Fly::new(&mut rng);
        fly.spot = Some(Spot::Banana);
        fly.pos = Spot::Banana.pos() + v3(0.05, 0.12, 0.0);
        fly.mode = Mode::Approach;
        fly.step(1.0 / 30.0, &mut rng);
        assert!(!fly.wants_decision);
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
