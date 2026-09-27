//! The whole scene: simulation, camera and layers, drawn back to front.

pub mod brain;
pub mod fly;
pub mod hud;
pub mod rain;
pub mod room;

use crate::config::{BrainMode, Config, Eyes, Pilot};
use crate::sim::Threat;
use crate::theme::Rgb;
use crate::fb::Frame;
use crate::raster::Cam;
use crate::rng::Rng;
use crate::sim::{Director, Fly};
use crate::theme::Theme;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Glitch {
    RowShift { from: usize, to: usize, by: i32 },
    Stutter,
    Title,
}

pub struct Scene {
    pub cfg: Config,
    pub theme: Theme,
    pub rng: Rng,
    pub fly: Fly,
    pub director: Director,
    rain: rain::Rain,
    brain: Option<brain::Brain>,
    logo: Vec<String>,
    pub t: f32,
    glitch: Option<(Glitch, f32)>,
    next_glitch: f32,
    /// Sub-pixel height / width, from the terminal's pixel size when it reports one.
    pub aspect: f32,
    /// Show the live brain's wall-clock step cost on the HUD (off for deterministic snapshots).
    pub show_timing: bool,
    /// The swatter, when one is swinging.
    pub threat: Option<Threat>,
    next_threat: f32,
    /// The latest thing the brain did, and when, for the HUD.
    pub note: Option<(String, f32)>,
}

/// The red-eyed mutant's eyes.
pub const RED_EYES: Rgb = Rgb(0xff, 0x30, 0x48);

impl Scene {
    pub fn new(cfg: Config, theme: Theme, seed: u64) -> Scene {
        let mut theme = if cfg.vivid { theme.vivid() } else { theme };
        if cfg.eyes == Eyes::Red {
            theme.eyes = RED_EYES;
        }
        let mut rng = Rng::new(seed);
        let fly = Fly::new(&mut rng);
        let director = Director::new(cfg.camera, &fly, &mut rng);
        let brain = cfg.has_layer("brain").then(|| brain::Brain::new(cfg.brain_points, cfg.brain == BrainMode::Live, cfg.brain_steps));
        let logo = if cfg.has_layer("logo") { hud::load_logo() } else { Vec::new() };
        let next_glitch = rng.range(20.0, 60.0);
        Scene {
            rain: rain::Rain::new(cfg.rain_density, &cfg.rain_glyphs, cfg.vivid),
            cfg,
            theme,
            fly,
            director,
            brain,
            logo,
            t: 0.0,
            glitch: None,
            next_glitch,
            aspect: 1.0,
            show_timing: true,
            threat: None,
            next_threat: 0.0,
            note: None,
            rng,
        }
    }

    /// Is the live brain flying the fly?
    pub fn piloted(&self) -> bool {
        self.cfg.pilot == Pilot::Brain && self.brain.as_ref().is_some_and(|b| b.live.is_some())
    }

    fn say(&mut self, text: String) {
        self.note = Some((text, self.t));
    }

    pub fn step(&mut self, dt: f32, cols: usize, rows: usize) {
        let dt = dt.min(0.1);
        self.t += dt;
        if self.next_threat == 0.0 {
            self.next_threat = self.rng.range(20.0, 50.0);
        }

        // The brain layer: last frame's settled output neurons steer this frame.
        let piloted = self.piloted();
        let live = self.brain.as_ref().and_then(|b| b.live.as_ref());
        self.fly.cmd = if piloted { live.map(|l| l.command()) } else { None };
        self.fly.threat_at = self.threat.as_ref().map(|t| t.pos);
        let (gf, asym) = live.map_or((0.0, 0.0), |l| (l.giant_fibre(), l.turn_asym()));
        self.fly.step(dt, &mut self.rng);
        for e in self.fly.events.clone() {
            match e {
                crate::sim::Event::Escape => self.say(format!("giant fibre {gf:.3} → escape")),
                crate::sim::Event::Hit => self.say(if piloted { "swatted: the giant fibre was too slow".into() } else { "swatted (instincts don't read the giant fibre)".into() }),
                _ => {}
            }
        }
        if piloted && self.fly.cmd.is_some_and(|c| c.turn_rate != 0.0) && self.note.as_ref().is_none_or(|(_, t)| self.t - t > 3.0) {
            self.say(format!("DNa02 turn {asym:+.3}"));
        }

        // Over a fruit, the mushroom body decides.
        if self.fly.wants_decision {
            if let Some(l) = self.brain.as_ref().and_then(|b| b.live.as_ref()).filter(|_| piloted) {
                let p = l.p_approach();
                let approach = self.rng.f() < p as f32;
                let fruit = self.fly.spot.map_or("fruit", |s| s.name());
                self.fly.decide(approach);
                self.say(format!("mushroom body: {fruit}, {} (p {p:.2})", if approach { "approach" } else { "avoid" }));
            }
        }

        // The swatter: it comes for a fly that has been sitting a while.
        if let Some(t) = &mut self.threat {
            if !t.step(dt, &mut self.fly) {
                self.threat = None;
            }
        } else if self.cfg.threats && self.t > self.next_threat && !self.fly.airborne() && self.fly.mode_t > 1.0 {
            self.threat = Some(Threat::aimed_at(self.fly.pos, &mut self.rng));
            self.next_threat = self.t + self.rng.range(40.0, 90.0);
        }

        self.director.step(dt, &self.fly, &mut self.rng);
        let threat_at = self.threat.as_ref().map(|t| t.pos);
        if let Some(b) = &mut self.brain {
            b.step(dt, &self.fly, threat_at, &mut self.rng);
        }
        if self.cfg.has_layer("rain") {
            self.rain.step(dt, cols, rows, &mut self.rng);
        }
        if let Some((_, left)) = &mut self.glitch {
            *left -= dt;
            if *left <= 0.0 {
                self.glitch = None;
            }
        }
        if self.cfg.glitch && self.t > self.next_glitch {
            self.next_glitch = self.t + self.rng.range(20.0, 60.0);
            let len = self.rng.range(0.08, 0.2);
            let kind = match self.rng.below(3) {
                0 => {
                    let from = self.rng.below(rows.max(1));
                    Glitch::RowShift { from, to: from + 1 + self.rng.below(4), by: if self.rng.chance(0.5) { 2 } else { -3 } }
                }
                1 => {
                    self.rain.frozen = len;
                    Glitch::Stutter
                }
                _ => Glitch::Title,
            };
            self.glitch = Some((kind, len));
        }
    }

    pub fn camera(&self, f: &Frame) -> Cam {
        let shot = self.director.shot(&self.fly);
        let mut cam = Cam::look_at(shot.pos, shot.target, shot.fov, f.sub_w(), f.sub_h(), self.aspect);
        let d = (shot.target - shot.pos).len();
        cam.fog_near = d * 0.6;
        cam.fog_far = d + 4.0;
        cam.fog_floor = if self.cfg.vivid { 0.35 } else { 0.12 };
        cam
    }

    pub fn draw(&self, f: &mut Frame) {
        f.clear();
        f.floor = if self.cfg.vivid { 0.5 } else { 0.25 };
        f.rain_mask.iter_mut().for_each(|m| *m = false);
        if !self.logo.is_empty() {
            hud::logo(f, &self.theme, &self.logo);
        }
        if self.cfg.has_layer("rain") {
            self.rain.draw(f, &self.theme);
        }
        let cam = self.camera(f);
        if self.cfg.has_layer("room") {
            room::draw(f, &cam, &self.theme, self.t);
        }
        if let Some(b) = &self.brain {
            b.draw(f, &cam, &self.theme);
        }
        if self.cfg.has_layer("fly") {
            fly::draw(f, &cam, &self.theme, &self.fly);
        }
        if let Some(t) = &self.threat {
            fly::draw_threat(f, &cam, t);
        }
        if self.cfg.has_layer("hud") {
            let jitter = match self.glitch {
                Some((Glitch::Title, _)) => if (self.t * 60.0) as i32 % 2 == 0 { 2 } else { -1 },
                _ => 0,
            };
            let info = hud::HudInfo {
                fly: &self.fly,
                camera: self.director.name(),
                neurons_drawn: self.brain.as_ref().map_or(0, |b| b.len()),
                live: self.brain.as_ref().and_then(|b| b.live.as_ref()).map(|l| hud::LiveInfo {
                    neurons: l.net.n,
                    active: l.active,
                    ms_per_step: self.show_timing.then_some(l.ms_per_step),
                }),
                jitter,
                pilot: if self.piloted() { "brain-piloted" } else { "instincts" },
                note: self.note.as_ref().filter(|(_, t)| self.t - t < 8.0).map(|(s, _)| s.as_str()),
            };
            hud::draw(f, &self.theme, &info);
        }
        f.compose();
        if let Some((Glitch::RowShift { from, to, by }, _)) = self.glitch {
            f.shift_rows(from, to, by);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Pilot;
    use crate::sim::{Event, Spot, Threat};

    /// A sitting fly, a swatter aimed at it; who wins?
    fn swat(pilot: Pilot, seed: u64) -> (bool, bool, Option<String>) {
        let cfg = Config { pilot, threats: false, glitch: false, ..Config::default() };
        let mut s = Scene::new(cfg, Theme::matrix(), seed);
        s.fly.perch(Spot::Bread);
        let dt = 1.0 / 30.0;
        for _ in 0..30 {
            s.step(dt, 80, 24); // settle the brain on the perched fly
        }
        s.threat = Some(Threat::aimed_at(s.fly.pos, &mut Rng::new(seed)));
        let (mut escaped, mut hit) = (false, false);
        for _ in 0..90 {
            s.step(dt, 80, 24);
            escaped |= s.fly.events.contains(&Event::Escape);
            hit |= s.fly.events.contains(&Event::Hit);
        }
        (escaped, hit, s.note.map(|(n, _)| n))
    }

    #[test]
    fn the_giant_fibre_saves_the_brain_piloted_fly() {
        for seed in 1..=5 {
            let (escaped, hit, note) = swat(Pilot::Brain, seed);
            assert!(escaped && !hit, "seed {seed}: escaped {escaped} hit {hit} ({note:?})");
        }
    }

    /// The mushroom body's naive preference over each fruit, from the real wiring.
    fn p_approach_over(spot: Spot) -> f64 {
        let cfg = Config { threats: false, glitch: false, ..Config::default() };
        let mut s = Scene::new(cfg, Theme::matrix(), 3);
        let dt = 1.0 / 30.0;
        s.step(dt, 80, 24); // warm-up and baseline in flight
        for _ in 0..60 {
            s.fly.pos = spot.pos() + crate::math::v3(0.03, 0.12, 0.0);
            s.fly.mode = crate::sim::Mode::Flying;
            s.fly.hunger = 0.7;
            if let Some(b) = &mut s.brain {
                b.step(dt, &s.fly, None, &mut s.rng);
            }
        }
        s.brain.as_ref().unwrap().live.as_ref().unwrap().p_approach()
    }

    #[test]
    fn the_naive_mushroom_body_prefers_the_bread() {
        let (bread, banana) = (p_approach_over(Spot::Bread), p_approach_over(Spot::Banana));
        eprintln!("p(approach): bread {bread:.2}, banana {banana:.2}");
        assert!(bread > 0.6 && banana < 0.45 && bread > banana + 0.25, "bread {bread}, banana {banana}");
    }

    #[test]
    fn escape_fires_before_contact() {
        let cfg = Config { threats: false, glitch: false, ..Config::default() };
        let mut s = Scene::new(cfg, Theme::matrix(), 1);
        s.fly.perch(Spot::Bread);
        for _ in 0..30 {
            s.step(1.0 / 30.0, 80, 24);
        }
        s.threat = Some(Threat::aimed_at(s.fly.pos, &mut Rng::new(1)));
        for k in 0..90 {
            let d = s.threat.as_ref().map(|t| (t.pos - s.fly.pos).len());
            s.step(1.0 / 30.0, 80, 24);
            if s.fly.events.contains(&Event::Escape) {
                let d = d.unwrap_or(0.0);
                eprintln!("escape at frame {k}, swatter {d:.3} m away, {:?}", s.note);
                assert!(d > crate::sim::SWATTER_R + 0.02, "escaped only at contact ({d})");
                return;
            }
        }
        panic!("no escape");
    }

    #[test]
    fn instincts_do_not_read_the_giant_fibre() {
        for seed in 1..=3 {
            let (escaped, hit, _) = swat(Pilot::Instincts, seed);
            assert!(!escaped && hit, "seed {seed}: escaped {escaped} hit {hit}");
        }
    }
}
