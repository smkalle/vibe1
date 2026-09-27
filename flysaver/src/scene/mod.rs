//! The whole scene: simulation, camera and layers, drawn back to front.

pub mod brain;
pub mod fly;
pub mod hud;
pub mod rain;
pub mod room;

use crate::config::{BrainMode, Config, Eyes, Pilot};
use crate::memory::Memory;
use crate::sim::{Spot, Threat};
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
    /// Phase 3: the open lesson, the sugar, and what the fly remembers.
    pub lesson: Option<OpenLesson>,
    pub sugar: Spot,
    /// Bitter lacing on the sugared fruit (0 = none).
    pub bitter: f32,
    sugar_elapsed: f32,
    pub stats: MemoryStats,
    memory_path: Option<std::path::PathBuf>,
}

/// A decision over a fruit whose outcome is not in yet.
#[derive(Clone, Copy, Debug)]
pub struct OpenLesson {
    pub fruit: Spot,
    pub approach: bool,
    pub landed: bool,
    pub since: f32,
    /// When it started tasting the sugared fruit, and MN9's peak since.
    pub tasting_since: Option<f32>,
    pub mn9_peak: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    pub lessons: u64,
    pub rewards: u64,
    pub blows: u64,
    /// Last approach probability per fruit (banana, bread); NaN until asked.
    pub last_p: [f32; 2],
    pub changed: usize,
}

fn fruit_index(s: Spot) -> usize {
    if s == Spot::Bread { 1 } else { 0 }
}

/// How long the fly tastes the sugared fruit before the lesson's outcome.
const TASTE_S: f32 = 2.0;
/// How often a newly placed sugar is laced with bitter.
const LACE_CHANCE: f32 = 0.4;

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
        let brain = cfg.has_layer("brain").then(|| brain::Brain::new(cfg.brain_points, cfg.brain == BrainMode::Live, cfg.brain_steps, cfg.learning));
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
            lesson: None,
            sugar: Spot::Banana,
            bitter: 0.0,
            sugar_elapsed: 0.0,
            stats: MemoryStats { last_p: [f32::NAN; 2], ..MemoryStats::default() },
            memory_path: None,
            rng,
        }
    }

    /// Is the live brain flying the fly?
    pub fn piloted(&self) -> bool {
        self.cfg.pilot == Pilot::Brain && self.brain.as_ref().is_some_and(|b| b.live.is_some())
    }

    fn live_mut(&mut self) -> Option<&mut brain::Live> {
        self.brain.as_mut().and_then(|b| b.live.as_mut())
    }

    /// Start the lessons: load the fly's memory (when remembering), else put the sugar somewhere.
    pub fn start_memory(&mut self, path: Option<std::path::PathBuf>) {
        self.sugar = if self.rng.chance(0.5) { Spot::Banana } else { Spot::Bread };
        self.lace();
        self.memory_path = path.filter(|_| self.cfg.remember && self.cfg.learning);
        let Some(p) = self.memory_path.clone() else { return };
        let Some(live) = self.live_mut() else { return };
        let (fnv, seam) = (live.net.weights_fnv.0, live.net.seam.len());
        let critic = live.net.sets.get(crate::learner::CRITIC).map_or(0, |v| v.len());
        if let Ok(m) = Memory::load(&p, fnv, seam, critic) {
            live.remember(&m);
            self.sugar = if m.sugar == 1 { Spot::Bread } else { Spot::Banana };
            self.sugar_elapsed = m.sugar_elapsed;
            self.bitter = if self.cfg.bitter { m.bitter } else { 0.0 };
            self.stats = MemoryStats { lessons: m.lessons, rewards: m.rewards, blows: m.blows, last_p: m.last_p, changed: 0 };
            self.refresh_changed();
            self.say(format!("remembers {} lessons", m.lessons));
        }
    }

    fn refresh_changed(&mut self) {
        if let Some(live) = self.brain.as_ref().and_then(|b| b.live.as_ref()) {
            if let Some(l) = &live.learner {
                self.stats.changed = l.changes(&live.net).0;
            }
        }
    }

    /// Write the memory now (after a lesson, and on exit).
    pub fn save_memory(&self) {
        let Some(path) = &self.memory_path else { return };
        let Some(live) = self.brain.as_ref().and_then(|b| b.live.as_ref()) else { return };
        let Some(l) = &live.learner else { return };
        let m = Memory {
            efficacy: l.efficacy.clone(),
            w_critic: l.w_critic.clone(),
            b_critic: l.b_critic,
            lessons: self.stats.lessons,
            rewards: self.stats.rewards,
            blows: self.stats.blows,
            sugar: fruit_index(self.sugar) as u8,
            sugar_elapsed: self.sugar_elapsed,
            last_p: self.stats.last_p,
            bitter: self.bitter,
        };
        let _ = m.save(path, live.net.weights_fnv.0);
    }

    /// The lesson's outcome: teach the learner, count it, remember it.
    fn close_lesson(&mut self, reward: f64, why: &str) {
        let Some(lesson) = self.lesson.take() else { return };
        let lesson_out = self.live_mut().and_then(|l| l.outcome(reward));
        if let Some(out) = lesson_out {
            self.stats.lessons += 1;
            if reward > 0.0 {
                self.stats.rewards += 1;
            }
            if reward < 0.0 {
                self.stats.blows += 1;
            }
            self.refresh_changed();
            let r = if reward.fract() == 0.0 { format!("{reward:+}") } else { format!("{reward:+.2}") };
            self.say(format!("lesson: {}, {why} {r} (dopamine {:+.2})", lesson.fruit.name(), out.dopamine));
            self.save_memory();
        }
    }

    /// Phase 3: decide over a fruit, and close lessons on their outcomes.
    fn lessons(&mut self, piloted: bool) {
        if self.fly.wants_decision && piloted {
            // A lesson still open (the fly came back without an outcome): it ends with nothing.
            self.close_lesson(0.0, "left it");
            let u = self.rng.f() as f64;
            let fruit = self.fly.spot.unwrap_or(Spot::Banana);
            if let Some((approach, p)) = self.live_mut().map(|l| l.decide(u)) {
                self.fly.decide(approach);
                self.stats.last_p[fruit_index(fruit)] = p as f32;
                let learning = self.brain.as_ref().and_then(|b| b.live.as_ref()).is_some_and(|l| l.learner.is_some());
                if learning {
                    self.lesson = Some(OpenLesson { fruit, approach, landed: false, since: self.t, tasting_since: None, mn9_peak: 0.0 });
                }
                self.say(format!("mushroom body: {}, {} (p {p:.2})", fruit.name(), if approach { "approach" } else { "avoid" }));
            }
        }
        let Some(l) = self.lesson else { return };
        let on_it = !self.fly.airborne() && self.fly.spot == Some(l.fruit);
        for e in self.fly.events.clone() {
            match e {
                crate::sim::Event::Land if on_it && l.approach => {
                    if let Some(open) = &mut self.lesson {
                        open.landed = true; // stays open while it sits: leaving pays 0, a blow -1
                        if l.fruit == self.sugar {
                            open.tasting_since = Some(self.t); // the labellum tastes; MN9 decides
                        }
                    }
                }
                crate::sim::Event::Hit if l.landed => {
                    self.close_lesson(-1.0, "struck there");
                    return;
                }
                crate::sim::Event::TakeOff if l.landed => {
                    self.close_lesson(0.0, "left it empty");
                    return;
                }
                _ => {}
            }
        }
        // Two seconds of tasting the sugared fruit: sugar if it fed, minus how bitter it was.
        if let Some(t0) = l.tasting_since {
            let now = self.fly.cmd.map_or(0.0, |c| c.mn9);
            if let Some(open) = &mut self.lesson {
                open.mn9_peak = open.mn9_peak.max(now);
            }
            if self.t - t0 >= TASTE_S {
                let fed = self.fly.feeding;
                let mn9 = self.lesson.map_or(0.0, |o| o.mn9_peak);
                let reward = if fed { 1.0 } else { 0.0 } - self.bitter as f64;
                let taste = if self.bitter > 0.0 { format!("sugar + caffeine {:.2}", self.bitter) } else { "sugar".into() };
                let what = format!("{taste}, MN9 peak {mn9:.3} → {}", if fed { "fed" } else { "refused" });
                self.close_lesson(reward, &what);
                return;
            }
        }
        let age = self.t - l.since;
        let approaching = matches!(self.fly.mode, crate::sim::Mode::Approach | crate::sim::Mode::Landing) && self.fly.spot == Some(l.fruit);
        if !l.approach && age > 3.0 {
            self.close_lesson(0.0, "avoided");
        } else if l.approach && !l.landed && !approaching && !on_it && age > 1.0 {
            self.close_lesson(0.0, "never got there");
        }
    }

    /// "memory 12 lessons · 9,173 synapses changed · sugar on the bread · p banana 0.38 / bread 0.68"
    fn memory_line(&self) -> Option<String> {
        let learning = self.brain.as_ref().and_then(|b| b.live.as_ref()).is_some_and(|l| l.learner.is_some());
        if !learning || !self.piloted() {
            return None;
        }
        let p = |x: f32| if x.is_nan() { "?".to_string() } else { format!("{x:.2}") };
        Some(format!(
            "memory {} lessons · {} synapses changed · sugar on the {}{} · p banana {} / bread {}",
            self.stats.lessons,
            self.stats.changed,
            self.sugar.name(),
            if self.bitter > 0.0 { format!(" (caffeine {:.2})", self.bitter) } else { String::new() },
            p(self.stats.last_p[0]),
            p(self.stats.last_p[1])
        ))
    }

    fn move_sugar(&mut self, dt: f32) {
        self.sugar_elapsed += dt;
        if self.sugar_elapsed > self.cfg.sugar_minutes * 60.0 {
            self.sugar_elapsed = 0.0;
            self.sugar = if self.sugar == Spot::Banana { Spot::Bread } else { Spot::Banana };
            self.lace();
            let laced = if self.bitter > 0.0 { format!(", laced with caffeine {:.2}", self.bitter) } else { String::new() };
            self.say(format!("the sugar moved to the {}{laced}", self.sugar.name()));
            self.save_memory();
        }
    }

    /// Maybe lace the sugared fruit with a bitter compound, at a random strength.
    fn lace(&mut self) {
        self.bitter = if self.cfg.bitter && self.rng.chance(LACE_CHANCE) { self.rng.range(0.2, 1.0) } else { 0.0 };
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
        self.fly.sugar = Some(self.sugar);
        self.fly.bitter = self.bitter;
        self.fly.step(dt, &mut self.rng);
        // The swatter: it comes for a fly that has been sitting a while. It moves right
        // after the fly, so a blow is among this frame's events for the notes and lessons.
        if let Some(t) = &mut self.threat {
            if !t.step(dt, &mut self.fly) {
                self.threat = None;
            }
        } else if self.cfg.threats && self.t > self.next_threat && !self.fly.airborne() && self.fly.mode_t > 1.0 {
            self.threat = Some(Threat::aimed_at(self.fly.pos, &mut self.rng));
            self.next_threat = self.t + self.rng.range(40.0, 90.0);
        }

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

        // Over a fruit, the mushroom body decides; outcomes teach it (Phase 3).
        self.lessons(piloted);
        self.move_sugar(dt);

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
            if self.piloted() {
                room::sugar(f, &cam, &self.theme, self.sugar.pos(), self.t);
            }
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
                memory: self.memory_line(),
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

    /// A few minutes of the real scene: decisions become lessons, sugar teaches.
    #[test]
    fn lessons_happen_and_change_the_memory_site() {
        let cfg = Config { glitch: false, ..Config::default() };
        let mut s = Scene::new(cfg, Theme::matrix(), 3);
        s.start_memory(None);
        for _ in 0..(4 * 60 * 30) {
            s.step(1.0 / 30.0, 80, 24);
        }
        let st = s.stats;
        assert!(st.lessons >= 4, "only {} lessons in 4 minutes", st.lessons);
        assert!(st.rewards >= 1, "no sugar found in 4 minutes");
        assert!(st.changed > 1000, "only {} synapses changed", st.changed);
    }

    #[test]
    fn memory_survives_to_the_next_idle_session() {
        let dir = std::env::temp_dir().join(format!("flysaver-scene-mem-{}", std::process::id()));
        let path = dir.join("memory.bin");
        let cfg = Config { glitch: false, ..Config::default() };
        let mut a = Scene::new(cfg.clone(), Theme::matrix(), 3);
        a.start_memory(Some(path.clone()));
        for _ in 0..(3 * 60 * 30) {
            a.step(1.0 / 30.0, 80, 24);
        }
        a.save_memory();
        let (lessons, sugar, changed) = (a.stats.lessons, a.sugar, a.stats.changed);
        let learned = a.brain.as_ref().unwrap().live.as_ref().unwrap().learner.as_ref().unwrap().efficacy.clone();
        assert!(lessons > 0);
        // The next session: a fresh process would start the same way.
        let mut b = Scene::new(cfg, Theme::matrix(), 99);
        b.start_memory(Some(path.clone()));
        let lb = b.brain.as_ref().unwrap().live.as_ref().unwrap().learner.as_ref().unwrap();
        assert_eq!(lb.efficacy, learned);
        assert_eq!((b.stats.lessons, b.sugar, b.stats.changed), (lessons, sugar, changed));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression: a blow landed after the lesson logic ran and was lost, so no lesson ever
    /// got its -1. The swatter now steps right after the fly.
    #[test]
    fn a_blow_on_the_fruit_teaches_minus_one() {
        let cfg = Config { pilot: Pilot::Instincts, threats: false, glitch: false, ..Config::default() };
        let mut s = Scene::new(cfg, Theme::matrix(), 4);
        s.start_memory(None);
        s.sugar = Spot::Bread;
        s.fly.perch(Spot::Banana);
        for _ in 0..30 {
            s.step(1.0 / 30.0, 80, 24);
        }
        let _ = s.live_mut().unwrap().decide(0.0); // an approach decision the outcome can credit
        s.lesson = Some(OpenLesson { fruit: Spot::Banana, approach: true, landed: true, since: s.t, tasting_since: None, mn9_peak: 0.0 });
        s.threat = Some(Threat::aimed_at(s.fly.pos, &mut Rng::new(4)));
        for _ in 0..90 {
            s.step(1.0 / 30.0, 80, 24);
        }
        assert_eq!(s.stats.blows, 1, "the blow was not taught ({:?})", s.note);
        assert!(s.lesson.is_none());
        assert!(s.note.as_ref().is_some_and(|(n, _)| n.contains("struck there -1")), "{:?}", s.note);
    }

    /// A brain-piloted scene whose brain has tasted sweet sugar once and flown off (as in
    /// the screensaver after its first sugar: a brain's first taste bursts ~10x higher),
    /// now hovering hungry over the sugared bread, laced at `bitter`.
    fn experienced(bitter: f32) -> Scene {
        let cfg = Config { threats: false, glitch: false, bitter: false, ..Config::default() };
        let mut s = Scene::new(cfg, Theme::matrix(), 5);
        s.start_memory(None);
        s.sugar = Spot::Bread;
        let run = |s: &mut Scene, n: usize| (0..n).for_each(|_| s.step(1.0 / 30.0, 80, 24));
        s.fly.pos = crate::math::v3(1.0, 1.3, 1.0);
        run(&mut s, 40);
        s.fly.perch(Spot::Bread);
        run(&mut s, 120);
        s.fly = Fly::new(&mut Rng::new(9));
        s.fly.pos = crate::math::v3(0.5, 1.4, 0.5);
        run(&mut s, 150);
        s.lesson = None;
        s.bitter = bitter;
        s.fly.hunger = 0.8;
        s.fly.pos = Spot::Bread.pos() + crate::math::v3(0.0, 0.15, 0.0);
        run(&mut s, 30);
        s
    }

    /// `experienced`, then it lands on the bread and tastes it for 3 s.
    fn taste(bitter: f32) -> Scene {
        let mut s = experienced(bitter);
        s.fly.perch(Spot::Bread);
        s.fly.hunger = 0.8;
        for _ in 0..90 {
            s.step(1.0 / 30.0, 80, 24);
        }
        s
    }

    #[test]
    fn the_proboscis_extension_reflex_decides_feeding() {
        let sweet = taste(0.0);
        let laced = taste(1.0);
        eprintln!(
            "sweet: MN9 {:.3}, proboscis {:.2}, feeding {}; laced 1.0: MN9 {:.3}, proboscis {:.2}, feeding {}",
            sweet.fly.cmd.unwrap().mn9, sweet.fly.proboscis, sweet.fly.feeding,
            laced.fly.cmd.unwrap().mn9, laced.fly.proboscis, laced.fly.feeding
        );
        assert!(sweet.fly.feeding && sweet.fly.proboscis > 0.5);
        assert!(!laced.fly.feeding && laced.fly.proboscis < 0.1);
    }

    #[test]
    fn a_laced_fruit_is_a_bitter_lesson() {
        for (bitter, fed, sign) in [(0.0f32, true, 1.0f64), (1.0, false, -1.0)] {
            let mut s = experienced(bitter);
            let before = s.stats.lessons;
            let _ = s.live_mut().unwrap().decide(0.0);
            // It lands: the lesson starts tasting as the Land event would start it.
            s.fly.perch(Spot::Bread);
            s.fly.hunger = 0.8;
            s.lesson = Some(OpenLesson { fruit: Spot::Bread, approach: true, landed: true, since: s.t, tasting_since: Some(s.t), mn9_peak: 0.0 });
            for _ in 0..75 {
                s.step(1.0 / 30.0, 80, 24);
            }
            let note = s.note.clone().map(|(n, _)| n).unwrap_or_default();
            eprintln!("bitter {bitter}: {note}");
            assert!(s.lesson.is_none() && s.stats.lessons == before + 1, "{note}");
            assert!(note.contains(if fed { "fed" } else { "refused" }), "{note}");
            let peak: f32 = note.split("MN9 peak ").nth(1).unwrap()[..5].parse().unwrap();
            assert!(if fed { peak > 0.005 } else { peak < 0.005 }, "{note}");
            let dopamine: f64 = note.rsplit("dopamine ").next().unwrap().trim_end_matches(')').parse().unwrap();
            assert!(dopamine * sign > 0.0, "{note}");
        }
    }

    #[test]
    fn instincts_do_not_read_the_giant_fibre() {
        for seed in 1..=3 {
            let (escaped, hit, _) = swat(Pilot::Instincts, seed);
            assert!(!escaped && hit, "seed {seed}: escaped {escaped} hit {hit}");
        }
    }
}

