//! The whole scene: simulation, camera and layers, drawn back to front.

pub mod brain;
pub mod fly;
pub mod hud;
pub mod rain;
pub mod room;

use crate::config::{BrainMode, Config};
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
}

impl Scene {
    pub fn new(cfg: Config, theme: Theme, seed: u64) -> Scene {
        let theme = if cfg.vivid { theme.vivid() } else { theme };
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
            rng,
        }
    }

    pub fn step(&mut self, dt: f32, cols: usize, rows: usize) {
        let dt = dt.min(0.1);
        self.t += dt;
        self.fly.step(dt, &mut self.rng);
        self.director.step(dt, &self.fly, &mut self.rng);
        if let Some(b) = &mut self.brain {
            b.step(dt, &self.fly, &mut self.rng);
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
            };
            hud::draw(f, &self.theme, &info);
        }
        f.compose();
        if let Some((Glitch::RowShift { from, to, by }, _)) = self.glitch {
            f.shift_rows(from, to, by);
        }
    }
}
