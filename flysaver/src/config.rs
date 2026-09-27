//! Configuration: `~/.config/omarchy/flysaver.toml` plus command-line overrides.
//!
//! Both this file and Omarchy's `colors.toml` are flat `key = value` TOML, so a
//! small parser covers them without pulling in a TOML crate.

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    Num(f64),
    Bool(bool),
    List(Vec<String>),
}

/// Parse flat TOML. Tables (`[section]`) are flattened to `section.key`.
pub fn parse_flat_toml(src: &str) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    let mut section = String::new();
    for raw in src.lines() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        let key = k.trim().trim_matches('"').to_string();
        let key = if section.is_empty() { key } else { format!("{section}.{key}") };
        if let Some(val) = parse_value(v.trim()) {
            out.insert(key, val);
        }
    }
    out
}

fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_value(v: &str) -> Option<Value> {
    if let Some(s) = v.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Some(Value::Str(s.to_string()));
    }
    if let Some(s) = v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        return Some(Value::Str(s.to_string()));
    }
    if let Some(inner) = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let items = inner
            .split(',')
            .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|s| !s.is_empty())
            .collect();
        return Some(Value::List(items));
    }
    match v {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {}
    }
    v.replace('_', "").parse::<f64>().ok().map(Value::Num)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraMode {
    Cycle,
    Follow,
    Room,
    Brain,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Palette {
    /// The Omarchy theme, or Matrix green when the theme has no accent colour.
    Theme,
    /// The Omarchy theme even when it is grey (vantablack, white, solitude).
    ThemeStrict,
    Matrix,
}

impl Palette {
    pub fn parse(s: &str) -> Option<Palette> {
        Some(match s {
            "theme" => Palette::Theme,
            "theme-strict" => Palette::ThemeStrict,
            "matrix" => Palette::Matrix,
            _ => return None,
        })
    }
}

/// How colours reach the terminal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Colors {
    /// Truecolor when the terminal says it has it (COLORTERM), else 256.
    Auto,
    TrueColor,
    Xterm256,
}

impl Colors {
    pub fn parse(s: &str) -> Option<Colors> {
        Some(match s {
            "auto" => Colors::Auto,
            "truecolor" | "24bit" => Colors::TrueColor,
            "256" => Colors::Xterm256,
            _ => return None,
        })
    }

    /// Settle `auto` against the terminal's COLORTERM.
    pub fn resolve(self, colorterm: Option<&str>) -> Colors {
        match self {
            Colors::Auto => match colorterm {
                Some("truecolor") | Some("24bit") => Colors::TrueColor,
                _ => Colors::Xterm256,
            },
            fixed => fixed,
        }
    }

    pub fn resolve_env(self) -> Colors {
        self.resolve(std::env::var("COLORTERM").ok().as_deref())
    }

    pub fn name(self) -> &'static str {
        match self {
            Colors::Auto => "auto",
            Colors::TrueColor => "truecolor",
            Colors::Xterm256 => "256",
        }
    }
}

/// Where the brain's firing comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrainMode {
    /// The Cadence rate model on the 60,000-neuron sub-net, fed by the fly's senses.
    Live,
    /// Region pulses tied to what the fly does; no model runs.
    Decorative,
}

/// Who steers the fly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pilot {
    /// The live brain's output neurons override the instincts (needs brain = live).
    Brain,
    /// The procedural instinct layer alone: the original's control condition.
    Instincts,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Eyes {
    /// The red-eyed mutant.
    Red,
    /// The theme's firing colour, as before.
    Theme,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub fps: f32,
    pub battery_fps: f32,
    pub rain_density: f32,
    pub rain_glyphs: String,
    pub layers: Vec<String>,
    pub brain_points: usize,
    pub camera: CameraMode,
    pub palette: Palette,
    pub colors: Colors,
    pub brain: BrainMode,
    pub brain_steps: usize,
    pub vivid: bool,
    pub pilot: Pilot,
    pub threats: bool,
    pub eyes: Eyes,
    pub glitch: bool,
    pub hud: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            fps: 30.0,
            battery_fps: 20.0,
            rain_density: 0.45,
            rain_glyphs: "katakana".into(),
            layers: ["rain", "room", "brain", "fly", "hud"].iter().map(|s| s.to_string()).collect(),
            brain_points: 8000,
            camera: CameraMode::Cycle,
            palette: Palette::Theme,
            colors: Colors::Auto,
            brain: BrainMode::Live,
            brain_steps: 1,
            vivid: true,
            pilot: Pilot::Brain,
            threats: true,
            eyes: Eyes::Red,
            glitch: true,
            hud: true,
        }
    }
}

impl Config {
    pub fn has_layer(&self, name: &str) -> bool {
        self.layers.iter().any(|l| l == name) && (name != "hud" || self.hud)
    }

    pub fn apply(&mut self, kv: &HashMap<String, Value>) -> Vec<String> {
        let mut warnings = Vec::new();
        for (k, v) in kv {
            let ok = match (k.as_str(), v) {
                ("fps", Value::Num(n)) => { self.fps = (*n as f32).clamp(5.0, 60.0); true }
                ("battery_fps", Value::Num(n)) => { self.battery_fps = (*n as f32).clamp(5.0, 60.0); true }
                ("rain_density", Value::Num(n)) => { self.rain_density = (*n as f32).clamp(0.0, 1.0); true }
                ("rain_glyphs", Value::Str(s)) if s == "katakana" || s == "ascii" => { self.rain_glyphs = s.clone(); true }
                ("layers", Value::List(l)) => { self.layers = l.clone(); true }
                ("brain_points", Value::Num(n)) => { self.brain_points = (*n as usize).clamp(500, 20_000); true }
                ("camera", Value::Str(s)) => match parse_camera(s) {
                    Some(c) => { self.camera = c; true }
                    None => false,
                },
                ("palette", Value::Str(s)) => match Palette::parse(s) {
                    Some(p) => { self.palette = p; true }
                    None => false,
                },
                ("colors", Value::Str(s)) => match Colors::parse(s) {
                    Some(c) => { self.colors = c; true }
                    None => false,
                },
                ("colors", Value::Num(n)) if *n == 256.0 => { self.colors = Colors::Xterm256; true }
                ("brain", Value::Str(s)) => match s.as_str() {
                    "live" => { self.brain = BrainMode::Live; true }
                    "decorative" => { self.brain = BrainMode::Decorative; true }
                    _ => false,
                },
                ("brain_steps", Value::Num(n)) => { self.brain_steps = (*n as usize).clamp(1, 4); true }
                ("vivid", Value::Bool(b)) => { self.vivid = *b; true }
                ("pilot", Value::Str(s)) => match s.as_str() {
                    "brain" => { self.pilot = Pilot::Brain; true }
                    "instincts" => { self.pilot = Pilot::Instincts; true }
                    _ => false,
                },
                ("threats", Value::Bool(b)) => { self.threats = *b; true }
                ("eyes", Value::Str(s)) => match s.as_str() {
                    "red" => { self.eyes = Eyes::Red; true }
                    "theme" => { self.eyes = Eyes::Theme; true }
                    _ => false,
                },
                ("glitch", Value::Bool(b)) => { self.glitch = *b; true }
                ("hud", Value::Bool(b)) => { self.hud = *b; true }
                _ => false,
            };
            if !ok {
                warnings.push(format!("ignoring config key {k} = {v:?}"));
            }
        }
        warnings
    }
}

pub fn parse_camera(s: &str) -> Option<CameraMode> {
    Some(match s {
        "cycle" => CameraMode::Cycle,
        "follow" => CameraMode::Follow,
        "room" => CameraMode::Room,
        "brain" => CameraMode::Brain,
        _ => return None,
    })
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

pub fn config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".config"));
    base.join("omarchy").join("flysaver.toml")
}

pub fn load() -> (Config, Vec<String>) {
    let mut cfg = Config::default();
    let warnings = match std::fs::read_to_string(config_path()) {
        Ok(src) => cfg.apply(&parse_flat_toml(&src)),
        Err(_) => Vec::new(),
    };
    (cfg, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_toml() {
        let kv = parse_flat_toml(
            "fps = 24 # comment\nlayers = [\"rain\", \"fly\"]\nhud = false\naccent = \"#82FB9C\"\n[colors.primary]\nbackground = \"0x000000\"\n",
        );
        assert_eq!(kv["fps"], Value::Num(24.0));
        assert_eq!(kv["layers"], Value::List(vec!["rain".into(), "fly".into()]));
        assert_eq!(kv["hud"], Value::Bool(false));
        assert_eq!(kv["accent"], Value::Str("#82FB9C".into()));
        assert_eq!(kv["colors.primary.background"], Value::Str("0x000000".into()));
    }

    #[test]
    fn applies_and_clamps() {
        let mut c = Config::default();
        let w = c.apply(&parse_flat_toml("fps = 500\ncamera = \"brain\"\nbogus = 1\n"));
        assert_eq!(c.fps, 60.0);
        assert_eq!(c.camera, CameraMode::Brain);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn colors_auto_follows_colorterm() {
        assert_eq!(Colors::Auto.resolve(Some("truecolor")), Colors::TrueColor);
        assert_eq!(Colors::Auto.resolve(Some("24bit")), Colors::TrueColor);
        assert_eq!(Colors::Auto.resolve(None), Colors::Xterm256);
        assert_eq!(Colors::Auto.resolve(Some("")), Colors::Xterm256);
        assert_eq!(Colors::Xterm256.resolve(Some("truecolor")), Colors::Xterm256);
        let mut c = Config::default();
        assert!(c.apply(&parse_flat_toml("colors = \"256\"\n")).is_empty());
        assert_eq!(c.colors, Colors::Xterm256);
        assert!(c.apply(&parse_flat_toml("colors = 256\n")).is_empty());
    }

    #[test]
    fn brain_and_vivid_keys() {
        let mut c = Config::default();
        assert_eq!((c.brain, c.brain_steps, c.vivid), (BrainMode::Live, 1, true));
        assert!(c.apply(&parse_flat_toml("brain = \"decorative\"\nbrain_steps = 9\nvivid = false\n")).is_empty());
        assert_eq!((c.brain, c.brain_steps, c.vivid), (BrainMode::Decorative, 4, false));
    }

    #[test]
    fn pilot_threat_and_eye_keys() {
        let mut c = Config::default();
        assert_eq!((c.pilot, c.threats, c.eyes), (Pilot::Brain, true, Eyes::Red));
        assert!(c.apply(&parse_flat_toml("pilot = \"instincts\"\nthreats = false\neyes = \"theme\"\n")).is_empty());
        assert_eq!((c.pilot, c.threats, c.eyes), (Pilot::Instincts, false, Eyes::Theme));
    }

    #[test]
    fn hud_flag_hides_hud_layer() {
        let mut c = Config::default();
        assert!(c.has_layer("hud"));
        c.hud = false;
        assert!(!c.has_layer("hud"));
    }
}
