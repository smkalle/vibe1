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
    Theme,
    Matrix,
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
                ("palette", Value::Str(s)) => match s.as_str() {
                    "theme" => { self.palette = Palette::Theme; true }
                    "matrix" => { self.palette = Palette::Matrix; true }
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
    fn hud_flag_hides_hud_layer() {
        let mut c = Config::default();
        assert!(c.has_layer("hud"));
        c.hud = false;
        assert!(!c.has_layer("hud"));
    }
}
