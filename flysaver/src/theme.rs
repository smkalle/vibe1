//! Colours: the active Omarchy theme mapped onto the scene's roles.

use crate::config::{self, Palette, Value};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub fn scale(self, k: f32) -> Rgb {
        let k = k.clamp(0.0, 1.0);
        Rgb((self.0 as f32 * k) as u8, (self.1 as f32 * k) as u8, (self.2 as f32 * k) as u8)
    }
    pub fn mix(self, o: Rgb, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let m = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
        Rgb(m(self.0, o.0), m(self.1, o.1), m(self.2, o.2))
    }
    pub fn luma(self) -> f32 {
        (0.2126 * self.0 as f32 + 0.7152 * self.1 as f32 + 0.0722 * self.2 as f32) / 255.0
    }
    /// Lift a colour until it reads on black (light themes carry dark accents).
    pub fn at_least(self, min_luma: f32) -> Rgb {
        let l = self.luma();
        if l >= min_luma {
            return self;
        }
        if l < 0.01 {
            let v = (min_luma * 255.0) as u8;
            return Rgb(v, v, v);
        }
        let mut c = self;
        for _ in 0..32 {
            if c.luma() >= min_luma {
                break;
            }
            c = c.mix(Rgb(255, 255, 255), 0.08);
        }
        c
    }
    pub fn parse(s: &str) -> Option<Rgb> {
        let h = s.trim().trim_start_matches('#').trim_start_matches("0x");
        if h.len() < 6 {
            return None;
        }
        let p = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok();
        Some(Rgb(p(0)?, p(2)?, p(4)?))
    }
}

/// Colour roles from spec section 5.
#[derive(Clone, Debug)]
pub struct Theme {
    pub rain: Rgb,
    pub rain_head: Rgb,
    pub wire: Rgb,
    pub fire: Rgb,
    pub fly: Rgb,
    pub hud: Rgb,
    pub title: Rgb,
}

impl Theme {
    /// The original page's palette: #39ff6a on black.
    pub fn matrix() -> Theme {
        Theme {
            rain: Rgb(0x39, 0xff, 0x6a),
            rain_head: Rgb(0xd8, 0xff, 0xe4),
            wire: Rgb(0x39, 0xff, 0x6a),
            fire: Rgb(0xb8, 0xff, 0x5a),
            fly: Rgb(0xc8, 0xff, 0xd8),
            hud: Rgb(0x3d, 0x8a, 0x52),
            title: Rgb(0x39, 0xff, 0x6a),
        }
    }

    pub fn from_colors_toml(src: &str) -> Theme {
        let kv = config::parse_flat_toml(src);
        let get = |k: &str| match kv.get(k) {
            Some(Value::Str(s)) => Rgb::parse(s),
            _ => None,
        };
        let m = Theme::matrix();
        let accent = get("accent").or(get("green")).unwrap_or(m.rain);
        let fg = get("foreground").unwrap_or(m.fly);
        let head = get("bright_foreground").unwrap_or(fg);
        let fire = get("bright_green").or(get("green")).unwrap_or(accent);
        let hud = get("dark_foreground").unwrap_or(accent.scale(0.5));
        // The saver always runs on black, so every role must read on black.
        Theme {
            rain: accent.at_least(0.35),
            rain_head: head.at_least(0.75),
            wire: accent.at_least(0.35),
            fire: fire.at_least(0.5),
            fly: fg.at_least(0.7),
            hud: hud.at_least(0.3),
            title: accent.at_least(0.45),
        }
    }

    pub fn load(p: Palette) -> Theme {
        if p == Palette::Matrix {
            return Theme::matrix();
        }
        let dir = config::home().join(".local/state/omarchy/current");
        match std::fs::read_to_string(dir.join("theme/colors.toml")) {
            Ok(src) => Theme::from_colors_toml(&src),
            Err(_) => Theme::matrix(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_forms() {
        assert_eq!(Rgb::parse("#82FB9C"), Some(Rgb(0x82, 0xfb, 0x9c)));
        assert_eq!(Rgb::parse("0x000000"), Some(Rgb(0, 0, 0)));
        assert_eq!(Rgb::parse("zz"), None);
    }

    #[test]
    fn light_theme_colours_are_lifted() {
        let t = Theme::from_colors_toml("accent = \"#1a1a1a\"\nforeground = \"#101010\"\n");
        assert!(t.rain.luma() >= 0.35);
        assert!(t.fly.luma() >= 0.7);
    }

    #[test]
    fn hackerman_keeps_its_accent() {
        let t = Theme::from_colors_toml("accent = \"#82FB9C\"\nbright_green = \"#9cf7c2\"\n");
        assert_eq!(t.rain, Rgb(0x82, 0xfb, 0x9c));
        assert_eq!(t.fire, Rgb(0x9c, 0xf7, 0xc2));
    }
}
