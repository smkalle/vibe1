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

/// The xterm 256-colour palette above the 16 terminal colours: a 6x6x6 cube
/// (16-231) and a 24-step grey ramp (232-255). Indices 0-15 are left alone:
/// terminals (and Omarchy themes) redefine them.
pub mod xterm {
    use super::Rgb;
    use std::f32::consts::{PI, TAU};
    use std::sync::OnceLock;

    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    /// Below this OKLab chroma a colour counts as grey.
    const NEUTRAL: f32 = 0.025;
    /// How far a candidate's hue may drift from the source's.
    const MAX_HUE_DRIFT: f32 = 30.0 * PI / 180.0;

    /// The RGB a standard xterm shows for palette index 16-255.
    pub fn rgb(idx: u8) -> Rgb {
        match idx {
            16..=231 => {
                let i = idx - 16;
                Rgb(LEVELS[(i / 36) as usize], LEVELS[(i / 6 % 6) as usize], LEVELS[(i % 6) as usize])
            }
            232..=255 => {
                let v = 8 + 10 * (idx - 232);
                Rgb(v, v, v)
            }
            _ => Rgb(0, 0, 0),
        }
    }

    /// OKLab as (lightness, chroma, hue angle).
    #[derive(Clone, Copy)]
    struct Lch(f32, f32, f32);

    fn lch(c: Rgb) -> Lch {
        let lin = |v: u8| {
            let v = v as f32 / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        let (r, g, b) = (lin(c.0), lin(c.1), lin(c.2));
        let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
        let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
        let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
        let ll = 0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s;
        let a = 1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s;
        let bb = 0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s;
        Lch(ll, a.hypot(bb), bb.atan2(a))
    }

    fn palette() -> &'static [(u8, Lch)] {
        static P: OnceLock<Vec<(u8, Lch)>> = OnceLock::new();
        P.get_or_init(|| (16..=255u8).map(|i| (i, lch(rgb(i)))).collect())
    }

    /// Nearest palette index that keeps the hue. Coloured sources only match
    /// palette colours of a similar hue (so dim greens never turn grey or red),
    /// greys only match greys, and a colour too dim for the darkest matching
    /// shade fades to black instead of jumping brighter.
    pub fn index(c: Rgb) -> u8 {
        // Darker than the palette's darkest grey (#080808): black, i.e. not drawn.
        if c.0.max(c.1).max(c.2) < 8 {
            return 16;
        }
        let src = lch(c);
        let mut best: Option<(f32, u8, f32)> = None;
        for &(i, p) in palette() {
            let d = if src.1 < NEUTRAL {
                if p.1 >= NEUTRAL {
                    continue;
                }
                (src.0 - p.0).abs()
            } else {
                if p.1 < NEUTRAL {
                    continue;
                }
                let dh = ((p.2 - src.2 + PI).rem_euclid(TAU) - PI).abs();
                if dh > MAX_HUE_DRIFT {
                    continue;
                }
                (src.0 - p.0).powi(2) + 0.5 * (src.1 - p.1).powi(2) + 0.02 * dh * dh
            };
            if best.is_none_or(|(bd, _, _)| d < bd) {
                best = Some((d, i, p.0));
            }
        }
        match best {
            Some((_, i, l)) if src.0 >= l * 0.55 => i,
            _ => 16,
        }
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
    fn xterm_maps_known_colours() {
        assert_eq!(xterm::index(Rgb(0x39, 0xff, 0x6a)), 83); // the original's green
        assert_eq!(xterm::index(Rgb(0, 0, 0)), 16);
        assert_eq!(xterm::index(Rgb(255, 255, 255)), 231);
        assert_eq!(xterm::index(Rgb(128, 128, 128)), 244);
    }

    #[test]
    fn xterm_round_trips_every_palette_colour() {
        for i in 16..=255u8 {
            let back = xterm::index(xterm::rgb(i));
            assert_eq!(xterm::rgb(back), xterm::rgb(i), "index {i} came back as {back}");
        }
    }

    #[test]
    fn xterm_keeps_dim_colours_in_their_hue() {
        // Every brightness step of a green is green or black: never grey, cyan or red.
        for base in [Rgb(0x39, 0xff, 0x6a), Rgb(0x82, 0xfb, 0x9c), Rgb(0xb8, 0xff, 0x5a)] {
            for step in 1..=8 {
                let c = xterm::rgb(xterm::index(base.scale(step as f32 / 8.0)));
                let green = c.1 >= c.0 && c.1 > c.2; // lime may dim to olive
                assert!(green || c == Rgb(0, 0, 0), "{base:?} at {step}/8 became {c:?}");
            }
        }
        // A far wireframe line in the original's green is still visible.
        assert_eq!(xterm::rgb(xterm::index(Rgb(14, 63, 26))), Rgb(0, 95, 0));
    }

    #[test]
    fn xterm_greys_stay_grey() {
        for v in (20..=250).step_by(10) {
            let c = xterm::rgb(xterm::index(Rgb(v, v, v)));
            assert!(c.0 == c.1 && c.1 == c.2, "grey {v} became {c:?}");
        }
    }

    #[test]
    fn xterm_dimming_never_brightens() {
        let base = Rgb(0x82, 0xfb, 0x9c);
        let luma = |k: f32| xterm::rgb(xterm::index(base.scale(k))).luma();
        let mut last = 0.0;
        for step in 1..=8 {
            let l = luma(step as f32 / 8.0);
            assert!(l + 1e-6 >= last, "level {step} got darker");
            last = l;
        }
    }

    #[test]
    fn hackerman_keeps_its_accent() {
        let t = Theme::from_colors_toml("accent = \"#82FB9C\"\nbright_green = \"#9cf7c2\"\n");
        assert_eq!(t.rain, Rgb(0x82, 0xfb, 0x9c));
        assert_eq!(t.fire, Rgb(0x9c, 0xf7, 0xc2));
    }
}
