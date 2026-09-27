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
    /// OKLab chroma: 0 for greys, ~0.1-0.2 for clearly coloured accents.
    pub fn chroma(self) -> f32 {
        xterm::lch(self).1
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

/// OKLab, a perceptual colour space: lightness plus two opponent axes.
pub mod oklab {
    use super::Rgb;

    pub fn from_rgb(c: Rgb) -> (f32, f32, f32) {
        let lin = |v: u8| {
            let v = v as f32 / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        let (r, g, b) = (lin(c.0), lin(c.1), lin(c.2));
        let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
        let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
        let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
        (
            0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
            1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
            0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        )
    }

    /// Back to sRGB, or None when the colour is outside what a screen can show.
    pub fn to_rgb(l: f32, a: f32, b: f32) -> Option<Rgb> {
        let l_ = (l + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
        let m_ = (l - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
        let s_ = (l - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
        let r = 4.076_741_7 * l_ - 3.307_711_6 * m_ + 0.230_969_94 * s_;
        let g = -1.268_438 * l_ + 2.609_757_4 * m_ - 0.341_319_38 * s_;
        let bl = -0.004_196_086_3 * l_ - 0.703_418_6 * m_ + 1.707_614_7 * s_;
        let enc = |v: f32| -> Option<u8> {
            if !(-1e-4..=1.0001).contains(&v) {
                return None;
            }
            let v = v.clamp(0.0, 1.0);
            let e = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            Some((e * 255.0).round().clamp(0.0, 255.0) as u8)
        };
        Some(Rgb(enc(r)?, enc(g)?, enc(bl)?))
    }

    /// Raise OKLab lightness to at least `min_l`, keeping hue and chroma where possible.
    pub fn lighten(c: Rgb, min_l: f32) -> Rgb {
        let (l, a, b) = from_rgb(c);
        if l >= min_l {
            return c;
        }
        // Lighter colours hold less chroma; shrink it until the colour fits.
        let mut k = 1.0f32;
        for _ in 0..20 {
            if let Some(out) = to_rgb(min_l, a * k, b * k) {
                return out;
            }
            k *= 0.85;
        }
        c
    }

    /// Scale a colour's chroma by `k`, keeping lightness and hue, backing off to
    /// the most saturated version a screen can show.
    pub fn saturate(c: Rgb, k: f32) -> Rgb {
        let (l, a, b) = from_rgb(c);
        let (mut lo, mut hi) = (1.0f32, k.max(1.0));
        if let Some(out) = to_rgb(l, a * hi, b * hi) {
            return out;
        }
        for _ in 0..16 {
            let mid = (lo + hi) * 0.5;
            if to_rgb(l, a * mid, b * mid).is_some() { lo = mid } else { hi = mid }
        }
        to_rgb(l, a * lo, b * lo).unwrap_or(c)
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
    pub(super) struct Lch(pub f32, pub f32, pub f32);

    pub(super) fn lch(c: Rgb) -> Lch {
        let (l, a, b) = super::oklab::from_rgb(c);
        Lch(l, a.hypot(b), b.atan2(a))
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

/// The proboscis, fixed like the eyes: amber-gold (sugar), shifting to violet as the
/// taste turns bitter. Chosen to stand apart from the red eyes and the theme-green body.
pub const PROBOSCIS: Rgb = Rgb(0xff, 0xb0, 0x20);
pub const PROBOSCIS_BITTER: Rgb = Rgb(0xa0, 0x70, 0xff);

/// The proboscis at a bitter level: amber, through peach and pale lilac, to violet. The mix
/// is in OKLab because an RGB mix of amber and violet passes through the eyes' dusty red.
pub fn proboscis(bitter: f32) -> Rgb {
    let t = bitter.clamp(0.0, 1.0);
    let (a, b) = (oklab::from_rgb(PROBOSCIS), oklab::from_rgb(PROBOSCIS_BITTER));
    let m = |x: f32, y: f32| x + (y - x) * t;
    oklab::to_rgb(m(a.0, b.0), m(a.1, b.1), m(a.2, b.2)).unwrap_or(PROBOSCIS)
}

/// An accent with less chroma than this is grey (vantablack, white and
/// solitude are 0-0.012; the warm neutrals kanagawa and last-horizon ~0.04).
const GREY_ACCENT: f32 = 0.03;

/// Colour roles from spec section 5.
#[derive(Clone, Debug)]
pub struct Theme {
    /// Where the colours came from, for `flysaver doctor`.
    pub origin: String,
    pub rain: Rgb,
    pub rain_head: Rgb,
    pub wire: Rgb,
    pub fire: Rgb,
    pub fly: Rgb,
    pub hud: Rgb,
    pub title: Rgb,
    /// The fly's compound eyes.
    pub eyes: Rgb,
}

impl Theme {
    /// The original page's palette: #39ff6a on black.
    pub fn matrix() -> Theme {
        Theme {
            origin: "matrix green".into(),
            rain: Rgb(0x39, 0xff, 0x6a),
            rain_head: Rgb(0xd8, 0xff, 0xe4),
            wire: Rgb(0x39, 0xff, 0x6a),
            fire: Rgb(0xb8, 0xff, 0x5a),
            fly: Rgb(0xc8, 0xff, 0xd8),
            hud: Rgb(0x3d, 0x8a, 0x52),
            title: Rgb(0x39, 0xff, 0x6a),
            eyes: Rgb(0xb8, 0xff, 0x5a),
        }
    }

    /// Map a theme's colors.toml onto the roles. Unless `strict`, a theme whose
    /// accent is grey gets the Matrix green instead: following it would turn the
    /// whole scene black and white.
    pub fn from_colors_toml(src: &str, strict: bool) -> Theme {
        let kv = config::parse_flat_toml(src);
        let get = |k: &str| match kv.get(k) {
            Some(Value::Str(s)) => Rgb::parse(s),
            _ => None,
        };
        let m = Theme::matrix();
        let accent = get("accent").or(get("green")).unwrap_or(m.rain);
        let hex = format!("#{:02x}{:02x}{:02x}", accent.0, accent.1, accent.2);
        if !strict && accent.chroma() < GREY_ACCENT {
            return Theme {
                origin: format!("matrix green (the theme's accent {hex} is grey; palette = \"theme-strict\" keeps it)"),
                ..Theme::matrix()
            };
        }
        let fg = get("foreground").unwrap_or(m.fly);
        let head = get("bright_foreground").unwrap_or(fg);
        let fire = get("bright_green").or(get("green")).unwrap_or(accent);
        let hud = get("dark_foreground").unwrap_or(accent.scale(0.5));
        // The saver always runs on black, so every role must read on black.
        Theme {
            origin: format!("Omarchy theme, accent {hex}"),
            rain: accent.at_least(0.35),
            rain_head: head.at_least(0.75),
            wire: accent.at_least(0.35),
            fire: fire.at_least(0.5),
            fly: fg.at_least(0.7),
            hud: hud.at_least(0.3),
            title: accent.at_least(0.45),
            eyes: fire.at_least(0.5),
        }
    }

    /// Firing colour for activity in [0, 1]: the theme's hue, through the fire
    /// colour, to white-hot at full activation.
    pub fn heat(&self, act: f32) -> Rgb {
        let a = act.clamp(0.0, 1.0);
        if a < 0.5 {
            self.wire.mix(self.fire, a * 2.0)
        } else {
            self.fire.mix(self.fire.mix(Rgb(255, 255, 255), 0.75), (a - 0.5) * 2.0)
        }
    }

    /// Vivid roles: OKLab chroma x1.8 and a lightness floor, so even muted theme
    /// accents glow on black; hue is kept and the result stays inside the gamut.
    pub fn vivid(self) -> Theme {
        let v = |c: Rgb| oklab::saturate(oklab::lighten(c, 0.72), 1.8);
        Theme {
            rain: v(self.rain),
            rain_head: oklab::saturate(self.rain_head, 1.3),
            wire: v(self.wire),
            fire: v(self.fire),
            fly: oklab::saturate(self.fly, 1.3),
            hud: oklab::saturate(oklab::lighten(self.hud, 0.6), 1.8),
            title: v(self.title),
            ..self
        }
    }

    pub fn load(p: Palette) -> Theme {
        if p == Palette::Matrix {
            return Theme::matrix();
        }
        let dir = config::home().join(".local/state/omarchy/current");
        match std::fs::read_to_string(dir.join("theme/colors.toml")) {
            Ok(src) => {
                let mut t = Theme::from_colors_toml(&src, p == Palette::ThemeStrict);
                if let Ok(name) = std::fs::read_to_string(dir.join("theme.name")) {
                    t.origin = format!("{} [{}]", t.origin, name.trim());
                }
                t
            }
            Err(_) => Theme { origin: "matrix green (no Omarchy theme found)".into(), ..Theme::matrix() },
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
        let t = Theme::from_colors_toml("accent = \"#1a2a5a\"\nforeground = \"#101010\"\n", false);
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
    fn grey_themes_fall_back_to_matrix_green() {
        // vantablack, white and solitude: no hue to follow.
        for accent in ["#8d8d8d", "#6e6e6e", "#798186"] {
            let t = Theme::from_colors_toml(&format!("accent = \"{accent}\"\nforeground = \"#ffffff\"\n"), false);
            assert_eq!(t.rain, Theme::matrix().rain, "{accent}");
            assert!(t.origin.contains("is grey"), "{}", t.origin);
        }
        // Warm neutrals with a visible hue keep their own colour.
        for accent in ["#dcd7ba", "#b59790"] {
            let t = Theme::from_colors_toml(&format!("accent = \"{accent}\"\n"), false);
            assert_ne!(t.rain, Theme::matrix().rain, "{accent}");
        }
        // theme-strict keeps the grey.
        let t = Theme::from_colors_toml("accent = \"#8d8d8d\"\n", true);
        assert_eq!(t.rain.chroma() < GREY_ACCENT, true);
    }

    #[test]
    fn oklab_round_trips() {
        for c in [Rgb(0x39, 0xff, 0x6a), Rgb(0, 0, 0), Rgb(255, 255, 255), Rgb(0x7a, 0xa2, 0xf7), Rgb(0xe6, 0x8e, 0x0d)] {
            let (l, a, b) = oklab::from_rgb(c);
            let back = oklab::to_rgb(l, a, b).unwrap();
            assert!((back.0 as i32 - c.0 as i32).abs() <= 1 && (back.1 as i32 - c.1 as i32).abs() <= 1 && (back.2 as i32 - c.2 as i32).abs() <= 1, "{c:?} -> {back:?}");
        }
    }

    #[test]
    fn vivid_saturates_but_keeps_hue_and_greys() {
        let t = Theme::from_colors_toml("accent = \"#7daea3\"\n", false).vivid(); // gruvbox, a muted teal
        let before = Rgb(0x7d, 0xae, 0xa3);
        assert!(t.rain.chroma() > before.chroma() * 1.4, "{:?}", t.rain);
        assert!(oklab::from_rgb(t.rain).0 >= 0.71);
        let hue = |c: Rgb| xterm::lch(c).2;
        assert!((hue(t.rain) - hue(before)).abs() < 0.08);
        assert_eq!(oklab::saturate(Rgb(128, 128, 128), 1.8), Rgb(128, 128, 128));
        // Already at the gamut edge: stays a valid colour, not garbage.
        let m = oklab::saturate(Rgb(0x39, 0xff, 0x6a), 1.8);
        assert!(m.1 > 200 && m.1 > m.0 && m.1 > m.2, "{m:?}");
    }

    #[test]
    fn heat_runs_from_hue_to_white_hot() {
        let t = Theme::matrix();
        assert_eq!(t.heat(0.0), t.wire);
        assert_eq!(t.heat(0.5), t.fire);
        assert!(t.heat(1.0).luma() > t.fire.luma());
    }

    /// The proboscis must stand apart from the red eyes and the (Matrix-green) body.
    #[test]
    fn proboscis_colour_is_distinct_from_eyes_and_body() {
        let hue = |c: Rgb| xterm::lch(c).2.to_degrees();
        let apart = |a: Rgb, b: Rgb| {
            let d = (hue(a) - hue(b)).abs() % 360.0;
            d.min(360.0 - d)
        };
        let eyes = Rgb(0xff, 0x30, 0x48);
        let m = Theme::matrix();
        for (name, other) in [("eyes", eyes), ("body", m.fly), ("wireframe", m.wire), ("firing", m.fire)] {
            let d = apart(PROBOSCIS, other);
            eprintln!("proboscis vs {name}: {d:.0} deg");
            assert!(d >= 40.0, "proboscis too close to the {name} ({d:.0} deg)");
        }
        // Every shade on the way to bitter stays off the eyes' red: far in hue, or pale.
        for k in 0..=10 {
            let c = proboscis(k as f32 / 10.0);
            let (d, chroma) = (apart(c, eyes), xterm::lch(c).1);
            eprintln!("proboscis at bitter {:.1}: {c:?}, {d:.0} deg from the eyes, chroma {chroma:.3}", k as f32 / 10.0);
            assert!(d >= 30.0 || chroma < 0.08, "bitter {k}/10 looks like the eyes");
        }
        assert_eq!((proboscis(0.0), proboscis(1.0)), (PROBOSCIS, PROBOSCIS_BITTER));
    }

    #[test]
    fn hackerman_keeps_its_accent() {
        let t = Theme::from_colors_toml("accent = \"#82FB9C\"\nbright_green = \"#9cf7c2\"\n", false);
        assert_eq!(t.rain, Rgb(0x82, 0xfb, 0x9c));
        assert_eq!(t.fire, Rgb(0x9c, 0xf7, 0xc2));
    }
}
