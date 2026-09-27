//! Title, live status line and credits, plus the optional Omarchy logo.

use crate::fb::Frame;
use crate::sim::Fly;
use crate::theme::Theme;

pub const TITLE: &str = "A FLY IN THE MATRIX";
pub const CREDIT: &str = "after Cadence · Pragma Research · connectome: BANC release 888 (CC BY)";

fn spaced(s: &str) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push(ch);
    }
    out
}

/// 60000 -> "60,000".
fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn fit(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).chain(std::iter::once('…')).collect()
    }
}

pub struct LiveInfo {
    pub neurons: usize,
    pub active: usize,
    pub ms_per_step: Option<f32>,
}

pub struct HudInfo<'a> {
    pub live: Option<LiveInfo>,
    pub pilot: &'a str,
    /// The brain's latest decision, shown for a few seconds.
    pub note: Option<&'a str>,
    pub fly: &'a Fly,
    pub camera: &'a str,
    pub neurons_drawn: usize,
    pub jitter: i32,
}

pub fn draw(f: &mut Frame, theme: &Theme, info: &HudInfo) {
    let w = f.cols;
    if w < 20 || f.rows < 8 {
        return;
    }
    let title = if w >= 50 { spaced(TITLE) } else { TITLE.to_string() };
    f.text(2 + info.jitter, 1, &fit(&title, w - 4), theme.title);
    f.text(2, 2, &fit("150,802 neurons, brain and nerve cord wired as measured", w - 4), theme.hud);

    let fly = info.fly;
    let recent = fly.saccade_log.iter().rev().take(3).map(|t| format!("{:.0} s saccade", fly.time - t)).collect::<Vec<_>>();
    let mut status = format!(
        "{} · {:.2} m/s · {:.0} cm up · {:.0} Hz · hunger {:.2}",
        fly.status(),
        if fly.airborne() { fly.speed } else { 0.0 },
        fly.pos.y * 100.0,
        fly.wingbeat_hz(),
        fly.hunger
    );
    if let Some(s) = fly.spot.filter(|_| !fly.airborne() || fly.mode == crate::sim::Mode::Approach) {
        status.push_str(&format!(" · {}", s.name()));
    }
    status = format!("{} · {status}", info.pilot);
    let row = f.rows as i32 - 4;
    if let Some(note) = info.note {
        f.text(2, row - 1, &fit(&format!("› {note}"), w - 4), theme.eyes);
    }
    f.text(2, row, &fit(&status, w - 4), theme.title);
    let mut second = match &info.live {
        Some(l) => {
            let mut s = format!("brain live · {} neurons settling · {} active", group(l.neurons), l.active);
            if let Some(ms) = l.ms_per_step {
                s.push_str(&format!(" · {ms:.1} ms/step"));
            }
            s.push_str(" · ");
            s
        }
        None if info.neurons_drawn > 0 => format!("{} of 150,802 neurons drawn · ", info.neurons_drawn),
        None => String::new(),
    };
    second.push_str(&format!("camera {}", info.camera));
    if !recent.is_empty() {
        second.push_str(&format!(" · {}", recent.join(" · ")));
    }
    f.text(2, row + 1, &fit(&second, w - 4), theme.hud);
    let credit = fit(CREDIT, w - 2);
    let col = (w as i32 - credit.chars().count() as i32) / 2;
    f.text(col.max(0), f.rows as i32 - 1, &credit, theme.hud);
}

/// Draw the user's screensaver.txt centred; the rain parts around it.
pub fn logo(f: &mut Frame, theme: &Theme, lines: &[String]) {
    let h = lines.len() as i32;
    let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as i32;
    if h == 0 || w == 0 || w > f.cols as i32 || h > f.rows as i32 {
        return;
    }
    let top = (f.rows as i32 - h) / 2;
    let left = (f.cols as i32 - w) / 2;
    for (k, line) in lines.iter().enumerate() {
        let row = top + k as i32;
        for (j, ch) in line.chars().enumerate() {
            if ch != ' ' {
                f.text(left + j as i32, row, &ch.to_string(), theme.title);
            }
        }
    }
    for row in (top - 1).max(0)..(top + h + 1).min(f.rows as i32) {
        for col in (left - 2).max(0)..(left + w + 2).min(f.cols as i32) {
            f.rain_mask[row as usize * f.cols + col as usize] = true;
        }
    }
}

pub fn load_logo() -> Vec<String> {
    let path = crate::config::home().join(".config/omarchy/branding/screensaver.txt");
    std::fs::read_to_string(path)
        .map(|s| {
            let mut v: Vec<String> = s.lines().map(|l| l.trim_end().to_string()).collect();
            while v.last().is_some_and(|l| l.is_empty()) {
                v.pop();
            }
            v
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_truncates_with_ellipsis() {
        assert_eq!(fit("abcdef", 4), "abc…");
        assert_eq!(fit("abc", 4), "abc");
    }

    #[test]
    fn groups_thousands() {
        assert_eq!(group(60000), "60,000");
        assert_eq!(group(150802), "150,802");
        assert_eq!(group(12), "12");
    }

    #[test]
    fn spaced_title() {
        assert_eq!(spaced("AB C"), "A B   C");
    }
}
