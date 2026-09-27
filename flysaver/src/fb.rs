//! Framebuffer: a braille sub-pixel canvas for the vector layers, a glyph layer for
//! the rain, an overlay for text, and a diffing ANSI writer.

use crate::theme::Rgb;
use std::fmt::Write as _;

/// Sub-pixels dimmer than this are not drawn.
const DOT_THRESHOLD: f32 = 0.06;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glyph {
    pub ch: char,
    pub fg: Rgb,
    pub i: f32,
}

impl Glyph {
    pub const EMPTY: Glyph = Glyph { ch: ' ', fg: Rgb(0, 0, 0), i: 0.0 };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Out {
    pub ch: char,
    pub fg: Rgb,
}

impl Out {
    pub const BLANK: Out = Out { ch: ' ', fg: Rgb(0, 0, 0) };
}

pub struct Frame {
    pub cols: usize,
    pub rows: usize,
    /// Sub-pixel intensity and colour, (2*cols) x (4*rows).
    pub dots: Vec<f32>,
    pub dot_col: Vec<Rgb>,
    /// Rain glyphs, one per cell.
    pub rain: Vec<Glyph>,
    /// Text overlay (HUD, logo); wins over everything.
    pub overlay: Vec<Option<Out>>,
    /// Cells the rain must leave alone (behind the logo).
    pub rain_mask: Vec<bool>,
    pub out: Vec<Out>,
}

impl Frame {
    pub fn new(cols: usize, rows: usize) -> Frame {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let n = cols * rows;
        Frame {
            cols,
            rows,
            dots: vec![0.0; n * 8],
            dot_col: vec![Rgb::default(); n * 8],
            rain: vec![Glyph::EMPTY; n],
            overlay: vec![None; n],
            rain_mask: vec![false; n],
            out: vec![Out::BLANK; n],
        }
    }

    pub fn sub_w(&self) -> usize {
        self.cols * 2
    }
    pub fn sub_h(&self) -> usize {
        self.rows * 4
    }

    pub fn clear(&mut self) {
        self.dots.iter_mut().for_each(|d| *d = 0.0);
        self.rain.iter_mut().for_each(|g| *g = Glyph::EMPTY);
        self.overlay.iter_mut().for_each(|o| *o = None);
    }

    /// Max-blend a sub-pixel.
    #[inline]
    pub fn plot(&mut self, x: i32, y: i32, i: f32, c: Rgb) {
        if x < 0 || y < 0 || x as usize >= self.sub_w() || y as usize >= self.sub_h() {
            return;
        }
        let idx = y as usize * self.sub_w() + x as usize;
        if i > self.dots[idx] {
            self.dots[idx] = i;
            self.dot_col[idx] = c;
        }
    }

    pub fn text(&mut self, col: i32, row: i32, s: &str, fg: Rgb) {
        if row < 0 || row as usize >= self.rows {
            return;
        }
        for (k, ch) in s.chars().enumerate() {
            let c = col + k as i32;
            if c < 0 {
                continue;
            }
            if c as usize >= self.cols {
                break;
            }
            let idx = row as usize * self.cols + c as usize;
            self.overlay[idx] = if ch == '\0' { None } else { Some(Out { ch, fg }) };
        }
    }

    /// Resolve the layers into one glyph and colour per cell.
    pub fn compose(&mut self) {
        let sw = self.sub_w();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let idx = row * self.cols + col;
                if let Some(o) = self.overlay[idx] {
                    self.out[idx] = o;
                    continue;
                }
                let mut bits = 0u32;
                let mut vmax = 0.0f32;
                let mut vcol = Rgb::default();
                for dy in 0..4 {
                    for dx in 0..2 {
                        let si = (row * 4 + dy) * sw + col * 2 + dx;
                        let v = self.dots[si];
                        if v > DOT_THRESHOLD {
                            bits |= braille_bit(dx, dy);
                            if v > vmax {
                                vmax = v;
                                vcol = self.dot_col[si];
                            }
                        }
                    }
                }
                let rain = self.rain[idx];
                self.out[idx] = if bits != 0 && vmax >= rain.i * 0.9 {
                    Out { ch: char::from_u32(0x2800 + bits).unwrap_or(' '), fg: vcol.scale(levels(0.25 + 0.75 * vmax)) }
                } else if rain.i > 0.02 {
                    Out { ch: rain.ch, fg: rain.fg.scale(levels(rain.i)) }
                } else {
                    Out::BLANK
                };
            }
        }
    }

    /// Plain-text rendering of the composed frame (golden tests, previews).
    pub fn to_text(&self) -> String {
        let mut s = String::with_capacity(self.cols * self.rows * 3);
        for row in 0..self.rows {
            for col in 0..self.cols {
                s.push(self.out[row * self.cols + col].ch);
            }
            s.push('\n');
        }
        s
    }

    /// Coloured HTML rendering of the composed frame (README screenshots, visual checks).
    pub fn to_html(&self) -> String {
        let mut s = String::from(
            "<!doctype html><meta charset=utf-8><style>body{margin:0;background:#000}\
             pre{margin:0;padding:8px;font:16px/1.0 'DejaVu Sans Mono',monospace;color:#39ff6a}</style><pre>",
        );
        for row in 0..self.rows {
            let mut cur: Option<Rgb> = None;
            for col in 0..self.cols {
                let o = self.out[row * self.cols + col];
                if o.ch != ' ' && cur != Some(o.fg) {
                    if cur.is_some() {
                        s.push_str("</span>");
                    }
                    let _ = write!(s, "<span style=\"color:#{:02x}{:02x}{:02x}\">", o.fg.0, o.fg.1, o.fg.2);
                    cur = Some(o.fg);
                }
                match o.ch {
                    '<' => s.push_str("&lt;"),
                    '>' => s.push_str("&gt;"),
                    '&' => s.push_str("&amp;"),
                    c => s.push(c),
                }
            }
            if cur.is_some() {
                s.push_str("</span>");
            }
            s.push('\n');
        }
        s.push_str("</pre>");
        s
    }

    /// Glitch: shift a band of rows sideways.
    pub fn shift_rows(&mut self, from: usize, to: usize, by: i32) {
        for row in from..to.min(self.rows) {
            let line: Vec<Out> = self.out[row * self.cols..(row + 1) * self.cols].to_vec();
            for col in 0..self.cols {
                let src = col as i32 - by;
                self.out[row * self.cols + col] =
                    if src >= 0 && (src as usize) < self.cols { line[src as usize] } else { Out::BLANK };
            }
        }
    }
}

/// Brightness in a few steps: colours then change rarely between frames, which
/// keeps the diff (and the terminal's work) small.
#[inline]
fn levels(x: f32) -> f32 {
    ((x * 8.0).round() / 8.0).clamp(0.0, 1.0)
}

#[inline]
fn braille_bit(dx: usize, dy: usize) -> u32 {
    match (dx, dy) {
        (0, 3) => 0x40,
        (1, 3) => 0x80,
        (0, y) => 1 << y,
        (_, y) => 1 << (y + 3),
    }
}

/// Writes only the cells that changed since the previous frame, as one
/// synchronized update.
pub struct Screen {
    prev: Vec<Out>,
    cols: usize,
    rows: usize,
    buf: String,
}

impl Screen {
    pub fn new() -> Screen {
        Screen { prev: Vec::new(), cols: 0, rows: 0, buf: String::new() }
    }

    pub fn render(&mut self, f: &Frame) -> &str {
        self.buf.clear();
        self.buf.push_str("\x1b[?2026h");
        let full = f.cols != self.cols || f.rows != self.rows;
        if full {
            self.buf.push_str("\x1b[0m\x1b[2J");
            self.prev = vec![Out::BLANK; f.out.len()];
            self.cols = f.cols;
            self.rows = f.rows;
        }
        let mut cursor: Option<(usize, usize)> = None;
        let mut color: Option<Rgb> = None;
        for row in 0..f.rows {
            for col in 0..f.cols {
                let idx = row * f.cols + col;
                let o = f.out[idx];
                let p = self.prev[idx];
                if o == p || (o.ch == ' ' && p.ch == ' ') {
                    continue;
                }
                // The last column can trigger a wrap/scroll on some terminals.
                if row == f.rows - 1 && col == f.cols - 1 {
                    continue;
                }
                match cursor {
                    Some((r, c)) if r == row && c == col => {}
                    // Same row, a few cells on: step forward instead of an absolute move.
                    Some((r, c)) if r == row && c < col => {
                        let _ = write!(self.buf, "\x1b[{}C", col - c);
                    }
                    _ => {
                        let _ = write!(self.buf, "\x1b[{};{}H", row + 1, col + 1);
                    }
                }
                if o.ch != ' ' && color != Some(o.fg) {
                    let _ = write!(self.buf, "\x1b[38;2;{};{};{}m", o.fg.0, o.fg.1, o.fg.2);
                    color = Some(o.fg);
                }
                self.buf.push(o.ch);
                cursor = Some((row, col + 1));
                self.prev[idx] = o;
            }
        }
        self.buf.push_str("\x1b[?2026l");
        &self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braille_bits_cover_all_dots() {
        let mut all = 0;
        for dy in 0..4 {
            for dx in 0..2 {
                all |= braille_bit(dx, dy);
            }
        }
        assert_eq!(all, 0xff);
    }

    #[test]
    fn compose_prefers_overlay_then_brighter_layer() {
        let mut f = Frame::new(3, 1);
        f.plot(0, 0, 1.0, Rgb(0, 255, 0));
        f.rain[1] = Glyph { ch: 'ｱ', fg: Rgb(0, 200, 0), i: 0.8 };
        f.plot(2, 0, 0.3, Rgb(0, 255, 0));
        f.rain[1].i = 0.8;
        f.text(2, 0, "X", Rgb(255, 255, 255));
        f.compose();
        assert_eq!(f.out[0].ch, '\u{2801}');
        assert_eq!(f.out[1].ch, 'ｱ');
        assert_eq!(f.out[2].ch, 'X');
    }

    #[test]
    fn screen_diff_writes_nothing_when_unchanged() {
        let mut f = Frame::new(4, 2);
        f.text(0, 0, "ab", Rgb(9, 9, 9));
        f.compose();
        let mut s = Screen::new();
        assert!(s.render(&f).contains("ab"));
        assert_eq!(s.render(&f), "\x1b[?2026h\x1b[?2026l");
    }
}
