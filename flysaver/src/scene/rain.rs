//! The digital rain: columns of half-width katakana falling behind everything.

use crate::fb::{Frame, Glyph};
use crate::rng::Rng;
use crate::theme::Theme;

const KATAKANA: &str = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789Z:・=*+-<>";
const ASCII: &str = "abcdefghijklmnopqrstuvwxyz0123456789Z:=*+-<>";
/// The original textures its rain at 0.55 opacity; the trail never outshines the scene.
const TRAIL: f32 = 0.42;
/// Brighter trail and a full-strength head when colours are vivid.
const VIVID_TRAIL: f32 = 0.65;

struct Column {
    active: bool,
    head: f32,
    speed: f32,
    len: f32,
    wait: f32,
}

pub struct Rain {
    glyphs: Vec<char>,
    cols: Vec<Column>,
    cells: Vec<char>,
    width: usize,
    height: usize,
    density: f32,
    tick: f32,
    pub frozen: f32,
    vivid: bool,
}

impl Rain {
    pub fn new(density: f32, style: &str, vivid: bool) -> Rain {
        let set = if style == "ascii" { ASCII } else { KATAKANA };
        Rain {
            glyphs: set.chars().collect(),
            cols: Vec::new(),
            cells: Vec::new(),
            width: 0,
            height: 0,
            density,
            tick: 0.0,
            frozen: 0.0,
            vivid,
        }
    }

    fn pick(&self, rng: &mut Rng) -> char {
        self.glyphs[rng.below(self.glyphs.len())]
    }

    fn spawn(&self, rng: &mut Rng, h: usize, anywhere: bool) -> Column {
        let active = rng.chance(self.density);
        let len = rng.range(8.0, 30.0);
        Column {
            active,
            head: if anywhere { rng.range(0.0, h as f32 + len) } else { -rng.range(0.0, h as f32 * 0.5) },
            speed: rng.range(5.0, 16.0),
            len,
            wait: if active { 0.0 } else { rng.range(0.5, 6.0) },
        }
    }

    fn resize(&mut self, w: usize, h: usize, rng: &mut Rng) {
        self.width = w;
        self.height = h;
        self.cols = (0..w).map(|_| self.spawn(rng, h, true)).collect();
        self.cells = (0..w * h).map(|_| self.pick(rng)).collect();
    }

    pub fn step(&mut self, dt: f32, w: usize, h: usize, rng: &mut Rng) {
        if w != self.width || h != self.height {
            self.resize(w, h, rng);
        }
        if self.frozen > 0.0 {
            self.frozen -= dt;
            return;
        }
        for i in 0..self.cols.len() {
            let c = &mut self.cols[i];
            if !c.active {
                c.wait -= dt;
                if c.wait <= 0.0 {
                    self.cols[i] = self.spawn(rng, h, false);
                    if !self.cols[i].active {
                        self.cols[i].wait = rng.range(0.5, 6.0);
                    }
                }
                continue;
            }
            c.head += c.speed * dt;
            if c.head - c.len > h as f32 {
                self.cols[i] = self.spawn(rng, h, false);
            }
        }
        // Glyphs flicker about ten times a second, like the original's texture.
        self.tick += dt;
        while self.tick >= 0.1 {
            self.tick -= 0.1;
            for _ in 0..(w * h / 30).max(1) {
                let k = rng.below(self.cells.len());
                self.cells[k] = self.pick(rng);
            }
        }
    }

    pub fn draw(&self, f: &mut Frame, theme: &Theme) {
        if self.width != f.cols || self.height != f.rows {
            return;
        }
        for (x, c) in self.cols.iter().enumerate() {
            if !c.active {
                continue;
            }
            let head = c.head.floor() as i32;
            let tail = (c.head - c.len).floor() as i32;
            for y in tail.max(0)..=head.min(self.height as i32 - 1) {
                let idx = y as usize * self.width + x;
                if f.rain_mask[idx] {
                    continue;
                }
                let t = (head - y) as f32 / c.len;
                f.rain[idx] = if y == head {
                    Glyph { ch: self.cells[idx], fg: theme.rain_head, i: if self.vivid { 1.0 } else { 0.8 } }
                } else {
                    let trail = if self.vivid { VIVID_TRAIL } else { TRAIL };
                    Glyph { ch: self.cells[idx], fg: theme.rain, i: trail * (1.0 - t).max(0.0).powf(1.6) }
                };
            }
        }
    }
}
