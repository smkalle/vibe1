//! The live brain: the Cadence rate model on the 60,000-neuron BANC sub-net that
//! "A fly in the Matrix" settles, ported from its `web/brain.js` (MIT) with the
//! library's arithmetic in the library's order, so tests/parity_cases.txt (made by
//! cadence.Brain) holds it to 1e-9.
//!
//!   total_i = sum over synapses e into i of s[pre_e] * w_e
//!   v_i    += dt * (total_i + (drive_i + bias_i) - v_i)
//!   s_i     = rectified sigmoid(v_i), exactly zero at rest
//!
//! The sums are computed by pushing from active senders rather than pulling over
//! every synapse. Senders are stored sorted within each row, so visiting senders
//! in ascending order adds each neuron's terms in exactly the row's order, and a
//! silent sender only ever contributes an exact zero: the result is bit-identical
//! to the library's, at a cost proportional to the synapses of active neurons.

use crate::math::{v3, V3};
use std::collections::HashMap;

static ASSET: &[u8] = include_bytes!("../assets/brain.bin");

pub struct Brain {
    pub n: usize,
    pub edges: usize,
    dt: f64,
    slope: f64,
    threshold: f64,
    amplitude: f64,
    rest: f64,
    rest_scale: f64,
    /// Synapses by sender (ascending), each with its receiver and weight.
    out_ptr: Vec<u32>,
    out_post: Vec<u16>,
    out_w: Vec<f64>,
    bias: Vec<f64>,
    pub v: Vec<f64>,
    pub s: Vec<f64>,
    drive: Vec<f64>,
    total: Vec<f64>,
    pub steps: u64,
    /// Soma position in the atlas frame ([-1, 1], y up the body axis).
    pub pos: Vec<V3>,
    pub sets: HashMap<String, Vec<u32>>,
    /// FNV-1a of the rebuilt weights and the value the asset was baked with.
    pub weights_fnv: (u64, u64),
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, k: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.i..self.i + k)?;
        self.i += k;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }
}

fn fnv1a64(ws: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in ws {
        for byte in w.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

impl Brain {
    pub fn load() -> Brain {
        Brain::parse(ASSET).expect("embedded brain asset is malformed")
    }

    pub fn parse(b: &[u8]) -> Option<Brain> {
        let mut r = Reader { b, i: 0 };
        if r.take(4)? != b"FLYB" || r.u32()? != 1 {
            return None;
        }
        let n = r.u32()? as usize;
        let edges = r.u32()? as usize;
        let (dt, slope, threshold, gain, amplitude, leak) = (r.f64()?, r.f64()?, r.f64()?, r.f64()?, r.f64()?, r.f64()?);
        if leak != 0.0 || n > u16::MAX as usize + 1 {
            return None; // the leaky variant and nets past u16 senders are not needed here
        }
        let fnv_baked = r.u64()?;

        let k = r.u8()? as usize;
        let classes: Vec<f64> = (0..k).map(|_| r.f64()).collect::<Option<_>>()?;
        let log_gain: Vec<f64> = r.take(n)?.iter().map(|c| classes.get(*c as usize).copied()).collect::<Option<_>>()?;

        let mut bias = vec![0.0; n];
        for _ in 0..r.u32()? {
            let i = r.u32()? as usize;
            *bias.get_mut(i)? = r.f64()?;
        }

        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0u32);
        for _ in 0..n {
            let len = r.varint()? as u32;
            row_ptr.push(row_ptr.last()? + len);
        }
        if *row_ptr.last()? as usize != edges {
            return None;
        }
        let mut pre = Vec::with_capacity(edges);
        for i in 0..n {
            let mut last = 0u64;
            for _ in row_ptr[i]..row_ptr[i + 1] {
                last += r.varint()?;
                pre.push(u16::try_from(last).ok()?);
            }
        }
        let count: Vec<u64> = (0..edges).map(|_| r.varint()).collect::<Option<_>>()?;
        let packed = r.take(edges.div_ceil(4))?;
        let mut efficacy: Vec<f64> = (0..edges)
            .map(|e| match (packed[e >> 2] >> (2 * (e & 3))) & 3 {
                0 => 0.0,
                1 => 1.0,
                _ => -1.0,
            })
            .collect();
        for _ in 0..r.u32()? {
            let e = r.u32()? as usize;
            *efficacy.get_mut(e)? = r.f64()?;
        }
        // The library's composition: (gain * count * exp(log_gain[pre])) * efficacy.
        let w: Vec<f64> =
            (0..edges).map(|e| (gain * count[e] as f64 * log_gain[pre[e] as usize].exp()) * efficacy[e]).collect();
        let fnv = fnv1a64(&w);
        // Transpose to sender order; a stable counting sort keeps receivers ascending.
        let mut out_ptr = vec![0u32; n + 1];
        for &p in &pre {
            out_ptr[p as usize + 1] += 1;
        }
        for i in 0..n {
            out_ptr[i + 1] += out_ptr[i];
        }
        let mut fill: Vec<u32> = out_ptr[..n].to_vec();
        let (mut out_post, mut out_w) = (vec![0u16; edges], vec![0.0f64; edges]);
        for i in 0..n {
            for e in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                let k = &mut fill[pre[e] as usize];
                out_post[*k as usize] = i as u16;
                out_w[*k as usize] = w[e];
                *k += 1;
            }
        }

        let mut pos = Vec::with_capacity(n);
        let q = |v: u16| v as f32 / 65535.0 * 2.0 - 1.0;
        for _ in 0..n {
            let (x, y, z) = (r.u16()?, r.u16()?, r.u16()?);
            pos.push(v3(q(x), q(y), q(z)));
            r.u8()?; // atlas region: kept in the asset, not needed to draw
        }

        let mut sets = HashMap::new();
        for _ in 0..r.u32()? {
            let len = r.u8()? as usize;
            let name = String::from_utf8_lossy(r.take(len)?).into_owned();
            let c = r.varint()? as usize;
            let mut idx = Vec::with_capacity(c);
            let mut last = 0u64;
            for _ in 0..c {
                last += r.varint()?;
                if last as usize >= n {
                    return None;
                }
                idx.push(last as u32);
            }
            sets.insert(name, idx);
        }

        let rest = 1.0 / (1.0 + (slope * threshold).exp());
        Some(Brain {
            n,
            edges,
            dt,
            slope,
            threshold,
            amplitude,
            rest,
            rest_scale: 1.0 / (1.0 - rest),
            out_ptr,
            out_post,
            out_w,
            bias,
            v: vec![0.0; n],
            s: vec![0.0; n],
            drive: vec![0.0; n],
            total: vec![0.0; n],
            steps: 0,
            pos,
            sets,
            weights_fnv: (fnv, fnv_baked),
        })
    }

    #[cfg(test)]
    pub fn reset(&mut self) {
        self.v.iter_mut().for_each(|x| *x = 0.0);
        self.s.iter_mut().for_each(|x| *x = 0.0);
        self.steps = 0;
    }

    pub fn clear_stimuli(&mut self) {
        self.drive.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Drive a named population at `level` in [0, 1]; overlapping drives combine by max.
    pub fn stimulate(&mut self, name: &str, level: f64) {
        let Some(idx) = self.sets.get(name) else { return };
        let d = self.amplitude * level;
        for &i in idx {
            let slot = &mut self.drive[i as usize];
            if d > *slot {
                *slot = d;
            }
        }
    }

    #[inline]
    fn activation(&self, v: f64) -> f64 {
        let mut r = ((-self.slope) * (v - self.threshold)).exp();
        r += 1.0;
        r = 1.0 / r;
        r -= self.rest;
        if r < 0.0 {
            r = 0.0;
        }
        r * self.rest_scale
    }

    /// One step of the neuron model under the current drive.
    pub fn step(&mut self) {
        self.total.iter_mut().for_each(|t| *t = 0.0);
        for j in 0..self.n {
            let sj = self.s[j];
            if sj == 0.0 {
                continue; // adds exact zeros only
            }
            let (a, b) = (self.out_ptr[j] as usize, self.out_ptr[j + 1] as usize);
            for (post, w) in self.out_post[a..b].iter().zip(&self.out_w[a..b]) {
                self.total[*post as usize] += sj * w;
            }
        }
        for i in 0..self.n {
            let mut t = self.total[i];
            t += self.drive[i] + self.bias[i];
            t -= self.v[i];
            t *= self.dt;
            self.v[i] += t;
            self.s[i] = self.activation(self.v[i]);
        }
        self.steps += 1;
    }

    pub fn mean(&self, name: &str) -> f64 {
        match self.sets.get(name) {
            Some(idx) if !idx.is_empty() => idx.iter().map(|&i| self.s[i as usize]).sum::<f64>() / idx.len() as f64,
            _ => 0.0,
        }
    }

    pub fn active_count(&self, level: f64) -> usize {
        self.s.iter().filter(|x| **x >= level).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_decodes_and_weights_rebuild_exactly() {
        let b = Brain::load();
        assert_eq!((b.n, b.edges), (60_000, 1_209_528));
        assert_eq!(b.weights_fnv.0, b.weights_fnv.1, "rebuilt weights differ from the payload's");
        assert_eq!(b.sets.len(), 321);
        for name in ["haltere:left", "orn:decaying_fruit:left", "grn:sugar:labellum", "vis:LPLC2", "mbon:MBON11:right", "kc"] {
            assert!(!b.sets[name].is_empty(), "{name}");
        }
    }

    #[test]
    fn step_cost() {
        let mut b = Brain::load();
        b.stimulate("haltere:left", 0.5);
        b.stimulate("orn:decaying_fruit:left", 0.8);
        let t = std::time::Instant::now();
        for _ in 0..50 {
            b.step();
        }
        eprintln!("step: {:.2} ms, {} active", t.elapsed().as_secs_f64() * 1e3 / 50.0, b.active_count(0.5));
    }

    #[test]
    fn rest_is_an_exact_fixed_point() {
        let mut b = Brain::load();
        for _ in 0..20 {
            b.step();
        }
        // Two readout neurons carry a calibrated bias; nothing else moves without input.
        assert!(b.active_count(0.5) <= 2, "{} active at rest", b.active_count(0.5));
    }

    /// The Cadence library's own numbers (tests/parity_cases.txt, from cadence.Brain).
    #[test]
    fn parity_with_the_cadence_library() {
        let src = include_str!("../tests/parity_cases.txt");
        let mut b = Brain::load();
        let (mut worst, mut cases) = (0.0f64, 0);
        let mut lines = src.lines().filter(|l| !l.starts_with('#'));
        while let Some(line) = lines.next() {
            let name = line.strip_prefix("case ").expect("case line");
            b.reset();
            b.clear_stimuli();
            let (mut readouts, mut final_active, mut t) = (Vec::new(), 0usize, 0);
            for line in lines.by_ref() {
                let mut it = line.split(' ');
                match it.next().unwrap() {
                    "stim" => {
                        let pop = it.next().unwrap();
                        b.stimulate(pop, it.next().unwrap().parse().unwrap());
                    }
                    "readouts" => readouts = it.map(str::to_string).collect(),
                    "final_active" => final_active = it.next().unwrap().parse().unwrap(),
                    "step" => {
                        b.step();
                        t += 1;
                        for (r, want) in readouts.iter().zip(it) {
                            let d = (b.mean(r) - want.parse::<f64>().unwrap()).abs();
                            worst = worst.max(d);
                        }
                    }
                    "end" => break,
                    other => panic!("unexpected line {other}"),
                }
            }
            assert!(t > 0, "{name}: no steps");
            assert_eq!(b.active_count(0.5), final_active, "{name}: active count differs from the library");
            cases += 1;
        }
        assert_eq!(cases, 6);
        eprintln!("parity: 6 cases, worst deviation {worst:e}");
        assert!(worst < 1e-9, "parity failed: worst deviation {worst:e}");
    }
}
