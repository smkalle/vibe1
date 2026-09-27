//! The mushroom body learns: the library's one-stream actor-critic, ported from
//! "A fly in the Matrix" web/learner.js (MIT), which states it is the same rule as
//! cadence.plasticity.ActorCritic. tests/learner_cases.txt, recorded from that code,
//! holds this port to it.
//!
//!   act:   on the live (free) state, sample approach/avoid from a softmax over the two
//!          output MBONs; two nudged settles (toward and away from the choice at +-beta)
//!          give every plastic synapse its contrast (a+ (b+ - b-) + (a+ - a-) b-) / 2beta,
//!          added to a trace that decays by gamma*lambda; the critic's trace likewise.
//!   learn: dopamine = clip(reward + gamma V(next) - V(decision)); every plastic efficacy
//!          moves by eta * dopamine * trace (clipped at the cap), the critic by
//!          eta_c * dopamine on its normalised trace; a finished lesson forgets its traces.

use crate::neuro::Brain;

/// The page's lesson configuration (fly-matrix web/page.js LESSONS).
pub const BETA: f64 = 0.1;
pub const TEMPERATURE: f64 = 0.3;
pub const NUDGED_STEPS: usize = 10;
pub const TOLERANCE: f64 = 1e-3;
pub const GAMMA: f64 = 0.95;
pub const LAMBDA: f64 = 0.9;
pub const ETA: f64 = 1.0;
pub const ETA_CRITIC: f64 = 0.05;
pub const CAP: f64 = 3.0;
pub const DOPAMINE_CAP: f64 = 1.0;
pub const OUTPUTS: [&str; 2] = ["mbon:MBON11:right", "mbon:MBON05:left"];
pub const CRITIC: &str = "kc";

/// A decision; every field is checked against the original by the parity test.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Decision {
    /// 0 approach, 1 avoid.
    pub choice: usize,
    pub p_approach: f64,
    pub value: f64,
}

/// What one outcome taught; every field is checked against the original by the parity test.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Lesson {
    pub td_error: f64,
    pub dopamine: f64,
    pub moved: usize,
    pub changed: usize,
    pub mean_abs_change: f64,
}

pub struct Learner {
    outputs: [usize; 2],
    critic: Vec<usize>,
    pub efficacy: Vec<f64>,
    trace: Vec<f64>,
    pub w_critic: Vec<f64>,
    pub b_critic: f64,
    trace_critic: Vec<f64>,
    /// The decision's value, while its outcome is outstanding.
    pending: Option<f64>,
    pub updates: u64,
    /// Outcomes that found no decision to credit.
    pub dropped: u64,
}

impl Learner {
    pub fn new(brain: &Brain) -> Learner {
        let one = |name: &str| brain.sets.get(name).and_then(|v| v.first()).map(|&i| i as usize).expect("output population");
        let critic: Vec<usize> = brain.sets.get(CRITIC).expect("critic population").iter().map(|&i| i as usize).collect();
        let e = brain.seam.efficacy0.clone();
        Learner {
            outputs: [one(OUTPUTS[0]), one(OUTPUTS[1])],
            trace: vec![0.0; e.len()],
            efficacy: e,
            w_critic: vec![0.0; critic.len()],
            b_critic: 0.0,
            trace_critic: vec![0.0; critic.len() + 1],
            critic,
            pending: None,
            updates: 0,
            dropped: 0,
        }
    }

    /// Put this learner's efficacies on the brain's weights.
    pub fn apply(&self, brain: &mut Brain) {
        for (k, &e) in self.efficacy.iter().enumerate() {
            brain.set_efficacy(k, e);
        }
    }

    pub fn probabilities(&self, s: &[f64]) -> [f64; 2] {
        let z = [s[self.outputs[0]] / TEMPERATURE, s[self.outputs[1]] / TEMPERATURE];
        let zmax = z[0].max(z[1]);
        let mut p = [(z[0] - zmax).exp(), (z[1] - zmax).exp()];
        let sum = p[0] + p[1];
        p[0] /= sum;
        p[1] /= sum;
        p
    }

    pub fn value(&self, s: &[f64]) -> f64 {
        let mut v = self.b_critic;
        for (k, &i) in self.critic.iter().enumerate() {
            v += self.w_critic[k] * s[i];
        }
        v
    }

    /// Decide on the live state with uniform draw `u`, and keep the choice's eligibility.
    pub fn act(&mut self, brain: &Brain, u: f64) -> Decision {
        let p = self.probabilities(&brain.s);
        // The library's draw: the count of cumulative sums below the uniform, capped.
        let (mut acc, mut choice) = (0.0, 0usize);
        for pj in p {
            acc += pj;
            if acc < u {
                choice += 1;
            }
        }
        choice = choice.min(1);
        let value = self.value(&brain.s);
        let mut target = [0.0; 2];
        target[choice] = 1.0;
        let plus = brain.settle_nudged(&self.outputs, &target, BETA, TEMPERATURE, NUDGED_STEPS, TOLERANCE);
        let minus = brain.settle_nudged(&self.outputs, &target, -BETA, TEMPERATURE, NUDGED_STEPS, TOLERANCE);
        let (span, decay) = (2.0 * BETA, GAMMA * LAMBDA);
        for k in 0..self.trace.len() {
            let (j, i) = (brain.seam.pre[k] as usize, brain.seam.post[k] as usize);
            let contrast = (plus[j] * (plus[i] - minus[i]) + (plus[j] - minus[j]) * minus[i]) / span;
            self.trace[k] = self.trace[k] * decay + contrast;
        }
        let nc = self.critic.len();
        for k in 0..nc {
            self.trace_critic[k] = self.trace_critic[k] * decay + brain.s[self.critic[k]];
        }
        self.trace_critic[nc] = self.trace_critic[nc] * decay + 1.0;
        self.pending = Some(value);
        Decision { choice, p_approach: p[0], value }
    }

    /// The outcome of the pending decision; `brain`'s live state is the next state.
    pub fn learn(&mut self, brain: &mut Brain, reward: f64, done: bool) -> Option<Lesson> {
        let Some(value) = self.pending.take() else {
            self.dropped += 1;
            return None;
        };
        let next = if done { 0.0 } else { self.value(&brain.s) };
        let td_error = reward + GAMMA * next - value;
        let delta = td_error.clamp(-DOPAMINE_CAP, DOPAMINE_CAP);
        let mut moved = 0;
        for k in 0..self.efficacy.len() {
            let step = ETA * delta * self.trace[k];
            if step == 0.0 {
                continue;
            }
            let eff = (self.efficacy[k] + step).clamp(-CAP, CAP);
            if eff - self.efficacy[k] != 0.0 {
                moved += 1;
                self.efficacy[k] = eff;
                brain.set_efficacy(k, eff);
            }
        }
        let energy: f64 = self.trace_critic.iter().map(|t| t * t).sum();
        let norm = 1.0 / (1.0 + energy);
        let nc = self.critic.len();
        for k in 0..nc {
            self.w_critic[k] += ETA_CRITIC * delta * self.trace_critic[k] * norm;
        }
        self.b_critic += ETA_CRITIC * delta * self.trace_critic[nc] * norm;
        if done {
            self.trace.iter_mut().for_each(|t| *t = 0.0);
            self.trace_critic.iter_mut().for_each(|t| *t = 0.0);
        }
        self.updates += 1;
        let (changed, mean_abs_change) = self.changes(brain);
        Some(Lesson { td_error, dopamine: delta, moved, changed, mean_abs_change })
    }

    /// Plastic synapses off their naive efficacy, and their mean absolute change.
    pub fn changes(&self, brain: &Brain) -> (usize, f64) {
        let (mut changed, mut sum) = (0, 0.0);
        for (e, e0) in self.efficacy.iter().zip(&brain.seam.efficacy0) {
            let d = (e - e0).abs();
            if d > 1e-9 {
                changed += 1;
                sum += d;
            }
        }
        (changed, if changed > 0 { sum / changed as f64 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    /// Ten lessons recorded from the original's learner.js on its brain.js (tools/record_learner.mjs).
    #[test]
    fn parity_with_the_originals_learner() {
        let src = include_str!("../tests/learner_cases.txt");
        let mut brain = Brain::load();
        let mut l = Learner::new(&brain);
        let mut lines = src.lines().filter(|x| !x.starts_with('#'));
        let head: Vec<&str> = lines.next().unwrap().split(' ').collect();
        assert_eq!((head[1].parse::<usize>().unwrap(), head[3].parse::<usize>().unwrap()), (brain.seam.len(), 4130));
        let mut worst = 0.0f64;
        let mut n = 0;
        for line in lines {
            let parts: Vec<&str> = line.split(" | ").collect();
            let a: Vec<&str> = parts[0].split(' ').collect();
            let d: Vec<f64> = parts[1].split(' ').map(|x| x.parse().unwrap()).collect();
            let r: Vec<f64> = parts[2].split(' ').map(|x| x.parse().unwrap()).collect();
            let (fruit, u, reward): (&str, f64, f64) = (a[1], a[2].parse().unwrap(), a[3].parse().unwrap());
            let odour = if fruit == "banana" { "orn:decaying_fruit" } else { "orn:yeasty" };
            brain.clear_stimuli();
            brain.stimulate(&format!("{odour}:left"), 0.8);
            brain.stimulate(&format!("{odour}:right"), 0.8);
            brain.stimulate("haltere:left", 0.5);
            brain.stimulate("haltere:right", 0.5);
            for _ in 0..40 {
                brain.step();
            }
            let dec = l.act(&brain, u);
            for _ in 0..5 {
                brain.step();
            }
            let les = l.learn(&mut brain, reward, true).unwrap();
            assert_eq!(dec.choice, d[1] as usize, "lesson {n}: choice");
            assert_eq!((les.moved, les.changed), (r[2] as usize, r[3] as usize), "lesson {n}: moved/changed");
            let got = [dec.p_approach, dec.value, les.td_error, les.dopamine, les.mean_abs_change, l.efficacy.iter().sum(), l.value(&brain.s), brain.s.iter().sum()];
            let want = [d[0], d[2], r[0], r[1], r[4], r[5], r[6], r[7]];
            for (g, w) in got.iter().zip(want) {
                assert!(close(*g, w), "lesson {n}: {got:?} vs {want:?}");
                worst = worst.max((g - w).abs() / (1.0 + w.abs()));
            }
            n += 1;
        }
        assert_eq!(n, 10);
        eprintln!("learner parity: 10 lessons, worst relative deviation {worst:e}");
    }

    /// The original's gate 4, a T-maze: lessons alternate the two odours; `outcome`
    /// pays each (fruit, choice). Returns the approach probabilities after each lesson.
    fn tmaze(brain: &mut Brain, l: &mut Learner, lessons: usize, seed: &mut u32, outcome: impl Fn(&str, usize) -> f64) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        for t in 0..lessons {
            let fruit = if t % 2 == 0 { "banana" } else { "bread" };
            let odour = if fruit == "banana" { "orn:decaying_fruit" } else { "orn:yeasty" };
            brain.clear_stimuli();
            brain.stimulate(&format!("{odour}:left"), 0.8);
            brain.stimulate(&format!("{odour}:right"), 0.8);
            brain.stimulate("haltere:left", 0.5);
            brain.stimulate("haltere:right", 0.5);
            for _ in 0..40 {
                brain.step();
            }
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = l.act(brain, *seed as f64 / 4294967296.0);
            for _ in 0..5 {
                brain.step();
            }
            l.learn(brain, outcome(fruit, d.choice), true);
            out.push((fruit.to_string(), d.p_approach));
        }
        out
    }

    fn last_p(log: &[(String, f64)], fruit: &str) -> f64 {
        log.iter().rev().find(|(f, _)| f == fruit).map(|(_, p)| *p).unwrap()
    }

    /// Train sugar on the banana until the fly prefers it; returns the learner and lessons taken.
    fn learn_banana(seed: &mut u32) -> (Brain, Learner, usize, f64, f64) {
        let mut brain = Brain::load();
        let mut l = Learner::new(&brain);
        let naive = tmaze(&mut brain, &mut l, 2, seed, |_, _| 0.0);
        let b0 = last_p(&naive, "banana");
        let sugar_on_banana = |f: &str, c: usize| if c == 0 && f == "banana" { 1.0 } else { 0.0 };
        let mut n = 0;
        let mut p = b0;
        while n < 200 && !(p > 0.5 && p > b0 + 0.15) {
            p = last_p(&tmaze(&mut brain, &mut l, 2, seed, sugar_on_banana), "banana");
            n += 2;
        }
        (brain, l, n, b0, p)
    }

    #[test]
    fn decision_and_lesson_cost() {
        let mut brain = Brain::load();
        let mut l = Learner::new(&brain);
        brain.stimulate("orn:yeasty:left", 0.8);
        brain.stimulate("orn:yeasty:right", 0.8);
        for _ in 0..40 {
            brain.step();
        }
        let t = std::time::Instant::now();
        let _ = l.act(&brain, 0.3);
        let act = t.elapsed();
        let t = std::time::Instant::now();
        l.learn(&mut brain, 1.0, true);
        eprintln!("decision (act, 2 nudged settles) {:.1} ms, lesson (learn) {:.2} ms", act.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3);
    }

    #[test]
    fn the_fly_learns_where_the_sugar_is() {
        let (_, _, n, b0, b1) = learn_banana(&mut 11);
        eprintln!("T-maze acquisition: banana {b0:.2} -> {b1:.2} in {n} lessons (sugar on the banana)");
        assert!(n < 200 && b1 > 0.5, "never learned the banana's sugar");
    }

    #[test]
    fn when_the_sugar_moves_the_fly_relearns() {
        // Extinction: the banana stops paying (the screensaver's usual case).
        let (mut brain, mut l, _, _, peak) = learn_banana(&mut 11);
        let (mut n, mut p) = (0, peak);
        while n < 300 && p > peak - 0.15 {
            p = last_p(&tmaze(&mut brain, &mut l, 2, &mut 11, |f, c| if c == 0 && f == "bread" { 1.0 } else { 0.0 }), "banana");
            n += 2;
        }
        eprintln!("reversal by extinction: banana {peak:.2} -> {p:.2} in {n} lessons (sugar moved to the bread)");
        assert!(n < 300, "the banana preference never faded");
        // The original's reversal: sugar moves and the old fruit is punished (a blow there).
        let (mut brain, mut l, _, _, peak) = learn_banana(&mut 11);
        let (mut n, mut p) = (0, peak);
        let rule = |f: &str, c: usize| match (f, c) {
            ("bread", 0) => 1.0,
            ("banana", 0) => -1.0,
            _ => 0.0,
        };
        while n < 300 && p > 0.4 {
            p = last_p(&tmaze(&mut brain, &mut l, 2, &mut 11, rule), "banana");
            n += 2;
        }
        eprintln!("reversal with blows at the banana: banana {peak:.2} -> {p:.2} in {n} lessons");
        assert!(n < 300 && p <= 0.4, "the banana preference did not reverse");
    }

}
