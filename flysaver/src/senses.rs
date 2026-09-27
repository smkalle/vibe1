//! The senses: what the room and the fly's own motion do to its afferents. A port of
//! "A fly in the Matrix" (web/senses.js and Life.senses in web/life.js). The
//! looming detectors see the swatter (sim::Threat), as the original's see the
//! visitor's hand: the rate of its angular expansion, with a short memory.

use crate::math::{Basis, V3};
use crate::sim::{Fly, BANANA, BREAD};

const HALTERE_TONE: f64 = 0.5;
const OCELLUS_ELEVATION: f32 = 45.0 * std::f32::consts::PI / 180.0;
const OCELLUS_AZIMUTH: f32 = 35.0 * std::f32::consts::PI / 180.0;
const AIRSPEED_SATURATION: f64 = 1.0;
const FLOW_REST: f64 = 0.5;
const FLOW_SATURATION: f64 = 20.0;
const ODOUR_SIGMA: f32 = 0.5;
const ODOUR_CORE: f32 = 0.1;
const ODOUR_CORE_WEIGHT: f32 = 0.85;
const ODOUR_BASELINE: f32 = 0.05;
const ODOUR_COMPARISON: f64 = 1.5;
/// Angular expansion (rad/s) that saturates the looming drive, and its memory per step.
const LOOM_SATURATION: f32 = 4.0;
const LOOM_MEMORY: f64 = 0.7;

/// Body rates in the original's convention: roll (right wing down +), pitch (nose down +), yaw (left turn +).
#[derive(Clone, Copy, Debug, Default)]
pub struct Rates {
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
}

/// Tracks the fly's angles between steps to give body rates, and the looming threat.
#[derive(Default)]
pub struct Motion {
    last: Option<(f32, f32, f32)>,
    last_theta: Option<f32>,
    pub loom: f64,
}

impl Motion {
    pub fn rates(&mut self, fly: &Fly, dt: f32) -> Rates {
        let now = (fly.yaw, fly.pitch, fly.roll);
        let r = match self.last {
            Some((y, p, r)) if dt > 0.0 => Rates {
                // Our yaw grows turning right and pitch grows nose up; the original's are the reverse.
                yaw: -(crate::math::angle_diff(y, now.0) / dt) as f64,
                pitch: -((now.1 - p) / dt) as f64,
                roll: ((now.2 - r) / dt) as f64,
            },
            _ => Rates::default(),
        };
        self.last = Some(now);
        r
    }

    /// Looming drive from a threat of radius `radius` at `at` (None: nothing looms).
    pub fn looming(&mut self, fly: &Fly, at: Option<V3>, radius: f32, dt: f32) -> f64 {
        let expansion = match at {
            Some(p) => {
                let d = (p - fly.pos).len().max(0.005);
                let theta = 2.0 * (radius / d).atan();
                let rate = match self.last_theta {
                    Some(last) if dt > 0.0 => ((theta - last) / dt).max(0.0) / LOOM_SATURATION,
                    _ => 0.0,
                };
                self.last_theta = Some(theta);
                rate as f64
            }
            None => {
                self.last_theta = None;
                0.0
            }
        };
        self.loom = (LOOM_MEMORY * self.loom + expansion).min(1.0);
        self.loom
    }
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// How much each ocellus (up 45 deg, out 35 deg) faces the sky: [left, right].
fn ocelli(b: &Basis) -> [f64; 2] {
    let (ce, se) = (OCELLUS_ELEVATION.cos(), OCELLUS_ELEVATION.sin());
    let (ca, sa) = (OCELLUS_AZIMUTH.cos(), OCELLUS_AZIMUTH.sin());
    let left = -b.r;
    [1.0f32, -1.0].map(|side| {
        let dir: V3 = b.f * (ce * ca) + left * (side * ce * sa) + b.u * se;
        clamp01(dir.y as f64)
    })
}

fn optic_flow(r: Rates, left: bool) -> f64 {
    let sign = if left { 1.0 } else { -1.0 };
    let hs = sign * r.yaw;
    let vs = sign * r.roll - r.pitch;
    clamp01(FLOW_REST + 0.5 * (hs + vs) / FLOW_SATURATION)
}

/// Concentration of a fruit's smell: a narrow core in a faint wide plume.
fn plume(p: V3, source: V3) -> f32 {
    let d2 = {
        let d = p - source;
        d.dot(d)
    };
    ODOUR_CORE_WEIGHT * (-d2 / (2.0 * ODOUR_CORE * ODOUR_CORE)).exp()
        + (1.0 - ODOUR_CORE_WEIGHT) * (-d2 / (2.0 * ODOUR_SIGMA * ODOUR_SIGMA)).exp()
}

/// Receptor drive [left, right]: concentration times hunger, sharpened by the bilateral comparison.
fn odour(fly: &Fly, b: &Basis, source: V3) -> [f64; 2] {
    let left = -b.r;
    let c = plume(fly.pos, source) as f64;
    let cl = plume(fly.pos + left * ODOUR_BASELINE, source) as f64;
    let cr = plume(fly.pos - left * ODOUR_BASELINE, source) as f64;
    let ratio = (cl + 1e-9).ln() - (cr + 1e-9).ln();
    let gain = 0.5 + fly.hunger as f64;
    [
        (gain * c * (1.0 + ODOUR_COMPARISON * ratio.max(0.0))).min(1.0),
        (gain * c * (1.0 + ODOUR_COMPARISON * (-ratio).max(0.0))).min(1.0),
    ]
}

/// Every channel's level in [0, 1], by population name. `loom` comes from Motion::looming.
pub fn sense(fly: &Fly, rates: Rates, loom: f64) -> Vec<(&'static str, f64)> {
    let b = Basis::from_euler(fly.yaw, fly.pitch, fly.roll);
    let flying = fly.airborne();
    let tone = if flying { HALTERE_TONE } else { 0.0 };
    let [ol, or] = ocelli(&b);
    let air = if flying { clamp01(fly.speed as f64 / AIRSPEED_SATURATION) } else { 0.0 };
    let (fl, fr) = (optic_flow(rates, true), optic_flow(rates, false));
    let [bl, br] = odour(fly, &b, BANANA);
    let [yl, yr] = odour(fly, &b, BREAD);
    // Taste on contact: the labellum on the sugared fruit, and bitter on labellum and
    // front legs when it is laced. Sugar is not put on the legs: in this model leg sugar
    // silences MN9 (specs/flysaver-taste.md).
    let (sugar, bitter) = fly.tasting().map_or((0.0, 0.0), |(s, b)| (s as f64, b as f64));
    vec![
        ("haltere:left", tone),
        ("haltere:right", tone),
        ("ocelli:left", ol),
        ("ocelli:right", or),
        ("lptc:hs:left", fl),
        ("lptc:vs:left", fl),
        ("lptc:hs:right", fr),
        ("lptc:vs:right", fr),
        ("jo:C:left", air),
        ("jo:C:right", air),
        ("jo:E:left", air),
        ("jo:E:right", air),
        ("vis:LC4", loom),
        ("vis:LPLC2", loom),
        ("orn:decaying_fruit:left", bl),
        ("orn:decaying_fruit:right", br),
        ("orn:yeasty:left", yl),
        ("orn:yeasty:right", yr),
        ("grn:sugar:labellum", sugar),
        ("grn:bitter:labellum", bitter),
        ("grn:bitter:front_leg", bitter),
        ("leg_touch", if flying { 0.0 } else { 0.6 }),
        ("dan:pam", if fly.feeding { 0.8 } else { 0.0 }),
        // Punishment dopamine: a blow, or a bitter taste (supplied, as sugar's PAM drive is).
        ("dan:ppl1", if fly.punish > 0.0 { 0.8 } else { 0.8 * bitter }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::sim::Mode;

    fn level(s: &[(&str, f64)], name: &str) -> f64 {
        s.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap()
    }

    fn fly() -> Fly {
        let mut f = Fly::new(&mut Rng::new(1));
        (f.yaw, f.pitch, f.roll) = (0.0, 0.0, 0.0);
        f
    }

    #[test]
    fn flight_tone_and_airspeed_only_in_flight() {
        let mut f = fly();
        let s = sense(&f, Rates::default(), 0.0);
        assert_eq!(level(&s, "haltere:left"), 0.5);
        assert!(level(&s, "jo:C:left") > 0.1);
        assert_eq!(level(&s, "leg_touch"), 0.0);
        f.mode = Mode::Sitting;
        let s = sense(&f, Rates::default(), 0.0);
        assert_eq!((level(&s, "haltere:left"), level(&s, "jo:C:left"), level(&s, "leg_touch")), (0.0, 0.0, 0.6));
    }

    #[test]
    fn level_flight_ocelli_see_the_sky_equally() {
        let s = sense(&fly(), Rates::default(), 0.0);
        let (l, r) = (level(&s, "ocelli:left"), level(&s, "ocelli:right"));
        assert!((l - r).abs() < 1e-6 && l > 0.5, "{l} {r}");
    }

    #[test]
    fn a_turn_drives_the_optic_flow_cells_apart() {
        let s = sense(&fly(), Rates { yaw: 10.0, ..Rates::default() }, 0.0);
        assert!(level(&s, "lptc:hs:left") > 0.7 && level(&s, "lptc:hs:right") < 0.3);
        let still = sense(&fly(), Rates::default(), 0.0);
        assert_eq!(level(&still, "lptc:hs:left"), 0.5);
    }

    #[test]
    fn banana_smells_like_decaying_fruit_up_close() {
        let mut f = fly();
        f.pos = BANANA + crate::math::v3(0.05, 0.1, 0.0);
        let s = sense(&f, Rates::default(), 0.0);
        assert!(level(&s, "orn:decaying_fruit:left") > 0.3);
        assert!(level(&s, "orn:decaying_fruit:left") > level(&s, "orn:yeasty:left"));
        f.pos = crate::math::v3(0.3, 1.9, 0.3);
        let far = sense(&f, Rates::default(), 0.0);
        assert!(level(&far, "orn:decaying_fruit:left") < 0.1);
    }

    #[test]
    fn feeding_tastes_sugar_and_a_swatter_looms() {
        let mut f = fly();
        f.perch(crate::sim::Spot::Bread);
        f.sugar = Some(crate::sim::Spot::Bread);
        assert_eq!(level(&sense(&f, Rates::default(), 0.0), "grn:sugar:labellum"), 1.0);
        assert_eq!(level(&sense(&f, Rates::default(), 0.0), "grn:bitter:labellum"), 0.0);
        f.bitter = 0.6;
        let s = sense(&f, Rates::default(), 0.0);
        assert_eq!((level(&s, "grn:bitter:front_leg"), level(&s, "dan:ppl1")), (0.6000000238418579, 0.8 * 0.6000000238418579));
        f.sugar = Some(crate::sim::Spot::Banana);
        assert_eq!(level(&sense(&f, Rates::default(), 0.0), "grn:sugar:labellum"), 0.0);
        // An approaching swatter looms; nothing looms once it is gone.
        let f = fly();
        let mut m = Motion::default();
        let mut loom = 0.0;
        for k in 0..20 {
            let at = f.pos + crate::math::v3(0.6 - 0.025 * k as f32, 0.0, 0.0);
            loom = m.looming(&f, Some(at), crate::sim::SWATTER_R, 1.0 / 30.0);
        }
        assert!(loom > 0.4, "{loom}");
        assert_eq!(level(&sense(&f, Rates::default(), loom), "vis:LPLC2"), loom);
        for _ in 0..30 {
            loom = m.looming(&f, None, crate::sim::SWATTER_R, 1.0 / 30.0);
        }
        assert!(loom < 0.01);
    }
}

/// The senses through the measured wiring: the model's own responses, not our code's.
#[cfg(test)]
mod through_the_wiring {
    use super::*;
    use crate::neuro::Brain;
    use crate::rng::Rng;

    fn settle(fly: &Fly, rates: Rates, steps: usize, b: &mut Brain) {
        b.clear_stimuli();
        for (n, l) in sense(fly, rates, 0.0) {
            b.stimulate(n, l);
        }
        for _ in 0..steps {
            b.step();
        }
    }

    fn level_fly() -> Fly {
        let mut f = Fly::new(&mut Rng::new(1));
        (f.yaw, f.pitch, f.roll) = (0.0, 0.0, 0.0);
        f.pos = crate::math::v3(0.4, 1.8, 0.4); // far from both fruits
        f
    }

    #[test]
    fn a_turn_reaches_the_wing_steering_motor_neurons() {
        let f = level_fly();
        let mut level = Brain::load();
        settle(&f, Rates::default(), 60, &mut level);
        let mut turn = Brain::load();
        settle(&f, Rates::default(), 50, &mut turn);
        settle(&f, Rates { yaw: 10.0, ..Rates::default() }, 10, &mut turn);
        // Through the optic lobes' HS/VS cells and on to the wing's steering muscles.
        let moved: f64 = ["mn:wing:iv4:right", "mn:wing:b1:left", "mn:wing:b3:right", "neck_mn:right"]
            .iter()
            .map(|k| (turn.mean(k) - level.mean(k)).abs())
            .sum();
        assert!(moved > 0.05, "steering motor neurons moved only {moved}");
        assert!(turn.active_count(0.5) > level.active_count(0.5));
    }

    #[test]
    fn a_smell_runs_the_olfactory_pathway_to_the_mushroom_body() {
        let far = level_fly();
        let mut near = far.clone();
        near.pos = crate::sim::BANANA + crate::math::v3(0.0, 0.08, 0.0);
        let (mut a, mut b) = (Brain::load(), Brain::load());
        settle(&far, Rates::default(), 60, &mut a);
        settle(&near, Rates::default(), 60, &mut b);
        assert!(b.mean("pn") > 0.1, "projection neurons {}", b.mean("pn"));
        // Kenyon cells: a sparse code, as the original's receipts require.
        let kc = b.mean("kc");
        assert!(kc > 0.001 && kc < 0.1, "kenyon cells {kc}");
        assert!(b.mean("mbon:MBON11:right") > a.mean("mbon:MBON11:right") + 0.05);
    }
}

/// The proboscis extension reflex through the measured wiring (specs/flysaver-taste.md §2).
/// Without sugar MN9 is exactly 0. Tasting sugar, it peaks about 1.3 s after landing: to
/// 0.3-0.48 on a brain's very first taste, and to 0.02-0.3 on every taste after it (the
/// first taste moves the network to a lasting state). Bitter suppresses the peak.
#[cfg(test)]
pub mod taste {
    use super::*;
    use crate::math::{v3, V3};
    use crate::neuro::Brain;
    use crate::rng::Rng;
    use crate::sim::{Mode, Spot};

    fn apply(b: &mut Brain, f: &Fly) {
        b.clear_stimuli();
        for (n, l) in sense(f, Rates::default(), 0.0) {
            b.stimulate(n, l);
        }
    }

    fn fly_at(b: &mut Brain, f: &mut Fly, pos: V3, steps: usize) {
        (f.mode, f.spot, f.pos) = (Mode::Flying, None, pos);
        apply(b, f);
        (0..steps).for_each(|_| b.step());
    }

    /// Land on `spot` and taste it: the peak MN9 over the next 4 s.
    fn land(b: &mut Brain, f: &mut Fly, spot: Spot, sugar: bool, bitter: f32) -> f64 {
        f.sugar = sugar.then_some(spot);
        f.bitter = bitter;
        f.perch(spot);
        apply(b, f);
        (0..120).map(|_| {
            b.step();
            b.mean("mn9")
        }).fold(0.0, f64::max)
    }

    /// A naive brain's first taste: flown from the room's middle straight onto `spot`.
    pub fn mn9_first(spot: Spot, sugar: bool, bitter: f32) -> f64 {
        let (mut b, mut f) = (Brain::load(), Fly::new(&mut Rng::new(1)));
        fly_at(&mut b, &mut f, v3(1.0, 1.3, 1.0), 40);
        land(&mut b, &mut f, spot, sugar, bitter)
    }

    /// An experienced brain, as in the screensaver: it has tasted sweet sugar once and
    /// flown to `away`, and now hovers hungry in the fruit's smell and lands.
    pub fn mn9_experienced(spot: Spot, sugar: bool, bitter: f32, away: V3) -> f64 {
        let (mut b, mut f) = (Brain::load(), Fly::new(&mut Rng::new(1)));
        f.hunger = 0.8;
        let hover = spot.pos() + v3(0.0, 0.15, 0.0);
        fly_at(&mut b, &mut f, hover, 60);
        land(&mut b, &mut f, spot, true, 0.0);
        fly_at(&mut b, &mut f, away, 300);
        fly_at(&mut b, &mut f, hover, 60);
        land(&mut b, &mut f, spot, sugar, bitter)
    }

    /// `floor`: sweet sugar's peak must pass it; `ceiling`: bitter 1.0 must keep it under.
    fn dose_response(label: &str, none: f64, curve: &[f64], floor: f64, ceiling: f64) {
        eprintln!("{label}: no sugar {none:.1e}; sugar + bitter 0/.25/.5/.75/1: {curve:.4?}");
        assert!(none < 1e-4, "{label}: MN9 must be silent without sugar ({none})");
        assert!(curve[0] > floor, "{label}: sugar must drive MN9 past {floor} ({})", curve[0]);
        assert!(curve.windows(2).all(|w| w[1] <= w[0] + 1e-12), "{label}: bitter must not raise MN9: {curve:?}");
        assert!(curve[4] < ceiling, "{label}: bitter 1.0 must suppress MN9 under {ceiling}: {curve:?}");
    }

    const BITTER: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

    #[test]
    fn a_first_taste_of_sugar_bursts_mn9_and_bitter_suppresses_it() {
        for spot in [Spot::Bread, Spot::Banana] {
            let curve: Vec<f64> = BITTER.iter().map(|b| mn9_first(spot, true, *b)).collect();
            // Bitter 1.0 cuts this burst by 97%, but not under the feeding level: a limit (§2).
            dose_response(&format!("first taste, {}", spot.name()), mn9_first(spot, false, 0.0), &curve, 0.3, 0.05);
        }
    }

    #[test]
    fn an_experienced_fly_still_tastes_sugar_and_bitter_still_suppresses_it() {
        for (spot, away) in [(Spot::Bread, v3(0.5, 1.4, 0.5)), (Spot::Bread, v3(2.5, 1.0, 0.8)), (Spot::Banana, v3(0.5, 1.4, 0.5))] {
            let curve: Vec<f64> = BITTER.iter().map(|b| mn9_experienced(spot, true, *b, away)).collect();
            // Sweet sugar four times over the feeding level (0.005), and bitter 1.0 under it.
            let label = format!("experienced, {} from {:?}", spot.name(), (away.x, away.y, away.z));
            dose_response(&label, mn9_experienced(spot, false, 0.0, away), &curve, 0.02, 0.005);
        }
    }
}
