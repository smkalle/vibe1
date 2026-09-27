//! The senses: what the room and the fly's own motion do to its afferents. A port of
//! "A fly in the Matrix" (web/senses.js and Life.senses in web/life.js), with one
//! adaptation: this room has no swatting hand, so the looming detectors see the
//! landing surface expanding during the final approach instead.

use crate::math::{Basis, V3};
use crate::sim::{Fly, Mode, BANANA, BREAD};

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
/// Distance at which an approached surface starts to loom.
const LOOM_RANGE: f32 = 0.3;

/// Body rates in the original's convention: roll (right wing down +), pitch (nose down +), yaw (left turn +).
#[derive(Clone, Copy, Debug, Default)]
pub struct Rates {
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
}

/// Tracks the fly's angles between steps to give body rates.
#[derive(Default)]
pub struct Motion {
    last: Option<(f32, f32, f32)>,
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

/// The surface the fly is landing on, expanding in its view.
fn looming(fly: &Fly) -> f64 {
    let Some(spot) = fly.spot else { return 0.0 };
    match fly.mode {
        Mode::Approach | Mode::Landing => {
            let d = (spot.pos() - fly.pos).len();
            clamp01(((LOOM_RANGE - d) / LOOM_RANGE) as f64 * 0.8)
        }
        _ => 0.0,
    }
}

/// Every channel's level in [0, 1], by population name.
pub fn sense(fly: &Fly, rates: Rates) -> Vec<(&'static str, f64)> {
    let b = Basis::from_euler(fly.yaw, fly.pitch, fly.roll);
    let flying = fly.airborne();
    let tone = if flying { HALTERE_TONE } else { 0.0 };
    let [ol, or] = ocelli(&b);
    let air = if flying { clamp01(fly.speed as f64 / AIRSPEED_SATURATION) } else { 0.0 };
    let (fl, fr) = (optic_flow(rates, true), optic_flow(rates, false));
    let [bl, br] = odour(fly, &b, BANANA);
    let [yl, yr] = odour(fly, &b, BREAD);
    let loom = looming(fly);
    let sugar = if fly.feeding { 1.0 } else { 0.0 };
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
        ("grn:sugar:front_leg", sugar),
        ("leg_touch", if flying { 0.0 } else { 0.6 }),
        ("dan:pam", if fly.feeding { 0.8 } else { 0.0 }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::sim::Spot;

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
        let s = sense(&f, Rates::default());
        assert_eq!(level(&s, "haltere:left"), 0.5);
        assert!(level(&s, "jo:C:left") > 0.1);
        assert_eq!(level(&s, "leg_touch"), 0.0);
        f.mode = Mode::Sitting;
        let s = sense(&f, Rates::default());
        assert_eq!((level(&s, "haltere:left"), level(&s, "jo:C:left"), level(&s, "leg_touch")), (0.0, 0.0, 0.6));
    }

    #[test]
    fn level_flight_ocelli_see_the_sky_equally() {
        let s = sense(&fly(), Rates::default());
        let (l, r) = (level(&s, "ocelli:left"), level(&s, "ocelli:right"));
        assert!((l - r).abs() < 1e-6 && l > 0.5, "{l} {r}");
    }

    #[test]
    fn a_turn_drives_the_optic_flow_cells_apart() {
        let s = sense(&fly(), Rates { yaw: 10.0, ..Rates::default() });
        assert!(level(&s, "lptc:hs:left") > 0.7 && level(&s, "lptc:hs:right") < 0.3);
        let still = sense(&fly(), Rates::default());
        assert_eq!(level(&still, "lptc:hs:left"), 0.5);
    }

    #[test]
    fn banana_smells_like_decaying_fruit_up_close() {
        let mut f = fly();
        f.pos = BANANA + crate::math::v3(0.05, 0.1, 0.0);
        let s = sense(&f, Rates::default());
        assert!(level(&s, "orn:decaying_fruit:left") > 0.3);
        assert!(level(&s, "orn:decaying_fruit:left") > level(&s, "orn:yeasty:left"));
        f.pos = crate::math::v3(0.3, 1.9, 0.3);
        let far = sense(&f, Rates::default());
        assert!(level(&far, "orn:decaying_fruit:left") < 0.1);
    }

    #[test]
    fn feeding_tastes_sugar_and_landing_looms() {
        let mut f = fly();
        f.feeding = true;
        f.mode = Mode::Sitting;
        assert_eq!(level(&sense(&f, Rates::default()), "grn:sugar:labellum"), 1.0);
        let mut f = fly();
        f.mode = Mode::Landing;
        f.spot = Some(Spot::Banana);
        f.pos = Spot::Banana.pos() + crate::math::v3(0.0, 0.05, 0.0);
        assert!(level(&sense(&f, Rates::default()), "vis:LPLC2") > 0.4);
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
        for (n, l) in sense(fly, rates) {
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
