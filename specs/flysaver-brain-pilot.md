# Spec: flysaver Phase 2, the brain flies the fly (a red-eyed mutant)

**Status:** BUILT (2026-09-27). §10 has the measured results.
**Builds on:** [flysaver-live-brain.md](flysaver-live-brain.md) (Phase 1: the live rate model, fed by the senses).

## 1. Goal

The live brain's **descending, giant-fibre and mushroom-body output neurons now steer the fly**, the way *A fly in the Matrix* does (`web/life.js`, "the brain layer"). The procedural behaviour stays as the declared instinct layer, which is the original's control condition and fallback. The brain layer reads its neurons as deviations from their level-flight rest and overrides the instincts.

The fly is also restyled as a red-eyed mutant.

## 2. What the real wiring does under our senses (probed before designing)

Probe: 60 model steps per condition, population means.

| Condition | What the model does |
|---|---|
| yaw rotation ±10 rad/s | DNa02 left/right move apart, **against** the turn: an optomotor stabilising reflex that emerges from the wiring |
| looming 0.3 | nothing (giant fibre 0.006, landing 0.003) |
| looming 0.45–0.55 | giant fibre climbs to about 0.50; `dn:landing` starts rising around 0.5–0.55 |
| looming ≥ 0.6 | giant fibre 0.54–0.92 (**fires**), `dn:landing` about 0.20 |
| hovering over the bread | MBON11 0.82 (approach) vs MBON05 0.38 (avoid). The original's receipt says 0.815 vs 0.385. |
| hovering over the banana | MBON11 0.21 vs MBON05 0.41: the naive fly prefers the bread |
| sugar, touch, wind | DNp09 (speed), MN9 (feeding) and `dn:grooming` stay at 0 |

Because landing and escape switch on at nearly the same looming level, the Phase 1 stand-in (the landing surface looming) would make the brain abort almost every landing. **Phase 2 removes surface looming** and adds the original's kind of threat instead.

## 3. The brain layer (ported rules and constants from `life.js`)

The **baseline** is the readouts after the 40-step warm-up in level flight. `dev(x) = mean(x) − baseline(x)`.

| Readout | Rule | Original constant |
|---|---|---|
| `dn:DNa02:right` − `dn:DNa02:left` (as deviations) | when \|asym\| > 0.02, turn at `K_TURN · asym` rad/s | K_TURN = 6 |
| `dn:DNp09` | speed = cruise · clamp(1 + 0.6·dev, 0.3, 1.5) | K_SPEED = 0.6 |
| `dn:landing` | dev > 0.15: land now (the nearest surface: table, fruit or floor) | LAND_LEVEL = 0.15 |
| `gf` (giant fibre) | > 0.5: **escape**. Sitting: jump-take-off away from the threat. Flying: evasive turn and burst. | ESCAPE_LEVEL = 0.5 |
| `mn9` | on sugar and > 0.2: feed | FEED_LEVEL = 0.2 |
| `dn:grooming` | sitting and dev > 0.15: groom | GROOM_LEVEL = 0.15 |
| MBON11 vs MBON05 | once per fruit approach, within reach: `p(approach) = softmax(m11/T, m05/T)`; approach lands, avoid leaves | T = 0.3 (the page's) |

Rules whose neurons stay silent under our senses (speed, feeding, grooming) are still ported. They just never fire, so the instinct layer carries those behaviours. The HUD and `about` say this plainly.

## 4. The threat: a swatter

Every 40–90 s, a wireframe **swatter disc** (radius 2.5 cm, the original's hand) comes at the fly from a random direction, starting 0.8 m away at 0.6 m/s.

- **Looming input.** It drives `vis:LC4` and `vis:LPLC2` with the original's formula: the rate of angular expansion `dθ/dt`, with θ = 2·atan(R/d), divided by 4 rad/s, plus a short memory of 0.7.
- **If the giant fibre fires first,** the fly escapes.
- **If it doesn't,** the swatter hits. The fly tumbles for 0.4 s, and `dan:ppl1` (punishment dopamine) is driven for 0.8 s.
- **In instinct mode** nothing reads the giant fibre, so the fly gets hit. That's the contrast the original's brain/instincts switch shows.

## 5. Settings

| Key | Values | Effect |
|---|---|---|
| `pilot` | `"brain"` (default) \| `"instincts"` | who steers. `brain` needs `brain = "live"`; with `decorative`, the pilot falls back to `instincts`. |
| `threats` | `true` (default) \| `false` | turns the swatter on or off |
| `eyes` | `"red"` (default) \| `"theme"` | eye colour; see §6 |

## 6. The red-eyed mutant

The compound eyes become a vivid red (`#ff3048`), in every theme and palette. The earlier look is kept as `eyes = "theme"`.

(A note for the README: real wild-type *Drosophila* eyes are brick red. Morgan's famous 1910 mutant is the *white*-eyed one. In the Matrix, green is normal, so the red-eyed fly is the mutant.)

## 7. HUD

The status line shows who is piloting and the last brain decision, for example:
- `giant fibre 0.93 → escape`
- `mushroom body: bread, approach (p 0.81)`
- `DNa02 turn −0.03`

## 8. Tests

- **Brain layer.** With scripted readouts: escape from sitting takes off, a landing request lands, the DNa02 asymmetry turns the fly, and an MB "avoid" leaves the fruit.
- **End to end with the real model.**
  - A swatter aimed at a sitting fly: with `pilot = "brain"` the giant fibre fires and the fly escapes before contact. With `pilot = "instincts"` it is hit.
  - MB choices over many seeds: bread is approached much more often than the banana.
- **Existing suites.** Parity, senses, the golden frames (regenerated), and the pty, Hyprland and install tests all pass.

## 9. Out of scope

- Learning, where the MB choices change with sugar and blows (Phase 3).
- Wingbeat-level body physics. The original supplies that as a hand-written inner loop; ours stays kinematic.
- A shuffled-wiring control.

## 10. Build notes (measured)

| Check | Result |
|---|---|
| Swatter at a sitting fly, `pilot = "brain"` | the giant fibre fires (0.50+) with the swatter about **0.10 m away**, 35 frames after it starts. The fly escapes unhurt on all 5 seeds tested. |
| Same, `pilot = "instincts"` | **hit** on all 3 seeds tested (nothing reads the giant fibre) |
| Mushroom body over each fruit (hunger 0.7, 12 cm up) | p(approach) **bread 0.70, banana 0.35** |
| Brain-layer rules, with scripted commands | escape jumps off away from the threat; a land request picks the table or the floor; a DNa02 command turns the fly; "avoid" leaves the fruit and the next approach skips it; instincts never ask |
| Cost at 230×65, 30 fps | brain-piloted 2.08 ms/frame (6.3% of a core); instincts 1.95 ms |

**What changed from Phase 1:** surface looming is removed. The swatter is now the only looming stimulus, because landing and escape switch on at nearly the same looming level (§2).

**Added:** `flysaver snapshot --swat SECONDS` renders the escape for screenshots.

**Not verified:** a real Omarchy desktop.
