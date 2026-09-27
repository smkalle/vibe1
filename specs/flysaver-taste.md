# Spec: flysaver Phase 4, taste (the proboscis extension reflex, and bitter)

**Status:** BUILT and VERIFIED (2026-09-27). §5 has the results.
**Approved to build** in the same request ("Build phase4 add distinct color for retractable proboscis").
**Builds on:** [flysaver-learning.md](flysaver-learning.md).

## 1. Goal

Feeding becomes the brain's decision, through the classic gustatory assay: the **proboscis extension reflex**.

- **Contact.** A fly that lands on the sugared fruit tastes it with its labellum. The proboscis motor neuron **MN9**, in the live model, extends the proboscis, and the fly feeds if MN9 fires.
- **Bitter.** Sometimes the sugared fruit is **laced with a bitter compound** (caffeine), at a random strength. Bitter taste suppresses MN9 through the measured wiring, and it is a punishment the mushroom body learns from.
- **Colour.** The proboscis gets its own colour: **amber-gold `#ffb020`**, shifting toward **violet `#a070ff`** as the taste turns bitter. It is distinct from the red eyes (`#ff3048`) and the theme-coloured body.

## 2. Measured before designing (and while building)

**MN9 is phasic.** Without sugar it is 0: exactly 0, or about 1e-5 on a naive brain. Tasting sugar, it rises about 1.3 s after landing (40 steps), peaks, and decays back to 0 within about 2 s. So the reflex is read as the **peak** over the first seconds of contact.

**A brain's first taste is special.** The first sugar a freshly loaded brain ever tastes bursts MN9 to 0.3–0.48. That taste moves the network into a lasting state, and every later taste peaks at only 0.02–0.3. This isn't dopamine: it happens with no reward drive at all, and 10 s of flight doesn't undo it. Every idle session loads a fresh brain, so each session has one naive first taste.

**Peak MN9, from the model** (`src/senses.rs`, test module `taste`). Bitter goes on the labellum and front legs:

| Condition | no sugar | sugar | + bitter 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| first taste, bread | 1e-5 | **0.476** | 0.475 | 0.442 | 0.229 | 0.013 |
| first taste, banana | 1e-5 | **0.435** | 0.433 | 0.385 | 0.206 | 0.015 |
| experienced, bread (flown to one corner) | 0 | **0.025** | 0.021 | 0.004 | 0.001 | 0.000 |
| experienced, bread (flown to another) | 0 | **0.116** | 0.109 | 0.029 | 0.004 | 0.000 |
| experienced, banana | 0 | **0.199** | 0.194 | 0.091 | 0.021 | 0.002 |

**In the full scene**, 152 landings on unlaced sugar (6 seeds × 30 min) peaked at **0.016–0.31**. Every one is above 0.005.

**Findings:**
- **Bitter suppresses MN9, graded and monotone,** in every state. That's one of the original's physiology facts, reproduced.
- **The original's `FEED_LEVEL` of 0.2 is reached only on the first taste.** An experienced fly would never feed. So the feeding level is set against the zero baseline instead: **0.005**. That's 3× under the weakest sweet peak seen, and far above no sugar.
- **Limit: the first taste.** On a naive brain, bitter 1.0 still leaves 0.013, above the feeding level. So the first sugar in an idle session is taken even when strongly laced. Its lesson is then fed at 1 − caffeine, which is about 0.
- **Limit: sugar on the front legs silences MN9 in this model.** Real flies extend the proboscis when their legs touch sugar. So sugar goes on the labellum only.
- **Food odour lowers the sitting level.** Sitting on the fruit in its smell, MN9 settles back to 0. That's why the reflex is read at its peak.

## 3. Design

| Part | Rule |
|---|---|
| Taste channels | Sitting on the sugared fruit drives `grn:sugar:labellum` 1.0. If the fruit is laced, `grn:bitter:labellum` and `grn:bitter:front_leg` get the bitter level. Sugar is **not** put on the legs (§2). |
| Feeding | Brain-piloted: when MN9 passes **0.005** while tasting, a feeding bout starts and is **latched** (it outlives the phasic peak). Instincts: as before. |
| Proboscis | Extension = √(clamp(MN9 / 0.05, 0, 1)), smoothed, and fully out while feeding. A refused, laced taste shows as a short flick. It's drawn last, over the legs, as a two-strand tube with the labellum's two lobes. |
| Bitter as a teacher | While bitter is tasted, the PPL1 punishment dopamine neurons are driven at 0.8 × bitter. That's supplied, just as sugar's PAM drive is, because in this sub-net bitter taste doesn't reach the DANs. |
| Lesson outcome on the sugared fruit | After **2 s** of tasting: reward = (fed ? 1 : 0) − bitter. The note shows the peak, for example `lesson: bread, sugar + caffeine 0.86, MN9 peak 0.001 → refused -0.86`. |
| Lacing | Each time the sugar is placed or moves, it's laced with probability 0.4, at a level uniform in [0.2, 1.0]. The level is shown on the HUD (`sugar on the bread (caffeine 0.60)`) and saved in memory (format v2; v1 files load as unlaced). |
| Colour | `proboscis` is a fixed role, like the eyes: `#ffb020` for sugar, mixed in **OKLab** toward `#a070ff` by the bitter level. An RGB mix passes through a dusty red (`#d09090` at 0.5) that reads as the eyes, which the first build did. |

**Settings:** `bitter = true` (lacing on or off). **Snapshot:** `flysaver snapshot --taste B` renders a fly feeding on sugar laced at B.

## 4. Verification

- **Dose-response from the model (asserted), for both a naive and an experienced brain:**
  - silent without sugar
  - sweet above the floor (0.3 naive, 0.02 experienced)
  - monotone under bitter
  - bitter 1.0 below 0.05 naive, and below the feeding level experienced
- **The feeding rule in the scene:** an experienced, brain-piloted fly on the unlaced sugared bread feeds and extends the proboscis; with bitter 1.0 it doesn't.
- **Lessons:** laced sugar gives a negative reward and dopamine. A T-maze with a strongly laced sugared banana makes the fly avoid the banana.
- **End to end,** with lacing on: fed and refused counts per caffeine level, and the approach probabilities.
- **Colour:**
  - the proboscis, eyes and body are at least 40° apart in OKLab hue
  - every shade from sweet to bitter stays 30° from the eyes, or is pale (chroma < 0.08)
- **Memory:** v1 files still load, and v2 round-trips the lacing.
- **Existing suites:** parity (engine, learner), the golden frames, and the pty, Hyprland and install tests.

## 5. Verification results

| Check | Result |
|---|---|
| Dose-response (§2 table) | naive and experienced both pass. No sugar ≤ 1.2e-5, bitter monotone. Experienced bitter 1.0 ≤ 0.002 |
| Feeding rule (scene) | sweet: fed, proboscis 1.00; bitter 1.0: refused, proboscis 0.00 |
| Bitter lesson (scene) | `sugar, MN9 peak 0.301 → fed +1 (dopamine +1.00)`; `sugar + caffeine 1.00, MN9 peak 0.002 → refused -1 (dopamine -1.00)` |
| Laced T-maze (learner) | laced banana approach p **0.35 → 0.20 in 10 lessons** |
| End to end: `flysaver train --minutes 45`, lacing on, 4 seeds | unlaced sugar: **94 fed, 0 refused**. Caffeine 0.34–0.40: **19 fed, 0 refused** (reward +0.60/+0.66). Caffeine 0.86–0.89: **30 refused, 1 fed**; the one fed was seed 1's naive first taste (§2 limit). |
| Learning from bitter | seed 1 (bread laced 0.89): bread p **0.52 → 0.34**. Seed 2 (bread laced 0.86 after the sugar moved): bread p **0.58 → 0.26**, while the unlaced banana rose 0.36 → 0.62 |
| Colour, OKLab hue from the proboscis | eyes **53°**, body **79°**, wireframe 72°, firing 55°. Violet end vs eyes **86°**. The path goes amber → peach → pale rose (chroma 0.065 where its hue nears the eyes) → lilac → violet |
| Visual | 90×34 truecolor renders show the amber tube under the red eyes at bitter 0, and lilac at 0.6 |
| Memory | v1 files load with bitter 0; v2 round-trips 0.4 |
| Cost | 1.84 ms/frame at 120×40 (5.5% of a core at 30 fps), the same as Phase 3 |
| Suites | `cargo test` 80 passed. Clippy clean on the binary (3 older lints in tests). Pty, Hyprland-contract and install suites pass. The goldens are unchanged, since no golden frame shows a feeding fly |

**Bugs found by verification and fixed:**

1. **Continuous gating never fed.** The first design fed while MN9 > 0.01. But MN9 is phasic, so feeding stopped as soon as the peak passed. Feeding is now latched on the peak.
2. **The original's 0.2 fed only once per session.** The first end-to-end runs used it, and the fly fed once in three 20-minute runs, because only a naive brain's first taste reaches 0.2. One seed learned nothing at all: every lesson paid 0, so the dopamine was 0. The threshold is now 0.005 (§2).
3. **Tests measured the wrong brain.** The first tests only saw the naive first taste, landing straight from flight. They now also run an experienced brain: a sweet taste, flight away, a hover in the smell, then landing.
4. **The bitter shade looked like the eyes.** The RGB mix gave `#d09090` at bitter 0.5. It's now an OKLab mix, with a test over the whole path.
5. **The proboscis was hidden.** The legs, drawn after it, took its braille cells. It's now drawn last, and a third longer.
6. **Unformatted rewards.** Fractional rewards printed as `-0.8923935294151306`; they're now rounded to 2 decimals.

**Not verified:** a real Omarchy desktop.
