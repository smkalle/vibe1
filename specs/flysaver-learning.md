# Spec: flysaver Phase 3, the fly learns, and remembers across idle sessions

**Status:** BUILT and VERIFIED (2026-09-27). §9 has the results.
**Builds on:** [flysaver-brain-pilot.md](flysaver-brain-pilot.md) (Phase 2: the brain's output neurons steer, and the mushroom body chooses approach or avoid over a fruit).

## 1. Goal

The mushroom body's choice stops being fixed. As in *A fly in the Matrix*, each choice is a **lesson**:

- sugar on the fruit it lands on is a reward of **+1**
- leaving an empty fruit, or avoiding one, is **0**
- the swatter striking it while it sits on a fruit is **−1**

The library's actor-critic moves the **16,795 Kenyon-cell → MBON synapses**, the site of the fly's olfactory memory. Sugar sits on one fruit and moves now and then, so the fly has to relearn: reversal learning. What it has learned is **saved to disk**, so the screensaver fly carries its memories from one idle period to the next.

## 2. What is ported

The source is `web/learner.js` (MIT), the original's one-stream actor-critic, which it states is the same rule as `cadence.plasticity.ActorCritic`. It uses the page's configuration:

| | |
|---|---|
| outputs | `mbon:MBON11:right` (approach), `mbon:MBON05:left` (avoid) |
| plastic synapses | from `kc` onto `mbon`: 16,795 classes, starting at the payload's naive seam efficacies |
| critic | the 4,130 Kenyon cells |
| constants | β 0.1, T 0.3, 10 nudged steps, tolerance 1e-3, γ 0.95, λ 0.9, η 1.0, η_critic 0.05, efficacy cap 3, dopamine cap 1 |

**Act.** On the live (free) state:
1. Compute p = softmax(s_out / T) and sample a choice with the library's cumulative draw.
2. Run two **nudged settles** (brain.js `settleNudged`), toward and away from the choice's one-hot target at strength ±β, on copies of the state.
3. For each plastic synapse, the trace becomes `trace·γλ + (a⁺(b⁺ − b⁻) + (a⁺ − a⁻)b⁻) / 2β`.
4. Update the critic's trace the same way.

**Learn.**
1. Dopamine δ = clip(reward + γ·V(next) − V(decision), ±1).
2. Each plastic efficacy moves by `η·δ·trace`, clipped to ±3, and its weight is rebuilt as `(gain·count·exp(log_gain[pre]))·efficacy`.
3. The critic moves by `η_c·δ·trace / (1 + ‖trace‖²)`.
4. A finished lesson clears the traces.

The weights update inside the engine's sender-ordered storage. A synapse → position map is kept for the plastic seam only.

## 3. Lessons in the screensaver

This follows `life.js`'s search → decide → outcome flow:

1. **Decide.** The fly hovers over a fruit, the brain acts, and the choice goes to the pilot. (Phase 2 used a plain softmax here without learning.)
2. **Approach and land:**
   - on the sugared fruit: +1 on landing, and the fly feeds (sugar taste and PAM reward dopamine)
   - on the other fruit: the lesson stays open while the fly sits. It ends with 0 when the fly leaves, or −1 if the swatter strikes (PPL1 punishment dopamine).
3. **Avoid:** the fly leaves, and the lesson ends with 0 a few seconds later.
4. **Approach that never arrives** (timed out): 0.
5. **Sugar** starts on a random fruit and moves after `sugar_minutes` of screensaver time (default 20). It's marked with a small sparkle.
6. **Feeding** now happens only on the sugared fruit, as in the original.

## 4. Memory

**Where.** `~/.local/state/flysaver/memory.bin` holds:
- a header with a magic number, a version, and the brain's weight checksum and seam size (anything that doesn't match is ignored)
- the 16,795 efficacies (f64)
- the critic weights and bias
- counters: lessons, rewards, blows
- where the sugar is and its elapsed minutes

**When it's written:** after every lesson and on exit, atomically (a temporary file, then rename). Each monitor runs its own fly, and the last writer wins. Every monitor loads the saved memory at start.

**Commands:** `flysaver forget` deletes it. `flysaver doctor` reports the lesson count, the changed synapses, and the approach probability for each fruit.

## 5. Settings

| Key | Default | |
|---|---|---|
| `learning` | `true` | lessons change the synapses; `false` keeps the naive, measured seam |
| `remember` | `true` | load and save `memory.bin` |
| `sugar_minutes` | `20` | minutes of screensaver time before the sugar moves (5–240) |

## 6. HUD

The lesson results show on the note line, for example `› lesson: banana, sugar +1 (δ 0.96)`. The status line gains a memory summary: `memory 12 lessons · sugar on banana · p banana 0.56 / bread 0.83`.

## 7. Verification

1. **Parity with the original's learner.** `tests/learner_cases.txt` is recorded by `tools/record_learner.mjs`, which runs the original `learner.js` on its `brain.js` with the page's configuration. It holds 10 lessons with recorded draws, sugar on the banana, and a blow in lesson 6. The Rust port must match, per lesson, to 1e-9 relative: p, the choice, V, the TD error, δ, the moved and changed synapse counts, the mean change, Σ efficacy, V after, and Σ activation.
2. **Learning happens (a T-maze, the original's gate 4).** Starting naive, sugar on the banana raises p(approach banana) past 0.5. Moving the sugar to the bread brings the fly back to preferring the bread. The test reports the number of lessons each direction takes.
3. **Memory round trip.** Save then load gives identical efficacies and choice probabilities. A wrong checksum or seam size is refused.
4. **End to end.** A headless run (`flysaver train --minutes N`) runs the full screensaver scene with lessons and logs them. It must show lessons with rewards and the sugar fruit's approach probability rising.
5. **Existing suites** all pass.

## 8. Out of scope

- The page's shuffled-wiring and frozen controls.
- Learning anywhere other than the Kenyon-cell → MBON seam.
- Sharing one memory live between monitors.

## 9. Verification results

| Check | Result |
|---|---|
| Parity with the original's `learner.js` (10 lessons, recorded draws, a blow in lesson 6) | worst relative deviation **7.6e-17**; the same choice, moved and changed counts in every lesson |
| Engine parity (Phase 1, unchanged) | 3.3e-16 |
| T-maze acquisition: sugar on the banana | p(approach banana) **0.35 → 0.55 in 10 lessons** |
| Reversal by extinction: the sugar moves, the banana pays 0 | banana **0.55 → 0.39 in 46 lessons** |
| Reversal with blows at the banana (the original's T-maze) | banana **0.55 → 0.37 in 6 lessons** |
| Memory round trip, and foreign files refused (another brain, another seam, truncated, not a memory) | pass; a new scene reloads identical efficacies, counters and sugar |
| End to end (`flysaver train --minutes 15`, 3 seeds, ~38 s each) | 31–42 lessons, 8–14 sugar rewards, 9.3k–9.7k of 16,795 synapses changed. The sugared fruit's approach p rose or the other fruit's fell on every seed (seed 3, sugar on the bread: bread 0.45 → 0.80). |
| Swatter across the three 15-minute runs | 25 escapes and 2 hits (the giant fibre was too slow twice). Each hit on a fruit became a −1 lesson. |
| Cost | a decision (act, two nudged settles) takes 29 ms once per lesson; learning takes 0.12 ms; the frame is 2.06 ms (6.2% of a core); peak RSS 56.2 MB |
| Suites | `cargo test` 73 passed, clippy clean; pty, Hyprland-contract and install suites pass |

**Bugs found by verification and fixed:**

1. **Asking before the smell.** The first end-to-end run showed every decision at p ≈ 0.37, the rest value, because the fly asked the moment it arrived and before the brain had settled into the smell. It now hovers for 1 s (about 30 steps) first, as the original waits for its brain.
2. **Sated flies deciding.** Decisions taken with hunger near zero had no smell drive. Flies now only search when hunger is above 0.2, the original's APPETITE.
3. **Lost blows.** The swatter's hit landed after the lesson logic in its frame and was lost, so no lesson ever got its −1 and every run reported 0 blows. The swatter now steps right after the fly, with a regression test. A new decision also closes a still-open lesson with 0, instead of silently dropping it.

**Not verified:** a real Omarchy desktop, and several monitors saving at once. The design is last-writer-wins with atomic renames, but that hasn't been run with real monitors.
