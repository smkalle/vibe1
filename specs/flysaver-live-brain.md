# Spec: flysaver Phase 1, a live brain (and more vivid colour)

**Status:** BUILT (2026-09-27). Section 10 has the measured results and the deviations from this spec.
**Builds on:** [omarchy-fly-matrix-screensaver.md](omarchy-fly-matrix-screensaver.md). This is Phase 1 of the "true neuron" roadmap.

## 1. Goal

Replace the decorative brain firing with the **real rate model** from *A fly in the Matrix*, running on the 60,000-neuron BANC sub-net that the original page runs. The fly's procedural life feeds its real sensory neurons. Every drawn neuron is lit by its actual activity. Behaviour stays procedural in this phase: the brain watches, it doesn't fly. That's Phase 2.

Also make the colours more vivid across the whole scene.

## 2. What is ported

- **Model.** The source is `fly-matrix/web/brain.js` (MIT), which follows the Cadence library's own order of operations. Each step:
  - `total_i = Σ_e s[pre_e]·w_e`, summed over the synapses into neuron i in stored order
  - `v_i += dt·(total_i + (drive_i + bias_i) − v_i)`
  - `s_i = max(0, σ(slope·(v_i − threshold)) − rest) / (1 − rest)`

  The parameters are `dt = 0.2`, `slope = 4`, `threshold = 1.5`, `gain = 0.02`, `amplitude = 3` and `leak = 0`. There's no adaptation, and no nudged or learning phase (those are Phase 3).
- **Weights.** `w_e = (gain · count_e · exp(log_gain[pre_e])) · efficacy_e`, where efficacy is the sign in {−1, 0, 1}, except for 16,771 listed values. Rebuilt this way, the weights match the payload's f64 weights **bit for bit** on all 1,209,528 synapse classes (checked).
- **Stimulus.** `stimulate(name, level)` sets `drive = amplitude·level` on the population's neurons, and levels combine by max. Drives are cleared and set again before every run.
- **Numbers.** Everything runs in f64 and in the library's order, so parity holds.

## 3. Asset

`tools/build_brain.py` (standard library only) reads the original's `web/data/brain.json` and `atlas.json` and writes `assets/brain.bin`, which is embedded in the binary. It contains:

- model parameters, `n = 60000`, `edges = 1209528`
- per neuron: the row length, log-gain class, bias (sparse), and atlas soma position (u16×3) and region (for drawing)
- per synapse: the sender as a delta varint (senders are sorted within each row), the count as a varint, and a 2-bit sign
- the efficacy exceptions as (index, f64)
- all 321 named populations (delta-varint index lists)

The target size is at most about 4 MB. The BANC data is CC BY, and the credits carry it.

## 4. Senses (procedural fly → afferents)

These follow `life.js` and `senses.js`. Two channels are adapted, because this room has no hand.

| Population(s) | Drive |
|---|---|
| `haltere:left/right` | 0.5 in flight, 0 at rest (the flight tone) |
| `ocelli:left/right` | how much each ocellus (45° up, 35° to the side) faces the sky, from the body orientation |
| `lptc:hs/vs:left/right` | 0.5 ± rotational optic flow from the yaw, roll and pitch rates (saccades drive them hard) |
| `jo:C/E:left/right` | airspeed / (1 m/s) |
| `orn:decaying_fruit:*` (banana), `orn:yeasty:*` (bread) | the original's odour plume: a narrow core plus a faint wide plume, bilateral comparison, and a gain raised by hunger |
| `grn:sugar:labellum`, `grn:sugar:front_leg` | 1 while feeding |
| `leg_touch` | 0.6 while standing |
| `dan:pam` | reward while feeding |
| `vis:LC4`, `vis:LPLC2` | **adapted:** the landing surface expanding in view during the final approach, instead of the page's swatting hand |

## 5. Running it

- **Warm-up:** 200 steps of the level-flight rest at start-up, as the page does.
- **Pace:** one brain step per rendered frame by default, `brain_steps = 1` (range 1–4). At 30 fps that's 30 steps/s of model time. The rate model's time constant is about 5 steps, so responses show within a fraction of a second.
- **Performance budget:** at most 2 ms per step on one core (a measured target, not a guess) and at most 60 MB RSS. If a step is slower, frames still render and the brain runs at whatever step rate fits.
- **Setting:** `brain = "live" | "decorative"`, default `live`. `decorative` keeps today's behaviour.

## 6. Drawing

- **Silhouette.** The whole-brain sample in `connectome.bin` stays as a dim silhouette, with today's adaptive sparsity.
- **Live neurons.** Every one of the 60,000 sub-net neurons with `s > 0.02` is drawn at its soma, lit by `s` on a vivid ramp: theme hue → fire colour → white-hot.
- **HUD.** It shows `live · N active · X.X ms/step`, and the credits line says the brain is simulated. `flysaver about` changes from "decorative" to what is simulated and what is supplied.

## 7. More vivid colour

- The theme's roles get an OKLab chroma boost (about ×1.35, clipped to the displayable range).
- Brightness floors go up: braille from 0.25 to 0.4, the fog floor from 0.12 to 0.25, and the rain trail from 0.42 to 0.55, with a full-brightness head.
- Firing uses a three-stop heat ramp.
- `vivid = true` is the default; `false` restores the previous look.
- 256-colour mode gets the same boost before mapping.

## 8. Tests

- **Parity with the Cadence library.** `tests/parity_cases.json` from the original (per-step population means and the final active count, produced by `cadence.Brain`) is converted to a plain text fixture. A Rust test replays every case and asserts a worst deviation below 1e-9 and identical active counts, the same bar as `parity.mjs`.
- **Asset.** It decodes, and the weights rebuilt from it match a checksum of the payload's weights.
- **Senses.** Flying gives a haltere tone, a saccade drives the LPTCs, being near the banana drives `orn:decaying_fruit`, and feeding drives `grn:sugar`.
- **End to end.** After warm-up, a saccade raises activity in the optic lobes compared with level flight.
- **Existing suites.** The golden frames are regenerated, and the pty, Hyprland and install suites must still pass.
- **Bench.** CPU per frame and output bytes in live mode are reported in the README.

## 9. Out of scope (later phases)

- The brain steering the fly (Phase 2).
- Learning and memory across sessions (Phase 3).
- The page's swatting hand, lesions, and the shuffled-wiring switch.

## 10. Build notes (measured)

| Spec target | Result |
|---|---|
| Parity with `cadence.Brain` < 1e-9, active counts equal | **3.3e-16** worst deviation over 6 cases × 40 steps; active counts equal |
| Weights rebuilt bit for bit | yes: FNV-1a `8bef125c009c1350` over all 1,209,528 f64 weights, checked by `flysaver doctor` and a unit test |
| Asset at most about 4 MB | **4.20 MB** (`assets/brain.bin`); binary 5.0 MB |
| At most 2 ms per step | about **1.1–1.6 ms** per step, from pushing out of active senders (see below) |
| At most 60 MB RSS | **50.5 MB** peak |
| Start-up | about **110 ms** from launch to first frame, including 40 warm-up steps |

**Deviations from the spec:**
- **Summation order.** The spec described the library's pull-style sum over all synapses. The build pushes from active senders in ascending order instead. That's bit-identical, because senders are sorted within each row and silent neurons are exactly zero, and it's about 35% cheaper per step (1.72 → 1.13 ms under a strong odour). The parity deviation is unchanged at 3.3e-16.
- **Warm-up.** 40 steps instead of 200, to keep start-up under 150 ms. That's about eight of the model's time constants.
- **Vivid colour.** A first pass (chroma ×1.35, floors 0.4/0.25/0.55) was barely visible on muted themes. The shipped defaults are chroma ×1.8 with an OKLab lightness floor of 0.72, and brightness floors of 0.5 (braille), 0.35 (fog) and 0.65 (rain trail).

**Frame cost at 230×65, 30 fps:**

| | CPU per frame | Share of one core |
|---|---|---|
| Live brain | 2.1–2.2 ms | 6.3–6.6% |
| Decorative brain | 0.6–0.65 ms | 1.8–2.0% |

Terminal output is about the same in both modes.

**Responses verified in the model** (tests `through_the_wiring`, not scripted):
- A 10 rad/s yaw moves the wing steering motor neurons (iv4, b1, b3) and the neck motor neurons, and raises the active count from 71 to 85.
- The banana's smell drives the projection neurons to 0.18 and a sparse Kenyon-cell code (0.9%), engages the APL neuron, and raises MBON11 from 0.10 to 0.20.

**Not verified:** how it looks and costs on a real Omarchy desktop. This environment has no display.
