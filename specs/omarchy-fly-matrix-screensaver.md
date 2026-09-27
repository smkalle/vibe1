# Spec: "Fly in the Matrix" screensaver for Omarchy

**Status:** APPROVED with the recommendations in §11 (2026-09-27). Built in [`flysaver/`](../flysaver/). §6 and §8 were revised during the build; see §13.
**Working name:** `flysaver` (binary), shown to users as "Fly in the Matrix".
**Inspiration:** [A fly in the Matrix](https://floatingpragma.io/cadence-examples/fly-matrix/) by Bernhard Mueller / Pragma Research ([source](https://github.com/Jarikononen/cadence-examples/tree/main/fly-matrix), MIT).

---

## 1. Summary

This is a drop-in alternative to Omarchy's stock screensaver. It renders a terminal-native take on the "fly in the Matrix" scene:

- digital rain in the Matrix style
- a green wireframe room
- a wireframe fruit fly that flies, saccades, lands and grooms
- a slowly rotating point cloud of the fly's real nervous system (neuron soma positions from the BANC connectome)

It fits into Omarchy's existing idle, launch and exit flow, uses the current Omarchy theme's colours, and can optionally show the user's `screensaver.txt` branding. It does not run a neural simulation. Section 9 covers why.

## 2. Research findings

### 2.1 How Omarchy's screensaver works today (basecamp/omarchy @ `c5b4db7`, 2026-09-26)

| Piece | What it does |
|---|---|
| `shell/plugins/services/idle/Service.qml` | Quickshell idle service. When `idle.screensaver` seconds pass (default 150, set in `~/.config/omarchy/shell.json`) it runs `omarchy-launch-screensaver`, unless the session is locked. The lock fires at `idle.lock` (default 300). The service watches Hyprland `openwindow`/`closewindow` events for class `org.omarchy.screensaver`. When all such windows close before the lock deadline, the idle cycle is cancelled (the dismissal counts as activity). |
| `bin/omarchy-launch-screensaver [force]` | Exits early if one is already running (`pgrep -f org.omarchy.screensaver`). Honours the toggle flag `~/.local/state/omarchy/toggles/screensaver-off` unless `force` is passed. Supports only Alacritty, Ghostty, Foot and Kitty. For each monitor it focuses the monitor, `exec`s the terminal with class `org.omarchy.screensaver` and a screensaver config (black background, font 18, no padding), then waits on the Hyprland socket2 `openwindow` event before moving to the next monitor. |
| `default/hypr/apps/system.lua` | Window rules for `org.omarchy.screensaver`: fullscreen, float, slide animation. |
| `bin/omarchy-screensaver` | Runs inside each terminal. Sets the background to black via OSC 11, hides the Hyprland cursor, waits for the PTY to grow past 80x24, then loops `ttfx -i ~/.config/omarchy/branding/screensaver.txt --random-effect --frame-rate 120 …`. It polls every 1 s and exits on **any stdin byte** or when **its window loses focus** (`hyprctl activewindow`). On exit it restores the cursor, `pkill ttfx`, and `pkill -f org.omarchy.screensaver` (closing every monitor's window). |
| `bin/omarchy-system-lock` | `pkill -x ttfx` plus `pidwait` before locking. |
| Branding | `~/.config/omarchy/branding/screensaver.txt`. Set from an image with `omarchy-transcode-ascii` (braille or block mode). |
| Theme | Active theme at `~/.local/state/omarchy/current/theme/colors.toml` (keys: `accent`, `background`, `foreground`, `green`, `bright_green`, …). The `theme-set.d` hook fires on theme change. |
| Extensibility | **There's no supported hook for replacing the screensaver program.** `~/.local/bin` is *appended* to `PATH`, so it can't shadow `omarchy-screensaver` or `ttfx`. `$OMARCHY_PATH` is a git checkout that updates overwrite. |

Community prior art: `arkits/omarchy-screensaver` (a macOS port), `limehawk/tte-screensaver` (Windows), `omacom/omarchy` discussion #1527 ("Screensaver improvements"), and Yoan Ivanov's "Creating a custom screensaver on Omarchy" blog post. Every one of them keeps the terminal and text-effect aesthetic.

### 2.2 What the fly-matrix page is

- A three.js (r160) WebGL page. The room is drawn as a **green wireframe on black** (`#39ff6a` lines, `#d8ffe4` rain heads), with a table, a banana and a piece of bread.
- **Digital rain** (half-width katakana, digits, `:・=*+-<>`) is textured onto two walls, stepped about 10×/s, with occasional "stutter"/"jitter"/"flicker" glitches.
- A detailed **wireframe fly** (body, compound eyes, wings drawn as translucent stroke-blur discs) flies bouts with saccades, lands, sits, grooms and feeds. The camera follows it.
- A **brain panel** draws 60k–150k neurons at their measured 3D soma positions as a glowing green point cloud (`brain_scan.js`, `data/atlas.json` ≈ 11 MB). It shows live activity.
- A **compound-eye panel** renders the fly's view as a hex-pixel mosaic. A **close-up panel** shows the fly in wireframe.
- The HUD uses Share Tech Mono / IBM Plex Mono, glass panels and scanlines, with stats such as "flying · 0.38 m/s · 132 cm up · 210 Hz".
- The behaviour runs on a real 60k-neuron connectome simulation plus a hand-written instinct layer. It's heavy: a 37 MB payload, a Web Worker, and WebGL.
- Licences: the example code is **MIT**. The connectome data is **BANC release 888, CC BY**, so attribution is required.

### 2.3 Key design tension

Omarchy's screensaver is a **terminal program**. The inspiration is a **WebGL page**. We can:

- **A. Port the look into the terminal** (truecolor plus Unicode braille and half-block sub-pixels). This keeps Omarchy's launcher, terminal configs, exit semantics and "it's just text" feel.
- **B. Run a GPU window** (for example Chromium `--kiosk --class=org.omarchy.screensaver` on a local copy of the page, or a native wgpu window). This is truer to the original, but heavier: a browser per monitor, GPU wake, a bigger attack surface, and it has to re-implement the launcher.

**Recommendation: A** for v1. B stays a possible later "hi-fi mode" (§11, Q1).

## 3. Goals and non-goals

**Goals**

1. It looks unmistakably like "A fly in the Matrix": rain, wireframe room, wireframe fly in flight, rotating connectome cloud, restrained HUD.
2. It behaves exactly like the stock screensaver: same window class, launch path, per-monitor instances, exit on key/mouse/focus-loss, cursor hide/restore, and co-operation with the idle-lock cycle.
3. It follows the active Omarchy theme and still works on light themes.
4. It uses little power: capped fps and bounded CPU per monitor (§8).
5. Installing it is one command and uninstalling it is clean. It survives `omarchy update`.
6. It credits Pragma Research / Cadence and BANC (CC BY).

**Non-goals (v1)**

- Running the Cadence brain or any learning. The motion is scripted and procedural.
- Interactivity (placing sugar, striking the fly). Any input exits.
- Supporting terminals other than the four Omarchy supports.
- Audio.

## 4. User experience

### 4.1 Scene layers (back to front)

| # | Layer | Terminal rendering | Notes |
|---|---|---|---|
| 0 | Background | Theme `background` (forced black on dark themes, like the stock saver) | OSC 11, same as the stock script |
| 1 | **Digital rain** | One glyph per cell. Half-width katakana `ｱ–ﾝ`, `0–9`, `Z:・=*+-<>`. The head is `bright_foreground`/`#d8ffe4`-ish, the trail fades from `accent` → `background` over 8–30 cells. | Column speeds vary. About 10 steps/s like the original, but interpolated so it runs smoothly at the render fps. Density is configurable. |
| 2 | **Wireframe room** | Braille sub-pixels (2×4 per cell) with depth-dimmed `accent`. Floor grid, three walls meeting in a corner, and a table with a banana and bread as low-poly wireframes. | The camera slowly orbits and dollies. Lines are depth-cued so they read in 3D. |
| 3 | **Connectome cloud** | Braille points. About 6–12k points downsampled from the BANC atlas, turning slowly on its long axis. Points "fire" (brighten to `bright_green`, then decay) in travelling waves. | It sits in the room like a hologram, or in a panel (Q3). The data is a pre-baked asset (§7.3). |
| 4 | **The fly** | A braille wireframe: thorax, abdomen, head with two large eyes, six legs, and wings drawn as two flickering ellipses (the stroke blur). About 10–15 % of screen height in close-up moments. | Behaviour is described in §4.2. It is the brightest object on screen. |
| 5 | **Branding (optional)** | The user's `screensaver.txt`, centred. Rain columns part around its bounding box, and it glows with `accent`. | Off by default for this saver (Q4). |
| 6 | **HUD** | Top-left title `A FLY IN THE MATRIX` in letter-spaced caps. A bottom-left status line such as `flying · 0.38 m/s · 132 cm up · 210 Hz · 150,802 neurons`. A footer credit: `after Cadence · Pragma Research · connectome: BANC 888 (CC BY)`. | Dim (`dark_foreground`). It can be hidden. Glyph-based "glitch" jitter fires every 20–60 s for 80–200 ms. |

The compound-eye mosaic and the close-up panel are **stretch goals** (§10, M4). They need extra framebuffers and add clutter on small monitors.

### 4.2 Fly behaviour (procedural, adapted from the original's instinct layer)

This is a small state machine with rates taken from the original ethogram:

- **Fly bout**, 5–15 s. The fly follows a smooth path (Catmull-Rom through random waypoints inside the room box) at about 0.3 m/s in scene scale. It makes **saccades** (quick yaw turns of about 90° in ~50 ms of scene time) at roughly 0.45 per second.
- **Land** on the table, the banana or the bread. The fly decelerates, pitches up and folds its wings.
- **Sit / groom**, 3–8 s. It sits still and occasionally rubs its forelegs or head (a leg-wiggle animation).
- **Feed** (rare), when it lands on the fruit: the proboscis extends.
- Then it takes off and starts the next bout.

**Camera modes**, cycled every 30–60 s with eased transitions:

- *follow*: behind the fly at a distance
- *room*: a wide orbit of the whole room
- *brain*: a slow push toward the connectome cloud

The random seed comes from the monitor index plus the time, so each monitor shows a different flight.

### 4.3 Multi-monitor

It runs as one process per monitor, the same as stock. Each monitor gets its own seed and camera phase. Syncing a shared flight across monitors is out of scope.

### 4.4 Exit and lifecycle contract (must match stock exactly)

1. The window class stays `org.omarchy.screensaver`. It's launched by the unmodified `omarchy-launch-screensaver`.
2. On start, the program:
   - sets the black background via OSC 11
   - hides the Hyprland cursor (`hyprctl eval 'hl.config({ cursor = { invisible = true } })'`, falling back to `hyprctl keyword cursor:invisible true`)
   - enters the alternate screen and hides the text cursor
   - enables mouse-motion reporting (DECSET 1003 + SGR 1006), so mouse movement arrives as stdin bytes
   - waits up to 2 s for the PTY to leave 80x24, then handles `SIGWINCH` for the rest of the run
3. **Exit triggers**:
   - any byte on stdin (key or mouse)
   - Hyprland focus leaving the window (polled every ≤250 ms, which is more responsive than stock's 1 s)
   - `SIGINT`, `SIGTERM`, `SIGHUP` or `SIGQUIT`
4. **On exit**, the program:
   - restores the terminal (leaves the alternate screen, turns off mouse reporting, shows the cursor, resets the background with OSC 111)
   - restores the Hyprland cursor
   - `pkill -f '[o]rg.omarchy.screensaver'` so every monitor's instance closes together (same as stock)
   - exits 0
5. Its process name must be something `omarchy-system-lock` can stop. Either it stops cleanly on `SIGTERM` within 100 ms, or (ideally) upstream adds `pkill -x flysaver` next to `ttfx` (§6).
6. It never grabs input, never inhibits idle, and never blocks the lock screen.

## 5. Configuration

The config file is `~/.config/omarchy/flysaver.toml`. It's optional, and every key has a default.

```toml
fps = 30                 # 15–60; 30 default for power
rain_density = 0.6       # 0–1 share of columns active
layers = ["rain", "room", "brain", "fly", "hud"]   # add "logo" to overlay screensaver.txt
brain_points = 8000      # 2k–20k
camera = "cycle"         # cycle | follow | room | brain
palette = "theme"        # theme | matrix (#39ff6a on black, like the original)
glitch = true
hud = true
```

Colour mapping from `colors.toml`, used when `palette = "theme"`:

| Role | Colour key |
|---|---|
| rain trail | `accent` |
| rain head | `bright_foreground` |
| wireframe | `accent` at 40–70 % |
| neuron fire | `bright_green` (falls back to `accent`) |
| fly | `foreground` |
| HUD | `dark_foreground` |

On `mode = "light"` themes, the saver still forces a dark background, because a screensaver should dim the room. Q5 asks whether you agree.

## 6. Integration with Omarchy

Since nothing can shadow `omarchy-screensaver`, there are three paths:

1. **Upstream hook (recommended long term).** Open a small PR against `basecamp/omarchy`. It would teach `omarchy-screensaver` to `exec ~/.config/omarchy/screensaver/run` when that file exists and is executable, and add `pkill` of the hooked process to `omarchy-system-lock`. This matches Omarchy's `hooks/*.d` convention. It's roughly 10 lines, with no behaviour change for anyone else.
2. **Local shim (v1 fallback).** Install a `post-update.d` hook that idempotently re-applies that same few-line patch to `$OMARCHY_PATH/bin/omarchy-screensaver` after every `omarchy update`, and skips with a notification if the upstream file's shape changed. Uninstalling reverts the patch.
3. **Manual trigger only.** Add a Hyprland bind or menu entry that launches the terminals ourselves with `-e flysaver`, and leave idle on the stock saver. This needs zero patching, but idle won't use the new saver.

**Proposal:** ship path 2 plus path 3 in v1, and open the upstream PR (path 1) in parallel. When path 1 lands, the installer detects it and drops the shim.

**Commands**

- `flysaver --preview` runs in the current terminal, without pkill or Hyprland calls.
- `flysaver install` and `flysaver uninstall`.
- `flysaver doctor` checks the terminal, truecolor, fonts and the hook state.

## 7. Technical design

### 7.1 Language and runtime

**Recommendation: Rust, as a single static binary**, with these dependencies:

- `crossterm` for terminal I/O
- `glam` for math
- `serde` and `toml` for config
- the connectome asset embedded via `include_bytes!`

Why Rust:

- the render loop is tight at 30 fps × up to 4 monitors
- there's no runtime to install
- it matches the stock `ttfx`, which is also a native binary

The alternative is pure-stdlib Python 3 (Arch ships it). That's simpler to hack on but uses roughly 5–10× more CPU. Q2 asks which you prefer.

The code is split into these modules:

| Module | Job |
|---|---|
| `term` | Setup and teardown, input watcher, SIGWINCH, OSC colours |
| `hypr` | Cursor hide/restore, focus poll via `hyprctl activewindow -j`, no-op in preview |
| `fb` | Cell framebuffer: per-cell glyph + fg/bg + a 2×4 braille sub-pixel mask with max-brightness per cell, diffed against the previous frame so only changed cells are written |
| `raster` | 3D → 2D projection, Bresenham/Wu lines into braille sub-pixels, depth cueing |
| `scene/rain`, `scene/room`, `scene/brain`, `scene/fly`, `scene/hud` | One module per layer |
| `sim` | Fly state machine and flight path, camera director, seeded RNG |
| `theme` | `colors.toml` parser and palette roles |

### 7.2 Rendering approach

- **Compositing.** Rain draws glyphs into cells. Vector layers draw braille dots. When a cell has both, the vector layer wins if its intensity is at least the rain's. Otherwise the rain glyph stays and gets tinted. This keeps the look of the original's rain showing through the wireframe.
- **Output.** Each frame is built into one buffer and written with a single `write`, wrapped in synchronized output (DEC mode 2026, which Ghostty, Kitty, Foot and Alacritty ≥0.13 support), so there's no tearing. Only changed cells are emitted, and colours are 24-bit.
- **Resolution.** At font 18 on a 2560×1440 panel, that's about 230×65 cells, or 460×260 braille sub-pixels. That's enough for a readable fly silhouette at close-up.
- **Frame pacing.** Frames are time-based (`dt`), so a slow terminal drops frames instead of slowing the simulation.

### 7.3 Connectome asset

- A one-off build script (not shipped at runtime) reads `fly-matrix/web/data/atlas.json`, or the BANC 888 `meta.feather` directly. It takes the 150,802 soma positions and region labels and makes a stratified downsample to 20k points per region, keeping the silhouette of the optic lobes, central brain and nerve cord. It quantises to `u16`×3 plus a `u8` region, giving about 140 KB, and embeds that in the binary.
- Runtime then subsamples to `brain_points`.
- Firing waves are procedural: region-seeded pulses that travel along the long axis. We say plainly (in `--about` and the README) that it's decorative, not simulated.
- Attribution strings for BANC (CC BY) and Cadence (MIT) ship in the binary and the README. The MIT notice from `cadence-examples` is included if any code or geometry is ported.

### 7.4 Fly model

- A hand-built low-poly wireframe of about 150 edges, informed by the original's `fly.js` proportions. It is not a copy, but it can be ported under MIT with the notice kept.
- Wings are two ellipses whose opacity flickers at a beat-frequency proxy.
- Legs animate with simple sine-based joint angles for walking and grooming.

## 8. Performance and power budget

| Metric | Target |
|---|---|
| CPU per instance at 30 fps, 230×65 cells | ≤ 5 % of one core on a 2023 laptop (Rust); stretch goal ≤ 3 % |
| Terminal throughput | ≤ 300 KB/s per instance average (diffed frames) |
| Memory | ≤ 30 MB RSS per instance |
| Startup to first frame | ≤ 150 ms after the PTY resize |
| Exit latency | key/mouse ≤ 50 ms; focus loss ≤ 300 ms |
| On battery | Auto-drop to 20 fps when `/sys/class/power_supply/*/status` is `Discharging` (configurable) |

## 9. Why not run the real Cadence brain?

- The browser brain needs the 26 MB `brain.json` sub-net, a worker, and roughly 7 ms per step for 60k neurons. That's a lot of CPU for a screensaver on every monitor, and it drains battery.
- The visual payoff in a terminal is marginal compared with procedural firing.
- If you want it anyway, a v2 "live" mode could run the `web/brain.js` engine under a headless JS runtime, or port it, and feed real activity to the cloud (Q6).

## 10. Milestones (after sign-off)

| M | Deliverable | Acceptance |
|---|---|---|
| M1 | Terminal shell: lifecycle contract (§4.4), rain layer, theme colours, `--preview` | Replacing the stock saver via the shim works on all four terminals and multi-monitor; the exit triggers all work; the idle→lock cycle is unaffected (the lock still fires at `idle.lock` while the saver is up, and dismissing cancels it) |
| M2 | Braille rasteriser, wireframe room, camera director | Steady 30 fps within the CPU budget at 4K |
| M3 | Connectome cloud (asset pipeline, firing waves), fly model plus behaviour state machine, HUD, credits | Side-by-side with the reference screenshot, a reviewer recognises the scene |
| M4 (stretch) | Compound-eye mosaic inset, close-up inset, glitch effects, logo layer | Toggleable via `layers` |
| M5 | Installer (`install`/`uninstall`/`doctor`), `post-update.d` hook, README, upstream PR for the hook | Clean install and uninstall on a fresh Omarchy; survives `omarchy update` |

**Testing:**

- Unit tests for the rasteriser, theme parsing and the state machine.
- Golden-frame snapshots: render a fixed seed at 120×40 to text and diff it.
- A manual checklist per terminal (Alacritty, Ghostty, Foot, Kitty) × single and dual monitor × dark and light theme.

**Where it lives:** a new top-level `flysaver/` directory in this repo. The repo is currently a LitServe/MCP starter, so a separate repo might be cleaner (Q7).

## 11. Open questions for sign-off

1. **Approach:** terminal-native port (recommended), or a GPU/Chromium kiosk "hi-fi" mode that runs close to the original page?
2. **Language:** Rust (recommended), or stdlib Python?
3. **Brain placement:** a hologram inside the room (recommended, one coherent scene), or a separate panel like the original's right-hand box?
4. **Omarchy logo:** off by default (the scene *is* the branding), or overlaid on the rain by default?
5. **Colours:** follow the Omarchy theme (recommended), or always use the original's Matrix green? Should light themes still get a dark saver?
6. **Real simulation:** skip it (recommended), or plan a v2 live-brain mode?
7. **Home:** `flysaver/` inside `smkalle/vibe1`, or a new repository?
8. **Integration:** is a `post-update.d` shim that patches `$OMARCHY_PATH` acceptable for v1, or do we wait for (or only do) the upstream hook PR?
9. **Name:** `flysaver`, or something else (for example `omarchy-flymatrix`)?

## 12. Sources

- Omarchy source: https://github.com/basecamp/omarchy (`bin/omarchy-launch-screensaver`, `bin/omarchy-screensaver`, `shell/plugins/services/idle/Service.qml`, `manual/13-toggles-idle-screensaver.md`)
- Omarchy manual, toggles, idle and screensaver: https://omarchy.org/manual/toggles-idle-screensaver/
- Omarchy screensaver page: https://omarchy.org/screensaver/
- Screensaver improvements discussion: https://github.com/omacom/omarchy/discussions/1527
- Custom screensaver write-up: https://yivanov.com/posts/creating-a-custom-screensaver-on-omarchy/
- Prior art: https://github.com/arkits/omarchy-screensaver, https://github.com/limehawk/tte-screensaver
- Fly in the Matrix (live): https://floatingpragma.io/cadence-examples/fly-matrix/
- Fly in the Matrix (source, MIT): https://github.com/Jarikononen/cadence-examples/tree/main/fly-matrix
- BANC connectome release 888 (CC BY), as cited in the example's README

## 13. Build notes (post sign-off)

These are the decisions from §11 as built: terminal-native, Rust, the brain as a hologram in the room, the logo off by default, theme colours on an always-dark background, no live simulation, `flysaver/` in this repo, and the name `flysaver`.

**§6 changed: a PATH shim, not a patch.** On a current Omarchy install `OMARCHY_PATH` is `/usr/share/omarchy`, owned by the `omarchy` pacman package. It's no longer a git checkout in `$HOME`. Patching `omarchy-screensaver` would need root and would fight pacman.

Instead, `flysaver install` does three things:

- writes `~/.config/uwsm/env.d/99-flysaver`, which puts `~/.local/share/flysaver/bin` first on the Hyprland session's `PATH` (Omarchy's `default/uwsm/env.d/10-omarchy` names `~/.config/uwsm/env.d/*` as the preferred place for user overrides)
- adds a single `omarchy-screensaver` shim in that directory, which runs flysaver or falls back to the stock one
- writes `~/.config/omarchy/screensaver/run`, so the upstream hook in `flysaver/upstream/` works unchanged if it lands

With the shim, no `post-update.d` hook is needed. `flysaver launch` covers path 3 until the next login. The upstream PR itself was not opened, because this session can't reach `basecamp/omarchy`. The patch is ready and applies cleanly at `c5b4db7`.

**§8 measured** at 230×65 cells, release build:

| Metric | Measured | Budget |
|---|---|---|
| CPU | 0.7–0.8 ms/frame, 2.2–2.3% of a core at 30 fps | ≤ 5%, met |
| Terminal output | 0.4–0.6 MB/s | ≤ 300 KB/s, **missed** |
| Exit | ≤ 2 ms on key/mouse; ~45 ms on focus loss | ≤ 50 / 300 ms, met |

The output budget is missed because the orbiting camera moves every wireframe line on every frame. Quantised brightness and relative cursor moves halved it, from 1.6 MB/s. Going further would mean choppier camera motion. GPU terminals handle this rate easily. `fps` and `battery_fps` are the knobs if it matters.

**Milestones.**

- M1–M3 and M5 are done.
- M4 is partly done: the glitches and the logo layer are in. The compound-eye and close-up insets are **not** built. They're still stretch goals.

**Tests.**

- Unit tests and golden frames: `cargo test`
- Exit and restore in a real PTY: `tests/pty_lifecycle.py`
- Focus, pointer and close-all against a fake Hyprland: `tests/hypr_contract.py`
- Install, shim, fallback and uninstall: `tests/install_roundtrip.sh`

Not yet verified on a real Omarchy/Hyprland desktop. This build environment has no display.
