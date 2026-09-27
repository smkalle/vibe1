# flysaver: A fly in the Matrix, as an Omarchy screensaver

A terminal screensaver for [Omarchy](https://omarchy.org). It's a take on
Pragma Research's [A fly in the Matrix](https://floatingpragma.io/cadence-examples/fly-matrix/)
drawn in truecolor text and braille. The scene has four parts:

- Matrix digital rain
- a green wireframe room with a table, a banana and a piece of bread
- a wireframe fruit fly that flies bouts, makes saccades, lands, grooms and feeds
- the fly's real nervous system as a turning hologram: 20,044 neurons sampled
  from the 150,802 in the BANC connectome, each drawn where it sits

![The nervous system view](screenshots/brain.png)
![Following the fly](screenshots/follow.png)
![The room](screenshots/room.png)

It uses the same windows, launcher and exit behaviour as the stock screensaver:

- any key or mouse movement exits
- switching focus away exits
- every monitor closes together
- the idle → lock timing is untouched

It follows your current Omarchy theme.

## Install

```bash
cd flysaver
cargo build --release
target/release/flysaver install
```

Then log out and back in. To try it right away:

```bash
flysaver launch     # every monitor, now
flysaver preview    # in this terminal
flysaver doctor     # check the setup
```

`install` touches nothing outside `$HOME` and needs no root:

| File | Purpose |
|---|---|
| `~/.local/bin/flysaver` | the binary: about 750 KB, static apart from libc, with the connectome embedded |
| `~/.local/share/flysaver/bin/omarchy-screensaver` | a shim that runs flysaver, or falls back to the stock screensaver |
| `~/.config/uwsm/env.d/99-flysaver` | puts the shim first on the Hyprland session's `PATH` |
| `~/.local/share/flysaver/flysaver-launch` | `flysaver launch`: Omarchy's launcher, pointed at flysaver |
| `~/.config/omarchy/screensaver/run` | ready for the [proposed upstream hook](upstream/omarchy-screensaver-user-hook.patch) |
| `~/.config/omarchy/flysaver.toml` | settings (kept by `uninstall`) |

Why a PATH shim: Omarchy runs its screensaver as `omarchy-screensaver` in a
terminal. It has no setting for swapping that program, and the stock script
is owned by the `omarchy` package. Putting a shim first on the session `PATH`
through `~/.config/uwsm/env.d/` changes no Omarchy file, and `omarchy update`
can't undo it. That directory is where Omarchy's own uwsm env tells users to
put overrides. `upstream/` has a small patch that would make this a
first-class hook.

- **Turn it off, keep it installed:** `touch ~/.config/omarchy/flysaver.disabled`
  (the shim then runs the stock screensaver).
- **Remove it:** `flysaver uninstall`, then log out and back in.

## Settings

`~/.config/omarchy/flysaver.toml`. Every key is optional; see
[flysaver.example.toml](flysaver.example.toml).

| Key | Default | |
|---|---|---|
| `fps` | `30` | 5–60 |
| `battery_fps` | `20` | used while any battery is discharging |
| `rain_density` | `0.45` | share of columns raining |
| `rain_glyphs` | `"katakana"` | or `"ascii"` if your fonts lack half-width katakana |
| `layers` | `["rain","room","brain","fly","hud"]` | add `"logo"` to show your `screensaver.txt` with the rain parting around it |
| `brain_points` | `8000` | 500–20000 neurons |
| `camera` | `"cycle"` | `follow`, `room`, `brain`, or `cycle` through them every 30–60 s |
| `palette` | `"theme"` | the current Omarchy theme, or `"matrix"` for the original's `#39ff6a` |
| `glitch` | `true` | the occasional stutter and jitter |
| `hud` | `true` | title, status line, credits |

The same options work as flags: `flysaver preview --camera brain --palette matrix`.

## Performance

These numbers come from `flysaver bench --size 230x65`, which is about a
2560×1440 screen at Omarchy's screensaver font size.

| | Measured |
|---|---|
| CPU | 0.7–0.8 ms per frame, which is 2.2–2.3% of one core at 30 fps |
| Output to the terminal | 0.4–0.6 MB/s. Only changed cells are written, in one synchronized update per frame. |
| Exit | under 2 ms after a key or mouse movement; about 45 ms after focus leaves |

## How it works

| File | Role |
|---|---|
| `src/sim.rs` | The fly's life as a state machine: fly, approach a smell, land, sit, groom, feed, take off. Rates follow the original's ethogram (about 0.45 saccades per second of flight, 5–15 s bouts). Also the camera director. |
| `src/scene/` | One module per layer. `brain.rs` lights regions to match what the fly is doing: saccades light the optic lobes, take-off lights the wing motor neurons and descending neurons, feeding lights the antennae. A wave also runs from the brain down the nerve cord. |
| `src/raster.rs`, `src/fb.rs` | Perspective projection and lines drawn into braille dots (2×4 per cell). The layers are composed into cells and written to the terminal as a diff. |
| `src/term.rs`, `src/hypr.rs` | Raw mode, mouse-motion reporting, signals, Hyprland pointer and focus handling, and the close-every-monitor exit. |
| `tools/build_connectome.py` | Builds `assets/connectome.bin` (140 KB) from the original's `atlas.json`. |

Nothing is simulated neurally. The firing is decorative, as `flysaver about`
says.

## Tests

```bash
cargo test --release                       # unit tests + golden frames
python3 tests/pty_lifecycle.py             # exits and terminal restore in a real pty
python3 tests/hypr_contract.py             # focus-loss exit, pointer, close-all, against a fake Hyprland
tests/install_roundtrip.sh                 # install, shim, fallback, uninstall in a scratch $HOME
NODE_PATH=$(npm root -g) node tools/shoot.cjs out.png < <(flysaver snapshot --html --camera brain)
```

After an intentional visual change, regenerate the golden frames with
`tests/golden/regen.sh`.

## Credits

- After *A fly in the Matrix* by Bernhard Mueller / Pragma Research, a
  [Cadence](https://floatingpragma.io/cadence/) example. The source is
  [github.com/Jarikononen/cadence-examples](https://github.com/Jarikononen/cadence-examples)
  under the MIT licence. No code was copied. The scene, the palette and the
  behaviour rates follow it.
- The neuron positions come from BANC release 888 (adult female *Drosophila*
  brain and nerve cord), by the Lee lab and the BANC community, CC BY 4.0.
  They were sampled from the atlas that ships with the example above.
- The launcher script follows Omarchy's `omarchy-launch-screensaver`
  (Omarchy, MIT).
