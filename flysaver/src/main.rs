//! flysaver: "A fly in the Matrix" as an Omarchy screensaver.

// Rasteriser helpers take coordinates, intensities and a colour; a struct per call would not help.
#![allow(clippy::too_many_arguments)]

mod config;
mod fb;
mod hypr;
mod install;
mod learner;
mod math;
mod memory;
mod neuro;
mod raster;
mod rng;
mod scene;
mod senses;
mod sim;
mod term;
mod theme;

use config::{Colors, Config, Palette};
use fb::{Frame, Screen};
use scene::Scene;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use theme::Theme;

pub const DEFAULT_CONFIG: &str = include_str!("../flysaver.example.toml");

const USAGE: &str = "\
flysaver - A fly in the Matrix, as an Omarchy screensaver

usage:
  flysaver                 run as the screensaver (what Omarchy's launcher starts)
  flysaver preview         run in this terminal; any key or mouse movement exits
  flysaver launch          start the screensaver on every monitor now
  flysaver install         install for the current user (no root, survives omarchy update)
  flysaver uninstall       remove everything install added (keeps your config)
  flysaver doctor          check the installation and the session
  flysaver train           run the fly headless and log its lessons (--minutes N, --save)
  flysaver forget          erase what the fly has learned
  flysaver snapshot        print one frame as text (for tests)
  flysaver bench           time the renderer
  flysaver about           credits and licences

options (override ~/.config/omarchy/flysaver.toml):
  --fps N  --camera cycle|follow|room|brain  --palette theme|theme-strict|matrix  --colors auto|truecolor|256
  --layers rain,room,brain,fly,hud,logo  --seed N
  --size COLSxROWS  --time SECONDS  --frames N  --html   (snapshot/bench)
  --swat SECONDS    snapshot: perch the fly, send the swatter, draw SECONDS later
  --taste BITTER    snapshot: land on the sugared bread laced with BITTER (0-1), draw --time later
";

const ABOUT: &str = "\
flysaver - A fly in the Matrix, as an Omarchy screensaver.

After \"A fly in the Matrix\" by Bernhard Mueller / Pragma Research, a Cadence
example: https://floatingpragma.io/cadence-examples/fly-matrix/
(source: github.com/Jarikononen/cadence-examples, MIT licence).

The connectome is BANC release 888 (adult female Drosophila brain and
nerve cord, 150,802 neurons), by the Lee lab and the BANC community,
CC BY 4.0. The silhouette is a 20,044-neuron sample of it.

What is simulated (brain = \"live\", the default): the 60,000-neuron sub-net
that \"A fly in the Matrix\" settles, 1,209,528 synapse classes wired as
measured, running the Cadence graded rate model with the library's numbers
(held to cadence.Brain's own outputs to 1e-9 by the parity test). The drawn
activity is that model's.

What the brain decides (pilot = \"brain\", the default): as in the original's
brain layer, its output neurons, read against their level-flight rest,
override the instincts. DNa02 left/right turn the fly, DNp09 sets its speed,
DNp07/DNp10 land it, the giant fibre fires an escape from the swatter, MN9
feeds and aDN grooms, and over a fruit the mushroom body's MBON11 and MBON05
decide approach or avoid (a softmax, T 0.3). Under these senses the speed
and grooming neurons stay silent, so the instincts carry those.

What it tastes (bitter = true, the default): landing on sugar makes MN9
burst, the proboscis (amber) extends and the fly feeds. Sometimes the sugar
is laced with caffeine: bitter suppresses the burst, the proboscis turns
violet and stays in, and the PPL1 punishment dopamine teaches avoidance.

What it learns (learning = true, the default): each decision over a fruit is
a lesson. Sugar where it lands is +1, an empty fruit left or avoided 0, the
swatter striking it there -1. The library's actor-critic (the original's
learner.js, held to it by a parity test) moves the 16,795 Kenyon-cell ->
MBON synapses. The sugar moves every sugar_minutes, and what the fly has
learned is kept in ~/.local/state/flysaver/memory.bin between idle sessions
(flysaver forget erases it).

What is supplied: the instinct layer (flight bouts, saccades, wall
avoidance, sits), the senses as the original page feeds them (halteres,
ocelli, optic flow, antennae, odours, sugar, touch, dopamine), the swatter
as the looming threat, and the body's kinematics.

The eyes are red: a mutant in the Matrix, where green is normal. (In real
Drosophila, red is the wild type; Morgan's 1910 mutant was white-eyed.)
";

struct Opts {
    cmd: String,
    seed: Option<u64>,
    size: (usize, usize),
    time: f32,
    frames: usize,
    html: bool,
    swat: Option<f32>,
    minutes: f32,
    save: bool,
    taste: Option<f32>,
}

fn parse_args(cfg: &mut Config) -> Result<Opts, String> {
    let mut o = Opts { cmd: "run".into(), seed: None, size: (120, 40), time: 12.0, frames: 300, html: false, swat: None, minutes: 10.0, save: false, taste: None };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "-h" | "--help" | "help" => o.cmd = "help".into(),
            "--preview" => o.cmd = "preview".into(),
            "--about" => o.cmd = "about".into(),
            "--fps" => cfg.fps = val("--fps")?.parse::<f32>().map_err(|e| e.to_string())?.clamp(5.0, 60.0),
            "--camera" => cfg.camera = config::parse_camera(&val("--camera")?).ok_or("unknown camera")?,
            "--palette" => {
                cfg.palette = Palette::parse(&val("--palette")?).ok_or("palette is theme, theme-strict or matrix")?
            }
            "--colors" => cfg.colors = Colors::parse(&val("--colors")?).ok_or("colors is auto, truecolor or 256")?,
            "--layers" => cfg.layers = val("--layers")?.split(',').map(|s| s.trim().to_string()).collect(),
            "--seed" => o.seed = Some(val("--seed")?.parse().map_err(|_| "bad seed")?),
            "--size" => {
                let v = val("--size")?;
                let (c, r) = v.split_once('x').ok_or("size is COLSxROWS")?;
                o.size = (c.parse().map_err(|_| "bad cols")?, r.parse().map_err(|_| "bad rows")?);
            }
            "--html" => o.html = true,
            "--minutes" => o.minutes = val("--minutes")?.parse().map_err(|_| "bad minutes")?,
            "--save" => o.save = true,
            "--taste" => o.taste = Some(val("--taste")?.parse().map_err(|_| "bad bitter level")?),
            "--swat" => o.swat = Some(val("--swat")?.parse().map_err(|_| "bad swat time")?),
            "--time" => o.time = val("--time")?.parse().map_err(|_| "bad time")?,
            "--frames" => o.frames = val("--frames")?.parse().map_err(|_| "bad frames")?,
            c if !c.starts_with('-') => o.cmd = c.to_string(),
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(o)
}

fn default_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    t ^ ((std::process::id() as u64) << 32)
}

fn on_battery() -> bool {
    std::fs::read_dir("/sys/class/power_supply")
        .map(|d| d.flatten().any(|e| std::fs::read_to_string(e.path().join("status")).is_ok_and(|s| s.trim() == "Discharging")))
        .unwrap_or(false)
}

fn main() {
    let (mut cfg, warnings) = config::load();
    let opts = match parse_args(&mut cfg) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("flysaver: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let seed = opts.seed.unwrap_or_else(default_seed);
    let code = match opts.cmd.as_str() {
        "run" => run(cfg, seed, false),
        "preview" => run(cfg, seed, true),
        "snapshot" => {
            print!("{}", match (opts.swat, opts.taste) {
                (Some(after), _) => swat_snapshot(cfg, seed, opts.size, after, opts.html),
                (None, Some(bitter)) => taste_snapshot(cfg, seed, opts.size, bitter, opts.time, opts.html),
                (None, None) => snapshot(cfg, seed, opts.size, opts.time, opts.html),
            });
            0
        }
        "bench" => bench(cfg, seed, opts.size, opts.frames),
        "train" => train(cfg, seed, opts.minutes, opts.save),
        "forget" => {
            let p = memory::path();
            match std::fs::remove_file(&p) {
                Ok(()) => println!("forgot everything: removed {}", p.display()),
                Err(_) => println!("nothing to forget ({} does not exist)", p.display()),
            }
            0
        }
        "launch" => launch(),
        "install" => report(install::install()),
        "uninstall" => report(install::uninstall()),
        "doctor" => {
            for w in &warnings {
                println!("[warn] {w}");
            }
            if install::doctor() { 0 } else { 1 }
        }
        "about" => {
            print!("{ABOUT}");
            0
        }
        "help" => {
            print!("{USAGE}");
            0
        }
        other => {
            eprintln!("flysaver: unknown command {other}\n\n{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

fn report(r: std::io::Result<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("flysaver: {e}");
            1
        }
    }
}

fn launch() -> i32 {
    let script = install::launch_path();
    let me = std::env::current_exe().ok();
    let mut cmd = if script.exists() {
        std::process::Command::new(&script)
    } else {
        eprintln!("flysaver: {} missing, run `flysaver install` first", script.display());
        return 1;
    };
    if let Some(me) = me {
        cmd.env("FLYSAVER_BIN", me);
    }
    match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("flysaver: {e}");
            1
        }
    }
}

/// Deterministic headless frame, used by the golden tests. Text snapshots are
/// characters only, so they always compose in truecolor and do not depend on
/// the terminal; HTML snapshots honour the colour mode.
pub fn snapshot(cfg: Config, seed: u64, (cols, rows): (usize, usize), secs: f32, html: bool) -> String {
    let mut cfg = cfg;
    cfg.glitch = false;
    let colors = if html { cfg.colors.resolve_env() } else { Colors::TrueColor };
    let theme = if html { Theme::load(cfg.palette) } else { Theme::matrix() };
    let mut scene = Scene::new(cfg, theme, seed);
    scene.show_timing = false;
    let mut f = Frame::new(cols, rows).with_colors(colors);
    let dt = 1.0 / 30.0;
    for _ in 0..(secs / dt) as usize {
        scene.step(dt, cols, rows);
    }
    scene.draw(&mut f);
    if html { f.to_html() } else { f.to_text() }
}

/// A demo frame: the fly perched on the bread, a swatter sent at it, the frame
/// `after` seconds later (README screenshots of the giant-fibre escape).
fn swat_snapshot(cfg: Config, seed: u64, (cols, rows): (usize, usize), after: f32, html: bool) -> String {
    let mut cfg = cfg;
    cfg.glitch = false;
    cfg.threats = false;
    let colors = if html { cfg.colors.resolve_env() } else { Colors::TrueColor };
    let theme = if html { Theme::load(cfg.palette) } else { Theme::matrix() };
    let mut scene = Scene::new(cfg, theme, seed);
    scene.show_timing = false;
    scene.fly.perch(sim::Spot::Bread);
    let dt = 1.0 / 30.0;
    for _ in 0..30 {
        scene.step(dt, cols, rows);
    }
    scene.threat = Some(sim::Threat::aimed_at(scene.fly.pos, &mut rng::Rng::new(seed)));
    for _ in 0..(after / dt) as usize {
        scene.step(dt, cols, rows);
    }
    let mut f = Frame::new(cols, rows).with_colors(colors);
    scene.draw(&mut f);
    if html { f.to_html() } else { f.to_text() }
}

/// Headless: the whole scene with lessons for `minutes` of screensaver time, logging what
/// the mushroom body decides and learns (the end-to-end check of Phase 3).
fn train(cfg: Config, seed: u64, minutes: f32, save: bool) -> i32 {
    let mut cfg = cfg;
    cfg.glitch = false;
    let mut scene = Scene::new(cfg, Theme::matrix(), seed);
    scene.show_timing = false;
    scene.start_memory(save.then(memory::path));
    let dt = 1.0 / 30.0;
    let frames = (minutes * 60.0 / dt) as usize;
    let mut last: Option<(String, f32)> = None;
    let mut decisions: [Vec<f32>; 2] = [vec![], vec![]];
    let mut seen = [f32::NAN; 2];
    println!("training {minutes} min, sugar on the {}{}", scene.sugar.name(), if save { ", with the saved memory" } else { "" });
    let start = Instant::now();
    for _ in 0..frames {
        scene.step(dt, 80, 24);
        if scene.note != last {
            if let Some((text, t)) = &scene.note {
                let keep = ["mushroom body", "lesson", "sugar", "giant fibre", "swatted", "remembers"];
                if keep.iter().any(|k| text.contains(k)) {
                    println!("{:>6.1}s  {text}", t);
                }
            }
            last = scene.note.clone();
        }
        for (k, p) in scene.stats.last_p.iter().enumerate() {
            if !p.is_nan() && p.to_bits() != seen[k].to_bits() {
                decisions[k].push(*p);
                seen[k] = *p;
            }
        }
    }
    let st = scene.stats;
    println!(
        "done in {:.0} s: {} lessons ({} sugar, {} blows), {} of the 16,795 synapses changed; sugar on the {}",
        start.elapsed().as_secs_f32(),
        st.lessons,
        st.rewards,
        st.blows,
        st.changed,
        scene.sugar.name()
    );
    let mean = |v: &[f32]| if v.is_empty() { f32::NAN } else { v.iter().sum::<f32>() / v.len() as f32 };
    for (k, name) in ["banana", "bread"].iter().enumerate() {
        let d = &decisions[k];
        let n = d.len().min(3);
        println!(
            "approach p, {name:<6}  first {n} decisions {:.2}  ->  last {n} {:.2}   ({} decisions)",
            mean(&d[..n]),
            mean(&d[d.len() - n..]),
            d.len()
        );
    }
    if save {
        scene.save_memory();
    }
    0
}

/// A demo frame: the fly lands on the sugared bread laced with `bitter`, drawn `after`
/// seconds later (README screenshots of the proboscis extension reflex).
fn taste_snapshot(cfg: Config, seed: u64, (cols, rows): (usize, usize), bitter: f32, after: f32, html: bool) -> String {
    let mut cfg = cfg;
    cfg.glitch = false;
    cfg.threats = false;
    let colors = if html { cfg.colors.resolve_env() } else { Colors::TrueColor };
    let theme = if html { Theme::load(cfg.palette) } else { Theme::matrix() };
    let mut scene = Scene::new(cfg, theme, seed);
    scene.show_timing = false;
    scene.sugar = sim::Spot::Bread;
    scene.bitter = bitter.clamp(0.0, 1.0);
    scene.fly.pos = math::v3(1.2, 1.3, 1.2);
    let dt = 1.0 / 30.0;
    for _ in 0..40 {
        scene.step(dt, cols, rows);
    }
    scene.fly.perch(sim::Spot::Bread);
    scene.fly.hunger = 0.8;
    for _ in 0..(after / dt) as usize {
        scene.step(dt, cols, rows);
    }
    let mut f = Frame::new(cols, rows).with_colors(colors);
    scene.draw(&mut f);
    if html { f.to_html() } else { f.to_text() }
}

fn bench(cfg: Config, seed: u64, (cols, rows): (usize, usize), frames: usize) -> i32 {
    let fps = cfg.fps;
    let colors = cfg.colors.resolve_env();
    let theme = Theme::load(cfg.palette);
    let mut scene = Scene::new(cfg, theme, seed);
    let mut f = Frame::new(cols, rows).with_colors(colors);
    let mut screen = Screen::new();
    let dt = 1.0 / fps;
    let mut bytes = 0usize;
    // Warm up past the first full-screen paint.
    scene.step(dt, cols, rows);
    scene.draw(&mut f);
    screen.render(&f);
    let start = Instant::now();
    for _ in 0..frames {
        scene.step(dt, cols, rows);
        scene.draw(&mut f);
        bytes += screen.render(&f).len();
    }
    let per = start.elapsed().as_secs_f64() / frames as f64;
    println!("{cols}x{rows} cells, {frames} frames, {} colours", colors.name());
    println!("  {:.2} ms/frame  -> {:.1}% of one core at {fps} fps", per * 1e3, per * fps as f64 * 100.0);
    println!("  {:.1} KB/frame  -> {:.0} KB/s to the terminal", bytes as f64 / frames as f64 / 1024.0, bytes as f64 / frames as f64 * fps as f64 / 1024.0);
    0
}

fn run(cfg: Config, seed: u64, preview: bool) -> i32 {
    if !term::is_tty() {
        eprintln!("flysaver: needs a terminal (try `flysaver preview` in one)");
        return 1;
    }
    term::install_signals();
    let hypr_on = !preview && hypr::available();
    if term::enter().is_err() {
        return 1;
    }
    if hypr_on {
        hypr::cursor_invisible(true);
    }
    if !preview {
        term::wait_for_resize(Duration::from_secs(2));
    }

    let base_fps = cfg.fps;
    let battery_fps = cfg.battery_fps;
    let colors = cfg.colors.resolve_env();
    let theme = Theme::load(cfg.palette);
    let mut scene = Scene::new(cfg, theme, seed);
    scene.start_memory(Some(memory::path()));
    let mut size = term::size();
    scene.aspect = size.aspect();
    let mut frame = Frame::new(size.cols, size.rows).with_colors(colors);
    let mut screen = Screen::new();
    let mut watch = hypr_on.then(hypr::FocusWatch::new);
    let mut fps = if on_battery() { battery_fps.min(base_fps) } else { base_fps };
    let mut power_check = Instant::now();
    let mut size_check = Instant::now();
    let mut last = Instant::now();

    'outer: loop {
        if term::STOP.load(Ordering::SeqCst) {
            break;
        }
        if term::RESIZED.swap(false, Ordering::SeqCst) || size_check.elapsed() > Duration::from_secs(1) {
            size_check = Instant::now();
            let s = term::size();
            if s != size {
                size = s;
                scene.aspect = size.aspect();
                frame = Frame::new(size.cols, size.rows).with_colors(colors);
            }
        }
        if power_check.elapsed() > Duration::from_secs(30) {
            power_check = Instant::now();
            fps = if on_battery() { battery_fps.min(base_fps) } else { base_fps };
        }

        let start = Instant::now();
        let dt = (start - last).as_secs_f32();
        last = start;
        scene.step(dt, frame.cols, frame.rows);
        scene.draw(&mut frame);
        if term::write_all(screen.render(&frame).as_bytes()).is_err() {
            break; // terminal went away
        }

        let deadline = start + Duration::from_secs_f32(1.0 / fps);
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let ms = (deadline - now).as_millis() as i32 + 1;
            let (input, events) = term::wait(watch.as_ref().and_then(|w| w.fd()), ms);
            if input {
                break 'outer; // any key or mouse movement
            }
            if events {
                if let Some(w) = &mut watch {
                    w.read_events();
                }
            }
            if term::STOP.load(Ordering::SeqCst) {
                break 'outer;
            }
            if term::RESIZED.load(Ordering::SeqCst) {
                break;
            }
        }
        if let Some(w) = &mut watch {
            if w.lost() {
                break;
            }
        }
    }

    scene.save_memory();
    term::leave();
    if hypr_on {
        hypr::cursor_invisible(false);
        hypr::close_all();
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_deterministic_and_sized() {
        let a = snapshot(Config::default(), 42, (80, 24), 3.0, false);
        let b = snapshot(Config::default(), 42, (80, 24), 3.0, false);
        assert_eq!(a, b);
        assert_eq!(a.lines().count(), 24);
        assert!(a.lines().all(|l| l.chars().count() == 80));
        assert!(a.contains("F L Y   I N"));
        assert!(a.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)), "no braille drawn");
    }

    /// Golden frames: the same seed must draw the same picture. After an
    /// intentional visual change, regenerate with tests/golden/regen.sh.
    #[test]
    fn golden_frames() {
        let cases: [(&str, &str, &str); 3] = [
            ("follow", include_str!("../tests/golden/follow_s7_100x30_t6.txt"), "follow"),
            ("room", include_str!("../tests/golden/room_s7_100x30_t6.txt"), "room"),
            ("brain", include_str!("../tests/golden/brain_s7_100x30_t6.txt"), "brain"),
        ];
        for (name, want, cam) in cases {
            let mut cfg = Config::default();
            cfg.camera = config::parse_camera(cam).unwrap();
            let got = snapshot(cfg, 7, (100, 30), 6.0, false);
            assert!(got == want, "golden frame '{name}' changed; run tests/golden/regen.sh if intended");
        }
    }

    #[test]
    fn default_config_file_parses_cleanly() {
        let mut c = Config::default();
        assert!(c.apply(&config::parse_flat_toml(DEFAULT_CONFIG)).is_empty());
    }
}
