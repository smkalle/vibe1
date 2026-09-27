//! `flysaver install | uninstall | doctor`.
//!
//! Omarchy runs its screensaver as `omarchy-screensaver` inside a terminal and
//! offers no hook to swap it, and on a packaged install its files are owned by
//! pacman. So instead of patching Omarchy, install puts a one-file shim dir at
//! the front of the Hyprland session's PATH via ~/.config/uwsm/env.d, the place
//! Omarchy's own uwsm env tells users to put overrides. Nothing outside $HOME
//! is touched, and `omarchy update` cannot undo it.

use crate::config::{self, home};
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHIM: &str = include_str!("../scripts/omarchy-screensaver");
const LAUNCH: &str = include_str!("../scripts/flysaver-launch");
const MARKER: &str = "flysaver";

pub fn bin_path() -> PathBuf {
    home().join(".local/bin/flysaver")
}
fn share() -> PathBuf {
    home().join(".local/share/flysaver")
}
fn shim_dir() -> PathBuf {
    share().join("bin")
}
fn shim_path() -> PathBuf {
    shim_dir().join("omarchy-screensaver")
}
pub fn launch_path() -> PathBuf {
    share().join("flysaver-launch")
}
fn env_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".config"));
    base.join("uwsm/env.d/99-flysaver")
}
/// Where an upstream Omarchy "user screensaver" hook would look (see upstream/).
fn hook_path() -> PathBuf {
    home().join(".config/omarchy/screensaver/run")
}

fn env_file() -> String {
    format!(
        "# Added by {MARKER} install; remove with `flysaver uninstall`.\n\
         # Puts the flysaver shim ahead of Omarchy's omarchy-screensaver in the Hyprland session.\n\
         case \":$PATH:\" in\n  *\":$HOME/.local/share/flysaver/bin:\"*) ;;\n  *) export PATH=\"$HOME/.local/share/flysaver/bin:$PATH\" ;;\nesac\n"
    )
}

fn hook_file() -> String {
    format!("#!/bin/bash\n# Added by {MARKER} install.\nexec \"$HOME/.local/bin/flysaver\" \"$@\"\n")
}

fn write_exec(path: &Path, body: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    // Write beside and rename, so a running copy is never truncated under itself.
    let tmp = path.with_extension("new");
    fs::write(&tmp, body)?;
    fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755))?;
    fs::rename(&tmp, path)
}

fn ours(path: &Path) -> bool {
    fs::read(path).map(|b| String::from_utf8_lossy(&b).contains(MARKER)).unwrap_or(false)
}

pub fn install() -> io::Result<()> {
    let me = std::env::current_exe()?;
    let bin = bin_path();
    if fs::canonicalize(&me).ok() != fs::canonicalize(&bin).ok() {
        write_exec(&bin, &fs::read(&me)?)?;
        println!("installed   {}", bin.display());
    }
    write_exec(&shim_path(), SHIM.as_bytes())?;
    println!("installed   {}", shim_path().display());
    write_exec(&launch_path(), LAUNCH.as_bytes())?;
    println!("installed   {}", launch_path().display());

    let env = env_path();
    fs::create_dir_all(env.parent().unwrap())?;
    fs::write(&env, env_file())?;
    println!("installed   {}", env.display());

    let hook = hook_path();
    if !hook.exists() {
        write_exec(&hook, hook_file().as_bytes())?;
        println!("installed   {} (for the proposed upstream hook)", hook.display());
    }

    if !config::config_path().exists() {
        fs::create_dir_all(config::config_path().parent().unwrap())?;
        fs::write(config::config_path(), crate::DEFAULT_CONFIG)?;
        println!("installed   {}", config::config_path().display());
    }

    println!();
    println!("Log out and back in so the Hyprland session picks up the new PATH.");
    println!("Until then:   flysaver launch    (all monitors, now)");
    println!("              flysaver preview   (this terminal)");
    println!("Check setup:  flysaver doctor");
    Ok(())
}

pub fn uninstall() -> io::Result<()> {
    for p in [env_path(), shim_path(), launch_path()] {
        if p.exists() {
            fs::remove_file(&p)?;
            println!("removed     {}", p.display());
        }
    }
    let _ = fs::remove_dir(shim_dir());
    let _ = fs::remove_dir(share());
    if ours(&hook_path()) {
        fs::remove_file(hook_path())?;
        println!("removed     {}", hook_path().display());
        let _ = fs::remove_dir(hook_path().parent().unwrap());
    }
    let bin = bin_path();
    if bin.exists() {
        fs::remove_file(&bin)?;
        println!("removed     {}", bin.display());
    }
    println!("kept        {} (delete it yourself if you like)", config::config_path().display());
    println!("\nLog out and back in to drop the PATH shim from the running session.");
    Ok(())
}

fn capture(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// PATH of the running Hyprland session, read from its /proc environ.
fn session_path() -> Option<String> {
    for entry in fs::read_dir("/proc").ok()?.flatten() {
        let comm = fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
        let comm = comm.trim();
        if comm == "Hyprland" || comm == ".Hyprland-wrapped" {
            let env = fs::read(entry.path().join("environ")).ok()?;
            return env
                .split(|b| *b == 0)
                .filter_map(|kv| std::str::from_utf8(kv).ok())
                .find_map(|kv| kv.strip_prefix("PATH=").map(str::to_string));
        }
    }
    None
}

fn resolve(path: &str, cmd: &str) -> Option<PathBuf> {
    path.split(':').filter(|d| !d.is_empty()).map(|d| Path::new(d).join(cmd)).find(|p| {
        fs::metadata(p).map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0).unwrap_or(false)
    })
}

pub fn doctor() -> bool {
    let mut ok = true;
    let mut check = |good: bool, warn_only: bool, msg: String| {
        let mark = if good { "ok  " } else if warn_only { "warn" } else { "FAIL" };
        if !good && !warn_only {
            ok = false;
        }
        println!("[{mark}] {msg}");
    };

    let bin = bin_path();
    check(bin.exists(), false, format!("binary at {}", bin.display()));
    check(shim_path().exists(), false, format!("PATH shim at {}", shim_path().display()));
    check(env_path().exists(), false, format!("session env at {}", env_path().display()));

    match session_path() {
        Some(p) => {
            let found = resolve(&p, "omarchy-screensaver");
            let shimmed = found.as_deref() == Some(shim_path().as_path());
            check(
                shimmed,
                false,
                match &found {
                    Some(f) if shimmed => format!("Hyprland session runs {}", f.display()),
                    Some(f) => format!("Hyprland session still runs {} (log out and back in)", f.display()),
                    None => "omarchy-screensaver not on the Hyprland session PATH".into(),
                },
            );
        }
        None => check(false, true, "no running Hyprland session found".into()),
    }

    let term = capture("xdg-terminal-exec", &["--print-id"]).unwrap_or_default();
    let supported = ["Alacritty", "ghostty", "foot", "kitty"].iter().any(|t| term.contains(t));
    check(supported, false, format!("default terminal '{term}' is one Omarchy's screensaver supports (Alacritty, Ghostty, Foot, Kitty)"));

    let kana = capture("fc-list", &[":charset=ff71", "family"]).is_some_and(|s| !s.is_empty());
    check(kana, true, "a font with half-width katakana (else set rain_glyphs = \"ascii\")".into());
    let braille = capture("fc-list", &[":charset=2800", "family"]).is_some_and(|s| !s.is_empty());
    check(braille, true, "a font with braille patterns".into());

    for tool in ["hyprctl", "socat", "jq"] {
        check(capture("which", &[tool]).is_some(), tool != "hyprctl", format!("{tool} on PATH"));
    }

    let colors = home().join(".local/state/omarchy/current/theme/colors.toml");
    check(colors.exists(), true, format!("theme colours at {}", colors.display()));
    let brain = crate::neuro::Brain::load();
    check(
        brain.weights_fnv.0 == brain.weights_fnv.1,
        false,
        format!(
            "live brain: {} neurons, {} synapse classes, weights rebuilt exactly (fnv {:016x})",
            brain.n, brain.edges, brain.weights_fnv.0
        ),
    );
    let (cfg0, _) = config::load();
    check(true, true, format!("palette: {}", crate::theme::Theme::load(cfg0.palette).origin));

    let (_, warnings) = config::load();
    let cfg_state = if !config::config_path().exists() {
        "not present, using defaults".to_string()
    } else if warnings.is_empty() {
        "parses".to_string()
    } else {
        warnings.join("; ")
    };
    check(warnings.is_empty(), true, format!("config {} {cfg_state}", config::config_path().display()));

    let (cfg, _) = config::load();
    let resolved = cfg.colors.resolve_env();
    check(
        true,
        true,
        format!(
            "colours: {} (COLORTERM={} in this shell; with auto, the screensaver's own terminal decides)",
            if cfg.colors == config::Colors::Auto { format!("auto -> {}", resolved.name()) } else { cfg.colors.name().to_string() },
            std::env::var("COLORTERM").unwrap_or_else(|_| "unset".into())
        ),
    );

    let off = home().join(".local/state/omarchy/toggles/screensaver-off");
    check(!off.exists(), true, "idle screensaver is enabled (omarchy toggle screensaver)".into());
    let disabled = home().join(".config/omarchy/flysaver.disabled");
    check(!disabled.exists(), true, format!("flysaver not disabled ({})", disabled.display()));
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_file_prepends_once() {
        let e = env_file();
        assert!(e.contains("export PATH=\"$HOME/.local/share/flysaver/bin:$PATH\""));
        assert!(e.contains(MARKER));
    }

    #[test]
    fn resolve_finds_first_executable() {
        let dir = std::env::temp_dir().join(format!("flysaver-test-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        write_exec(&b.join("tool"), b"#!/bin/sh\n").unwrap();
        fs::write(a.join("tool"), "not executable").unwrap();
        let path = format!("{}:{}", a.display(), b.display());
        assert_eq!(resolve(&path, "tool"), Some(b.join("tool")));
        fs::remove_dir_all(&dir).unwrap();
    }
}
