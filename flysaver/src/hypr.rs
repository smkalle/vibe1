//! Hyprland: hide/restore the pointer, notice when the screensaver loses focus,
//! and close every monitor's screensaver on exit - the same contract as
//! Omarchy's own omarchy-screensaver script.

use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const CLASS: &str = "org.omarchy.screensaver";
/// Monitors get their screensavers one after another; focus hops around meanwhile.
const LAUNCH_GRACE: Duration = Duration::from_millis(1500);
const POLL_EVERY: Duration = Duration::from_millis(1000);

fn quiet(cmd: &mut Command) -> bool {
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

fn hyprctl(args: &[&str]) -> bool {
    quiet(Command::new("hyprctl").args(args))
}

pub fn available() -> bool {
    std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some()
}

pub fn cursor_invisible(on: bool) {
    let v = if on { "true" } else { "false" };
    if !hyprctl(&["eval", &format!("hl.config({{ cursor = {{ invisible = {v} }} }})")]) {
        hyprctl(&["keyword", "cursor:invisible", v]);
    }
}

/// Close every screensaver window (all monitors), like the stock exit path.
pub fn close_all() {
    quiet(Command::new("pkill").args(["-f", "[o]rg.omarchy.screensaver"]));
}

/// Is a screensaver window the active one? Any monitor's counts.
pub fn screensaver_in_focus() -> bool {
    match Command::new("hyprctl").args(["activewindow", "-j"]).stdin(Stdio::null()).stderr(Stdio::null()).output() {
        Ok(o) => class_is_screensaver(&String::from_utf8_lossy(&o.stdout)),
        Err(_) => true, // no hyprctl: never exit on focus
    }
}

pub fn class_is_screensaver(json: &str) -> bool {
    // "class": "org.omarchy.screensaver" - tolerate spacing without a JSON parser.
    json.split("\"class\"").nth(1).is_some_and(|rest| {
        let v = rest.trim_start().trim_start_matches(':').trim_start();
        v.starts_with(&format!("\"{CLASS}\""))
    })
}

/// Watches Hyprland's event socket and confirms focus loss with hyprctl.
pub struct FocusWatch {
    sock: Option<UnixStream>,
    started: Instant,
    last_poll: Instant,
    dirty: bool,
    buf: Vec<u8>,
}

impl FocusWatch {
    pub fn new() -> FocusWatch {
        let sock = (|| {
            let dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
            let sig = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").ok()?;
            let s = UnixStream::connect(format!("{dir}/hypr/{sig}/.socket2.sock")).ok()?;
            s.set_nonblocking(true).ok()?;
            Some(s)
        })();
        FocusWatch { sock, started: Instant::now(), last_poll: Instant::now(), dirty: false, buf: Vec::new() }
    }

    pub fn fd(&self) -> Option<i32> {
        self.sock.as_ref().map(|s| s.as_raw_fd())
    }

    /// Drain pending events; remember whether focus may have moved.
    pub fn read_events(&mut self) {
        let Some(s) = &mut self.sock else { return };
        let mut chunk = [0u8; 4096];
        loop {
            match s.read(&mut chunk) {
                Ok(0) => {
                    self.sock = None;
                    break;
                }
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
        while let Some(nl) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            if let Some(rest) = line.strip_prefix("activewindow>>") {
                if !rest.starts_with(CLASS) {
                    self.dirty = true;
                }
            } else if line.starts_with("workspace>>") || line.starts_with("focusedmon>>") {
                self.dirty = true;
            }
        }
    }

    /// True once focus has really left the screensaver.
    pub fn lost(&mut self) -> bool {
        if self.started.elapsed() < LAUNCH_GRACE {
            return false;
        }
        let due = self.last_poll.elapsed() >= POLL_EVERY;
        if !(self.dirty || due) {
            return false;
        }
        self.dirty = false;
        self.last_poll = Instant::now();
        !screensaver_in_focus()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_class_in_hyprctl_json() {
        assert!(class_is_screensaver("{\n  \"address\": \"0x1\",\n  \"class\": \"org.omarchy.screensaver\",\n}"));
        assert!(!class_is_screensaver("{\"class\": \"firefox\"}"));
        assert!(!class_is_screensaver("{}"));
    }
}
