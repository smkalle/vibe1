//! Terminal lifecycle: raw mode, alternate screen, mouse-motion reporting,
//! signals, size, and a restore that also runs on panic.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

pub static STOP: AtomicBool = AtomicBool::new(false);
pub static RESIZED: AtomicBool = AtomicBool::new(false);
static SAVED: Mutex<Option<libc::termios>> = Mutex::new(None);

const ENTER: &str = concat!(
    "\x1b]11;rgb:00/00/00\x07", // black background, like the stock saver
    "\x1b[?1049h",              // alternate screen
    "\x1b[?25l",                // hide cursor
    "\x1b[?7l",                 // no autowrap
    "\x1b[?1003h\x1b[?1006h",   // report every mouse motion, SGR encoded
    "\x1b[0m\x1b[2J",
);
const LEAVE: &str = concat!(
    "\x1b[?1003l\x1b[?1006l",
    "\x1b[0m\x1b[2J\x1b[H",
    "\x1b[?7h",
    "\x1b[?25h",
    "\x1b[?1049l",
    "\x1b]111\x07", // reset background
);

extern "C" fn on_stop(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}
extern "C" fn on_winch(_: libc::c_int) {
    RESIZED.store(true, Ordering::SeqCst);
}

fn handle(sig: libc::c_int, f: extern "C" fn(libc::c_int)) {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = f as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0; // no SA_RESTART: poll() returns EINTR so the loop sees the flag
        libc::sigaction(sig, &sa, std::ptr::null_mut());
    }
}

pub fn install_signals() {
    for s in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT] {
        handle(s, on_stop);
    }
    handle(libc::SIGWINCH, on_winch);
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
}

pub fn is_tty() -> bool {
    unsafe { libc::isatty(0) == 1 && libc::isatty(1) == 1 }
}

pub fn enter() -> std::io::Result<()> {
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(0, &mut t) == 0 {
            *SAVED.lock().unwrap() = Some(t);
            let mut raw = t;
            libc::cfmakeraw(&mut raw);
            libc::tcsetattr(0, libc::TCSANOW, &raw);
        }
        libc::tcflush(0, libc::TCIFLUSH);
    }
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        leave();
        prev(info);
    }));
    write_all(ENTER.as_bytes())
}

/// Idempotent restore.
pub fn leave() {
    if let Ok(mut saved) = SAVED.lock() {
        if let Some(t) = saved.take() {
            let _ = write_all(LEAVE.as_bytes());
            unsafe {
                libc::tcsetattr(0, libc::TCSANOW, &t);
            }
        }
    }
}

pub fn write_all(b: &[u8]) -> std::io::Result<()> {
    let mut out = std::io::stdout().lock();
    out.write_all(b)?;
    out.flush()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size {
    pub cols: usize,
    pub rows: usize,
    pub px_w: usize,
    pub px_h: usize,
}

impl Size {
    /// Sub-pixel height / width; 1.0 when the terminal does not report pixels.
    pub fn aspect(&self) -> f32 {
        if self.px_w == 0 || self.px_h == 0 {
            return 1.0;
        }
        let cell_w = self.px_w as f32 / self.cols as f32;
        let cell_h = self.px_h as f32 / self.rows as f32;
        ((cell_h / 4.0) / (cell_w / 2.0)).clamp(0.5, 2.0)
    }
}

pub fn size() -> Size {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
            return Size { cols: ws.ws_col as usize, rows: ws.ws_row as usize, px_w: ws.ws_xpixel as usize, px_h: ws.ws_ypixel as usize };
        }
    }
    Size { cols: 80, rows: 24, px_w: 0, px_h: 0 }
}

/// Terminals start the pty at 80x24 and resize once the compositor has sized the
/// window; wait (briefly) for that so the first frame fills the screen.
pub fn wait_for_resize(max: std::time::Duration) {
    let start = std::time::Instant::now();
    while start.elapsed() < max && !STOP.load(Ordering::SeqCst) {
        let s = size();
        if !(s.cols == 80 && s.rows == 24) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Wait up to `timeout_ms` for input on fd 0 or the extra fd. Returns
/// (stdin readable, extra readable).
pub fn wait(extra: Option<i32>, timeout_ms: i32) -> (bool, bool) {
    let mut fds = [
        libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 },
        libc::pollfd { fd: extra.unwrap_or(-1), events: libc::POLLIN, revents: 0 },
    ];
    let n = unsafe { libc::poll(fds.as_mut_ptr(), 2, timeout_ms.max(0)) };
    if n <= 0 {
        return (false, false);
    }
    let hit = |p: &libc::pollfd| p.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0;
    (hit(&fds[0]), fds[1].fd >= 0 && hit(&fds[1]))
}
