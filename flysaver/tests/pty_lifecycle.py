#!/usr/bin/env python3
"""Drive flysaver in a real pseudo-terminal and check the exit contract:
any key exits, mouse motion exits, SIGTERM/SIGHUP exit, and the terminal is
restored (alternate screen left, cursor shown, mouse reporting off).

usage: python3 tests/pty_lifecycle.py [path/to/flysaver]
"""

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time

BIN = sys.argv[1] if len(sys.argv) > 1 else "target/release/flysaver"
RESTORE = [b"\x1b[?1003l", b"\x1b[?25h", b"\x1b[?1049l", b"\x1b]111\x07"]


def spawn(args, cols=100, rows=30):
    pid, fd = pty.fork()
    if pid == 0:
        env = dict(os.environ)
        env.pop("HYPRLAND_INSTANCE_SIGNATURE", None)
        os.execve(BIN, [BIN] + args, env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    return pid, fd


def drain(fd, secs):
    out = b""
    end = time.time() + secs
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
    return out


def wait_exit(pid, fd, secs=1.0):
    t0 = time.time()
    out = b""
    while time.time() - t0 < secs:
        out += drain(fd, 0.05)
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            out += drain(fd, 0.1)
            return time.time() - t0, os.waitstatus_to_exitcode(status), out
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    return None, None, out


def case(name, args, trigger):
    pid, fd = spawn(args)
    first = drain(fd, 0.8)
    assert b"\x1b[?1049h" in first and b"\x1b[?1003h" in first, f"{name}: terminal not set up"
    assert b"BANC" in first, f"{name}: no HUD drawn"
    trigger(pid, fd)
    took, code, out = wait_exit(pid, fd)
    assert took is not None, f"{name}: did not exit"
    for seq in RESTORE:
        assert seq in out, f"{name}: missing restore sequence {seq!r}"
    print(f"ok  {name:<14} exited {code} after {took * 1000:.0f} ms, {len(first) // 1024} KB in first 0.8 s")
    os.close(fd)


case("key", ["preview", "--seed", "1"], lambda pid, fd: os.write(fd, b"x"))
case("mouse motion", ["preview", "--seed", "2"], lambda pid, fd: os.write(fd, b"\x1b[<35;10;5M"))
case("SIGTERM", ["preview", "--seed", "3"], lambda pid, fd: os.kill(pid, signal.SIGTERM))
case("SIGHUP", ["preview", "--seed", "4"], lambda pid, fd: os.kill(pid, signal.SIGHUP))
case("run mode key", ["--seed", "5"], lambda pid, fd: os.write(fd, b"\r"))
print("all lifecycle checks passed")
