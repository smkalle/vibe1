#!/usr/bin/env python3
"""Check the Hyprland half of the screensaver contract against a fake Hyprland:
a stub hyprctl/pkill on PATH that log their arguments, and a fake socket2 that
emits events. Verifies: pointer hidden on start and restored on exit; focus
moving between screensaver windows (other monitors) does not exit; focus
leaving the screensaver exits; exit closes every screensaver window.

usage: python3 tests/hypr_contract.py [path/to/flysaver]
"""

import os
import pty
import select
import socket
import struct
import sys
import tempfile
import threading
import time
import fcntl
import termios

BIN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "target/release/flysaver")
tmp = tempfile.mkdtemp(prefix="flysaver-hypr-")
log = os.path.join(tmp, "calls.log")
active = os.path.join(tmp, "active.json")
bindir = os.path.join(tmp, "bin")
os.makedirs(bindir)
os.makedirs(os.path.join(tmp, "hypr", "SIG"))

stub = f"""#!/bin/sh
echo "$(basename "$0") $*" >> {log}
if [ "$1" = activewindow ]; then cat {active}; fi
exit 0
"""
for name in ["hyprctl", "pkill"]:
    p = os.path.join(bindir, name)
    open(p, "w").write(stub)
    os.chmod(p, 0o755)


def set_active(cls):
    open(active, "w").write('{\n  "address": "0x1",\n  "class": "%s"\n}\n' % cls)


set_active("org.omarchy.screensaver")

sock_path = os.path.join(tmp, "hypr", "SIG", ".socket2.sock")
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sock_path)
server.listen(1)
clients = []
threading.Thread(target=lambda: clients.append(server.accept()[0]), daemon=True).start()


def emit(line):
    for c in clients:
        c.sendall((line + "\n").encode())


pid, fd = pty.fork()
if pid == 0:
    env = dict(os.environ, PATH=bindir + ":" + os.environ["PATH"], HYPRLAND_INSTANCE_SIGNATURE="SIG", XDG_RUNTIME_DIR=tmp)
    os.execve(BIN, [BIN, "--seed", "9"], env)
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))


def pump(secs):
    end = time.time() + secs
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                os.read(fd, 65536)
            except OSError:
                return


reaped = False


def alive():
    global reaped
    if not reaped and os.waitpid(pid, os.WNOHANG)[0] != 0:
        reaped = True
    return not reaped


def calls():
    return open(log).read() if os.path.exists(log) else ""


pump(0.5)
assert clients, "flysaver never connected to the event socket"
assert "cursor = { invisible = true }" in calls(), calls()

# During launch focus hops between monitors: other screensaver windows count as focus.
emit("activewindow>>org.omarchy.screensaver,")
pump(2.0)
assert alive(), "exited while a screensaver window had focus"
print("ok  stays up while any screensaver window has focus")

set_active("firefox")
t0 = time.time()
emit("activewindow>>firefox,Mozilla Firefox")
while alive() and time.time() - t0 < 2.0:
    pump(0.02)
took = time.time() - t0
assert not alive(), "did not exit when focus left the screensaver"
print(f"ok  exits {took * 1000:.0f} ms after focus leaves")

c = calls()
assert "cursor = { invisible = false }" in c, c
assert "pkill -f [o]rg.omarchy.screensaver" in c, c
print("ok  restores the pointer and closes every screensaver window")
print("all hyprland contract checks passed")
