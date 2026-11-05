"""Where are we running? Termux, a proot distro (UserLAnd / proot-distro), or a
regular Linux/macOS box. Used by ``doctor`` and to pick sensible defaults."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_env(environ: dict | None = None, proc_root: str = "/proc") -> str:
    """Return ``termux`` | ``proot`` | ``android-other`` | ``linux`` | ``macos`` | ``other``."""
    env = os.environ if environ is None else environ
    prefix = env.get("PREFIX", "")
    if "com.termux" in prefix or env.get("TERMUX_VERSION"):
        return "termux"
    # proot hides the real kernel but leaves traces: PROOT_* vars, the
    # UserLAnd support dir, or "proot" as our tracer process.
    if any(k.startswith("PROOT") for k in env) or Path("/support/userland_profile").exists():
        return "proot"
    try:
        status = Path(proc_root, "self", "status").read_text()
        for line in status.splitlines():
            if line.startswith("TracerPid:"):
                pid = line.split()[1]
                if pid != "0":
                    comm = Path(proc_root, pid, "comm").read_text().strip()
                    if "proot" in comm:
                        return "proot"
    except (OSError, IndexError):
        pass
    if env.get("ANDROID_ROOT") or Path("/system/build.prop").exists():
        return "android-other"
    system = platform.system()
    if system == "Linux":
        return "linux"
    if system == "Darwin":
        return "macos"
    return "other"


def mem_info(meminfo_path: str = "/proc/meminfo") -> dict:
    """Total / available RAM in MiB (0 if unreadable)."""
    out = {"total_mb": 0, "available_mb": 0}
    try:
        for line in Path(meminfo_path).read_text().splitlines():
            key, _, rest = line.partition(":")
            val = rest.strip().split()
            if not val:
                continue
            if key == "MemTotal":
                out["total_mb"] = int(val[0]) // 1024
            elif key == "MemAvailable":
                out["available_mb"] = int(val[0]) // 1024
    except OSError:
        pass
    return out


@dataclass
class Report:
    env: str
    machine: str
    python: str
    total_ram_mb: int
    available_ram_mb: int
    free_disk_gb: float
    modules: dict
    torch_device: str
    notes: list

    def as_dict(self) -> dict:
        return asdict(self)


# RAM to hold the 2.17B-parameter weights (+~25% for activations / KV / codec).
WEIGHT_GB = {"float32": 8.7, "bfloat16": 4.4, "float16": 4.4}


def recommend(env: str, total_mb: int, modules: dict) -> list[str]:
    notes: list[str] = []
    total_gb = total_mb / 1024 if total_mb else 0
    if env == "termux":
        notes.append(
            "Termux: PyTorch wheels are not on PyPI for Android/Bionic. Use the "
            "WebGPU tab (inference in Chrome) or the remote engine; use UserLAnd "
            "Arch or proot-distro if you want the Python baseline on the phone."
        )
    if env == "proot":
        notes.append(
            "proot (UserLAnd / proot-distro): glibc aarch64 wheels for torch and "
            "transformers install from PyPI. CPU only - no GPU access inside proot."
        )
    if total_gb:
        for dtype, need in WEIGHT_GB.items():
            ok = total_gb >= need * 1.25
            notes.append(
                f"{dtype}: needs ~{need:.1f} GB for weights -> "
                f"{'fits' if ok else 'too big'} in {total_gb:.1f} GB RAM"
            )
    if not modules.get("streamlit"):
        notes.append("streamlit is missing: run the install script for your environment.")
    if modules.get("torch") and not modules.get("transformers"):
        notes.append("torch found but transformers missing: pip install -r requirements-local.txt")
    return notes


def torch_device() -> str:
    if not has_module("torch"):
        return "n/a"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception as exc:  # broken install
        return f"error: {exc.__class__.__name__}"


def report(data_dir: str = ".") -> Report:
    env = detect_env()
    mem = mem_info()
    try:
        free_gb = shutil.disk_usage(data_dir).free / 1e9
    except OSError:
        free_gb = 0.0
    modules = {
        m: has_module(m)
        for m in ("streamlit", "numpy", "PIL", "requests", "torch", "transformers", "fastapi", "uvicorn")
    }
    return Report(
        env=env,
        machine=platform.machine(),
        python=platform.python_version(),
        total_ram_mb=mem["total_mb"],
        available_ram_mb=mem["available_mb"],
        free_disk_gb=round(free_gb, 1),
        modules=modules,
        torch_device=torch_device(),
        notes=recommend(env, mem["total_mb"], modules),
    )
