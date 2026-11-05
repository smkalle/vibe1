# Install on UserLAnd / proot Arch Linux ARM

UserLAnd (Play Store/F-Droid) and Termux's `proot-distro install archlinux`
give you a **glibc aarch64** userland. That matters because PyTorch publishes
aarch64 Linux wheels on PyPI, so the full Python baseline engine can run on
the phone's CPU.

## 1. Get a shell

- **UserLAnd**: add a session with *Arch* and *Terminal*, then log in.
- **proot-distro** (from Termux): `pkg install proot-distro && proot-distro install archlinux && proot-distro login archlinux`

## 2. Install

```bash
sudo pacman -Syu --needed git      # in proot-distro you are root: drop sudo
git clone https://github.com/smkalle/vibe1 && cd vibe1/omni-pocket

./scripts/install_userland_arch.sh            # UI + mock/remote engines only (fast)
./scripts/install_userland_arch.sh --local    # + torch (CPU) + transformers ≥ 5.20.2
./scripts/install_userland_arch.sh --server   # + FastAPI/uvicorn to serve other devices
```

If the system Python is too new for PyTorch wheels, the script uses `uv` to
fetch Python 3.12 into the venv.

## 3. RAM and disk budget for the `transformers` engine

| dtype | Weights in RAM | Practical phone RAM |
|---|---|---|
| bfloat16 (default) | ~4.4 GB | 8 GB works if few apps are open; 12 GB is comfortable |
| float32 | ~8.7 GB | 12–16 GB phones only |

Disk: about 5 GB in `~/.cache/huggingface` plus headroom. To put the cache on
a larger volume, set `HF_HOME=/path/to/dir`.

`python -m omnipocket doctor` prints what fits on your device.

## 3b. Faster: llama.cpp in proot

The GGUF build through llama.cpp is much lighter than PyTorch (~1.5 GB, and
quicker on CPU). Arch's packages may lag, so build llama.cpp v0.8.0 or newer:

```bash
sudo pacman -S --needed cmake gcc make curl openssl
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build && cmake --build ~/llama.cpp/build -j"$(nproc)" --target llama-server
export PATH="$HOME/llama.cpp/build/bin:$PATH"
./scripts/run.sh --llama
```

(In Termux, `pkg install llama-cpp` gives you the same thing without building.)

## 4. Run

```bash
./scripts/run.sh
```

Open `http://localhost:8501` in Chrome on the phone. Ports are shared between
proot and Android, so localhost works. Pick **transformers** in the sidebar.
The first question loads the model (minutes on a phone; later questions reuse
it).

Expectations: CPU-only proot decoding of a 2B model in bf16 is slow (a few
tokens/s on a recent flagship). Use it for privacy-critical one-off questions.
For a real-time feel, use the **Live (WebGPU)** tab, which runs on the GPU, or
the **remote** engine. Compare them on your phone in the **Benchmarks** tab.

Proot has no GPU access: `torch device` will always say `cpu`.
