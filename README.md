# MyriadForge

**Reviving the Intel Neural Compute Stick (Movidius Myriad 2) — discontinued silicon, modern edge-AI.**

MyriadForge brings the original **Intel NCS1 / Myriad 2 (MA2450)** back to life as a
useful 2026 edge accelerator: a dual-backend OpenVINO **shim**, **async-pipelined
throughput**, and five real application pilots — all running on the actual stick,
not a simulator.

> The NCS1 was discontinued and dropped from OpenVINO after 2020.3. This project
> pins the last supporting runtime, wraps it behind a modern surface, and squeezes
> measurable speed out of it — honestly, with the hardware's real limits documented.

---

## Highlights (measured on real Myriad 2 silicon)

| Workload | Result |
|---|---|
| **MobileNetV2 gate** — sync baseline | 15.7 inf/s (p50 ~64 ms) |
| **MobileNetV2 gate** — async pipeline (2 requests) | **23.2 inf/s → +48%** |
| **kNN-as-MatMul** (10k×256 FP16 table, batch 32) | **~285 queries/sec** |
| **Footage triage** (video → gate → index) | 20 frames gated on-stick |
| **CPU smoke vs onnxruntime** | max diff **2.7e-06** |

The +48% comes from **overlapping USB transfer with on-stick compute** via multiple
in-flight async requests — not faster math (the firmware bounds single-inference
latency). Model batching does *not* help on Myriad 2; async pipelining does. See
[docs/NCS1_ROADMAP.md](docs/NCS1_ROADMAP.md) for the full evidence.

## Why this exists

The Myriad 2 is a 12-SHAVE VLIW vector DSP USB accelerator that Intel abandoned.
It can't run LLMs or ASR — but it's a genuinely capable, ~1 W always-on **decision
engine** for static-shape FP16 workloads: CNN gates, MLPs, kNN-as-GEMM, DSP filters.
The 2026 application shape: a cheap always-on gate on the stick flags interesting
ranges, and an expensive host model (the "brain") only wakes up when needed.

## The shim architecture

```
modern apps / libs   → async orchestration, batching, modern preprocessing
THE SHIM             → apps/common/ov.py (dual-backend) + ncs1/ncs2 bridges
legacy device access → IECore 2020.3 (NCS1) / openvino.runtime 2022.3 (NCS2), isolated venvs
fixed hardware       → WinUSB + Myriad firmware + XLink USB protocol
```

One compat layer (`apps/common/ov.py`) detects whichever OpenVINO imports in the
active venv and exposes a single surface — so the **same five apps run unchanged**
on either the NCS1 (Myriad 2) or an NCS2 (Myriad X).

## The five application pilots (`apps/`)

| App | What it does |
|---|---|
| `apps/gate` | Decision-model gate: classifier + threshold → `myriadforge.event/1` trigger |
| `apps/footage` | Video → sampled frames → gate → timestamp/label/confidence index |
| `apps/podcast` | WAV → windows → gate → speech-event index |
| `apps/knn_engine` | kNN-as-MatMul vector search (table as constant GEMM weights) + capacity calculator |
| `apps/cascade` | Always-on gate fires → host Mirage describes the frame (latency chain) |

## Quickstart

Native Windows only (the MYRIAD driver story is Windows-native — not WSL2).

```powershell
# 1. Set up the isolated OpenVINO 2020.3.2 venv for the NCS1 (idempotent):
powershell -ExecutionPolicy Bypass -File tools\ncs1\setup_env.ps1

# 2. Install the WinUSB driver for the stick (one elevated run; approve UAC):
powershell -ExecutionPolicy Bypass -File tools\ncs2\install_driver.ps1

# 3. Functional smoke test (ONNX -> FP16 IR -> CPU plugin vs onnxruntime):
tools\ncs1\.venv\Scripts\python.exe tools\ncs1\smoke_cpu.py

# 4. Discover lanes + run a real benchmark on the stick:
uv sync
uv run myriadforge devices
uv run myriadforge bench onnx-models/mobilenetv2-7 --device ncs1-myriad

# 5. Run every pilot on the stick, one summary report:
tools\ncs2\.venv\Scripts\python.exe tools\ncs2\demo_mvp.py --lane ncs1 --device myriad --full
```

Individual apps (run inside the NCS1 venv, from the repo root):

```powershell
tools\ncs1\.venv\Scripts\python.exe -m apps.footage.triage clip.mp4 --device myriad `
    --model models/ncs1/mobilenetv2-7.onnx --fps 2 --threshold 0.1 --pipeline 2
tools\ncs1\.venv\Scripts\python.exe -m apps.knn_engine.engine --n 10000 --device myriad
```

> Model artifacts and sample media are downloaded locally (git-ignored). The
> registry (`models/registry.json`) records their sources; `myriadforge ingest`
> and `tools/ncs*/setup_env.ps1` fetch what's needed.

## Device lanes

| id | device | runtime | status |
|---|---|---|---|
| `ncs1-myriad` | Intel NCS1 / Myriad 2 (MA2450) | OpenVINO 2020.3.2 (IECore, Py3.7) | the acquired unit |
| `ncs2-myriad` | Intel NCS2 / Myriad X (MA2485) | OpenVINO 2022.3.1 (runtime, Py3.10) | ready for a unit |
| `myriad2-sim` | Myriad 2 behavioral sim | calibrated on real ma2450 data | always available |
| `myriad-sim` | Myriad X behavioral sim | synthetic constants | always available |

Simulated records are stamped `backend.simulated: true` — real and modeled data
never blend silently.

## Documentation

- [docs/NCS1_ROADMAP.md](docs/NCS1_ROADMAP.md) — the Myriad 2 revival: hardware reality,
  bring-up gotchas, measured envelope, throughput shim, what fits/doesn't.
- [docs/NCS2_ROADMAP.md](docs/NCS2_ROADMAP.md) — the Myriad X lane (workstreams WS0–WS6).
- [docs/index.html](docs/index.html) — the project site (also served via GitHub Pages).

## Development

```powershell
uv sync
uv run pytest        # hardware-free contract + sim + shim tests
```

Hardware-dependent paths are exercised by the pilots themselves inside the isolated
venvs (`tools/ncs2/demo_mvp.py`), not by the unit suite.

## License

MIT — see [LICENSE](LICENSE).
