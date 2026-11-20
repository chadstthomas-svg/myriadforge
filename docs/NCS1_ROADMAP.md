# NCS1 (Myriad 2) Revival — engineering record

**Status (2026-11-19): the original Intel Neural Compute Stick is working.**
Real inference, real bench records, and all five WS5 application pilots run on
the actual Myriad 2 silicon. This doc records what the hardware really is, the
non-obvious bring-up path, and the measured envelope. Companion to
[NCS2_ROADMAP.md](NCS2_ROADMAP.md) (the Myriad X lane, still valid for a future NCS2).

---

## What the hardware actually is

The dongle enumerated as `Movidius MA2X5X`, `USB\VID_03E7&PID_2150`. **PID 0x2150
is the original NCS1 / Myriad 2 (MA2450)** — not the NCS2 / Myriad X (MA2485,
PID 0x2485) the NCS2 roadmap targeted. The user confirmed the stick is labelled
"Myriad 2 VPU".

| Fact | NCS1 / Myriad 2 (this device) | NCS2 / Myriad X (roadmap assumption) |
|---|---|---|
| Chip | MA2450 | MA2485 |
| SHAVE VLIW DSPs | **12** (fixed, not tunable here) | 16 (tunable via `MYRIAD_NUMBER_OF_SHAVES`) |
| Neural Compute Engine | none | yes |
| USB PID (unbooted) | **0x2150** | 0x2485 |
| Firmware blob | `usb-ma2450.mvcmd` | `usb-ma2x8x.mvcmd` |
| Memory | 512 MB DDR (usable budget an estimate; plugin metric `NOT_IMPLEMENTED`) | 4 GB (~1 GB usable) |
| FP16 throughput class | ~100 GFLOPS | ~1 TOPS |
| Last supporting OpenVINO | **2020.3.x** | 2022.3.1 LTS |

## Why OpenVINO 2020.3.2 (source-verified)

The MYRIAD plugin's USB boot table (`XLink/pc/protocols/usb_boot.c`,
`supportedDevices[]`) lists the Myriad 2 PID **enabled in 2020.3.2** and
**commented out from 2020.4** ("Myriad 2 device has been deprecated since 2020.4
release"). 2022.3.1 therefore cannot drive this stick at all. 2020.3.2 is the
last release that accepts PID 0x2150, and its Windows USB stack
(`XLink/pc/Win/src/win_usb.c`) enumerates boot-state devices via
`GUID_DEVINTERFACE_Myriad2 = {19E08104-0543-40A5-B107-87EF463DCEF1}` — exactly the
WinUSB interface this device registers.

2020.3 uses the older **`IECore`** Python API (`openvino.inference_engine`), not
2022.3's `openvino.runtime.Core`, and ships **cp37** bindings only.

## Bring-up gotchas (all captured in `tools/ncs1/setup_env.ps1`)

1. **Python 3.7.9** from python.org (silent per-user install). uv has no 3.7
   standalone build (python-build-standalone starts at 3.8); the IE bindings are
   cp35/36/37 only.
2. **OpenVINO 2020.3.2 dev archive** (`w_openvino_toolkit_dev_p_2020.3.355.zip`,
   storage.openvinotoolkit.org). No PyPI wheel exists for 2020.3. Extract
   `inference_engine` (DLLs + `myriadPlugin.dll` + `usb-ma2450.mvcmd`),
   `model_optimizer`, **`ngraph`** (`ngraph.dll` is a plugin dependency), and the
   `python/python3.7` IE API.
3. **Intel C++ runtime DLLs** — `inference_engine_legacy.dll` and
   `myriadPlugin.dll` import `libmmd.dll` + `svml_dispmd.dll`, which Intel does
   **not** bundle in the OpenVINO archive. Sourced from the
   **`intel-cmplr-lib-rt`** pip wheel (`tools/ncs1/redist/`).
4. **PATH wiring** mirrors `bin/setupvars.bat`: TBB (`external/tbb/bin`) +
   Release + Debug + hddl + ngraph/lib + the redist dir. Missing TBB or the Intel
   DLLs gives `ImportError: DLL load failed`.
5. **numpy 1.19.5** (cp37 `ie_api` ABI); onnx 1.7 / IR version 6; opencv 4.6 /
   pillow 9.5 (last py3.7-friendly lines).
6. **WinUSB driver** — installed via `tools/ncs2/install_driver.ps1`
   (`ncs2_winusb.inf` covers PID_2150 and PID_2485; USBDevice class is
   signature-exempt). The device auto-binds WinUSB via its WCID
   (`USB\MS_COMP_WINUSB`) compatible ID.
7. **IECore config values must be strings** (`dict_to_c_map` rejects ints).
8. **MO 2020.3 + opset 12 fails** shape inference on MobileNetV2's
   GlobalAveragePool (`[0,0]` collapse). Use **opset 7** (`mobilenetv2-7.onnx`);
   it converts cleanly to FP16 IR v10.
9. **No SHAVE-count config key** in the 2020.3 Myriad plugin (only
   `FORCE_RESET`/`PLATFORM`/etc.) — Myriad 2 runs a fixed 12-SHAVE allocation, so
   the NCS2-style 1→16 sweep does not apply.
10. **IECore stdout noise** (`[BSL]`, `XLinkResetRemote`) lands after the JSON a
    bridge prints — parse by marker / `raw_decode`, never "last line".

## Measured envelope (real ma2450 silicon, 2026-11-19)

`tools/ncs1/calibrate.py` → `tools/ncs1/calibration.json`:

- **MobileNetV2-7 gate** (6.65 MB FP16 weights): load **1918 ms** (firmware boot +
  weight upload over USB), steady **p50 63.5 ms / p90 64.6 / p99 65.9**,
  **~15.8 inf/s**. Latency is very consistent (low variance).
- **Functional correctness**: MYRIAD top-3 indices match the CPU plugin exactly;
  scores agree within FP16 rounding. CPU smoke vs onnxruntime: max diff 2.7e-06.
- **kNN-as-MatMul** (10k×256 FP16 table, batch 32): **~285 queries/sec**,
  112 ms/batch. The GEMM lives on-device; TopK on host.
- **Wide-GEMM limit**: Myriad 2 MatMul **hangs on very wide outputs** — a
  `[8,100000]` score tensor never returns (MO converts fine in 2.5 s, MYRIAD
  loads fine in 3 s, first infer hangs). `[8,10000]` works (29 ms). Practical kNN
  tables stay ≤ ~10k–20k; larger tables need host-side tiling of the GEMM. The
  100k headline in the NCS2 roadmap was a Myriad X target.

## Architecture (how it plugs into myriadforge)

- **Dual-backend compat layer** `apps/common/ov.py`: detects whichever OpenVINO
  imports in the active venv — `openvino.runtime` (2022.3 / ncs2) or `IECore`
  (2020.3 / ncs1) — and exposes one surface (`compile_model`, ports with
  `get_partial_shape()`, `create_infer_request().infer({port: arr})`). The five
  apps run **unchanged** on either stick; `lane_prefix()` labels records
  `ncs1-*` vs `ncs2-*`.
- **Lane** `src/myriadforge/devices/ncs1.py` (`Ncs1Provider`) +
  `tools/ncs1/bridge_runner.py` (IECore-native), mirroring the NCS2 lane.
  Devices: `ncs1-myriad` (stick) / `ncs1-ovcpu` (CPU plugin fallback).
- **Simulator** `myriad2-sim` lane: `SimMyriadProvider({"chip": "ma2450"})` in
  `sim_myriad.py`, latency constants calibrated from the measured gate point
  (12 SHAVEs fixed, ~0.4 GB usable estimate). Stamped `backend.simulated: true`.
- **MVP** `tools/ncs2/demo_mvp.py --lane ncs1 --device myriad [--full]` runs every
  pilot on the stick and writes one summary report.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File tools\ncs1\setup_env.ps1   # idempotent
tools\ncs1\.venv\Scripts\python.exe tools\ncs1\smoke_cpu.py          # PASS
uv run myriadforge devices --all                                       # ncs1-myriad ready
uv run myriadforge bench onnx-models/mobilenetv2-7 --device ncs1-myriad
tools\ncs2\.venv\Scripts\python.exe tools\ncs2\demo_mvp.py --lane ncs1 --device myriad --full
```

## Throughput shim — async pipelining (the "speed up" lever)

The legacy IECore library is treated as a thin **device-access shim**; modern
orchestration plugs in above it. The boundary that matters:

```
modern apps / libs   → async orchestration, batching, modern preprocessing
THE SHIM             → apps/common/ov.py (num_requests surface) + ncs1 bridge
legacy device access → IECore 2020.3 (isolated Py3.7 venv), frozen
fixed hardware       → WinUSB + usb-ma2450 firmware + XLink USB protocol
```

**Honest boundary:** a newer host library cannot make the firmware/SHAVEs
compute faster — single-inference latency (~60 ms MobileNetV2) is firmware-bound.
The throughput gain comes from **overlapping USB transfer with on-stick compute**
via multiple in-flight async requests, not from faster math.

Measured on real ma2450 (MobileNetV2-7, 60 runs):

| mode | requests | throughput | effective ms/infer |
|---|---|---|---|
| sync (baseline) | 1 | **15.7 inf/s** | 63.8 |
| **pipelined** | **2** | **23.2 inf/s** | 43.2 |
| pipelined | 4 | 21.5 inf/s | 46.6 |
| pipelined | 6 | 20.8 inf/s | 48.0 |

**~+48% throughput at 2 requests** (the sweet spot; more requests add contention).
Key API detail: the request count is set via the **`num_requests` kwarg** to
`IECore.load_network` — the generic `NUMBER_OF_INFER_REQUESTS` *config key* is
rejected by the VPU plugin. `OPTIMAL_NUMBER_OF_INFER_REQUESTS` reports 2 for this
graph (graph-compiler-dependent: `2 × numExecutors`).

**What does NOT help (measured):** model batching. batch=4 gives 16.7 images/s
vs batch=1's 16.3 — 239 ms/batch ≈ 4×60 ms, i.e. Myriad 2 serializes per-image;
batching does not amortize the USB round-trip. `HW_STAGES_OPTIMIZATION` is
Myriad X-only. Async pipelining is the real lever.

**Where it's wired:**
- `tools/ncs1/bridge_runner.py --requests N` (default 2): continuous round-robin
  pipeline; records stamped `backend.pipelined`, `config.measurement_mode =
  pipelined-throughput` (metrics are inter-completion cadence, not single-shot
  latency). `--requests 1` = sync-latency bench for comparison.
- `Ncs1Provider` passes `requests` (devices.toml `[ncs1] requests = 2`).
- `apps/common/ov.py compile_model(..., num_requests=N)` + `compiled.requests`
  (raw IECore InferRequests: `async_infer`/`wait`/`outputs[name]`/`latency`).
- `apps/gate/runner.py infer_batch()` + `apps/footage/triage.py --pipeline N`:
  **continuous** async scheduling (next frame submitted the instant any request
  completes, no wave barrier). Measured app gain **+44%** (footage gated_fps
  15.4 → 22.1 at fps 6), matching the bridge's +48% hardware ceiling.
- Profiling hooks available: `PERF_COUNT` config + `InferRequest.get_perf_counts`
  (per-layer USB-vs-compute breakdown) for future transport tuning.

## What fits / what doesn't (Myriad 2)

Fits: static-shape FP16 CNNs (2D/1D), MLPs, decision-model gates, kNN-as-GEMM
(modest tables), DSP filters. Doesn't fit: autoregressive LLM/VLM decoding, ASR,
dynamic shapes, pointer-chasing (HNSW), very wide GEMM outputs (≥100k), and
anything needing >~0.4 GB. The 2026 application shape is unchanged: cheap
always-on gate on the stick → expensive host brain on flagged ranges.

## Open items

- **Capability shim** — modern API surface + HETERO Myriad/CPU partitioning so
  models with unsupported ops still partially accelerate; `VPU_CUSTOM_LAYERS`
  for extending op support (SHAVE kernels = the firmware-toolchain deep end).
- **Transport profiling** — `PERF_COUNT`/`get_perf_counts` to split the ~60 ms
  into USB-transfer vs on-stick compute; informs any user-mode WinUSB tuning.
- Memory budget is an estimate (plugin metric `NOT_IMPLEMENTED`); measure via a
  fit/no-fit sweep if a precise gate is needed.
- kNN host-side tiling to lift the wide-output limit (path to larger tables).
- Single-point latency calibration; refine across model sizes if the sim is used
  for predictive sizing.
- Thermal/long-run behavior not yet characterized.
