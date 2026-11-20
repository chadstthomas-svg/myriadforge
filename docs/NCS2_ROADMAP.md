# NCS2 Sub-Project Roadmap

**Goal:** revive the Intel Neural Compute Stick 2 (Movidius Myriad X, discontinued)
as an myriadforge adapter lane and a general static-shape FP16 offload engine —
with the majority of the software built and demoed **before the hardware
arrives**, against a two-layer simulator.

Scoped 2026-11-16. Extends [SCOPE.md](../SCOPE.md) (NCS2 moves from
*out of scope* to a **Phase 5 adapter lane**, same treatment as Coral).

**Delivery status (2026-11-16, wrapper-app build):** WS3 delivered as
`src/myriadforge/devices/sim_myriad.py` (envelope gates + CALIBRATION latency
model); WS4 delivered as `src/myriadforge/devices/ncs2.py` +
`tools/ncs2/bridge_runner.py` (subprocess bridge; activates when the venv
and dongle exist); WS5 partially delivered — the kNN engine and cascade
gate→Mirage demo run via `myriadforge demo run knn|cascade`. Still open:
WS0 (venv pin), WS1 (op matrix), WS2 (shim), full pipeline skeletons.

**Correction (2026-11-19):** the acquired dongle is an **original NCS1 / Myriad 2
(MA2450, PID 0x2150)**, *not* an NCS2 / Myriad X. OpenVINO 2022.3.1 cannot drive
it (Myriad 2 was dropped from 2020.4). The NCS1 was revived separately on
OpenVINO 2020.3.2 — see **[NCS1_ROADMAP.md](NCS1_ROADMAP.md)** for the working
lane (`ncs1-myriad`), measured envelope, and bring-up record. This NCS2 lane
remains valid and ready for a future Myriad X unit; the apps now run on either
stick via the dual-backend compat layer in `apps/common/ov.py`.

**Delivery status (2026-11-19, hardware arrival):** the dongle landed and
**WS0 is delivered** — `tools/ncs2/setup_env.ps1` (uv-managed Py 3.10 venv,
OpenVINO 2022.3.1 LTS; the PyPI wheel ships **no MYRIAD plugin DLL**, so the
script falls back to the official LTS archive for plugin + `usb-ma2x8x`
firmware), `tools/ncs2/smoke_cpu.py` (ONNX→FP16 IR→CPU plugin matches
onnxruntime to 4.7e-10), and `tools/ncs2/install_driver.ps1` +
`tools/ncs2/driver/ncs2_winusb.inf` (WinUSB INF for VID_03E7&PID_2150/2485;
USBDevice class is signature-exempt). The bridge lane is live:
`myriadforge devices --all` shows `ncs2-ovcpu` ready and the first bridge bench
recorded (MobileNetV2-12 FP16 IR, p50 ~4 ms on the CPU plugin). **WS5 is
delivered as real apps** — `apps/gate`, `apps/footage`,
`apps/podcast`, `apps/knn_engine`, `apps/cascade` speak the
`ncs2.event/1` + `ncs2.index/1` contracts, and `tools/ncs2/demo_mvp.py`
runs all pilots with one `--device` flag. Still open: **WS0 driver contact**
(the WinUSB INF needs one elevated `pnputil /add-driver`; UAC approval
pending), WS1 (op matrix), WS2 (shim), SHAVE sweep + MYRIAD.SIM calibration
(queued — needs the driver).

---

## Hardware envelope (what all software must respect)

| Fact | Value | Consequence for code |
|---|---|---|
| Chip | Movidius Myriad X MA2485, 16 SHAVE VLIW vector DSPs + NCE conv engine | latency models scale with SHAVE count |
| Memory | 4 GB LPDDR4, **~1 GB usable** for graph+weights | hard budget gate in simulator and adapter |
| Precision | **FP16 only** (int64 auto-downcast to int32) | shim must emit FP16-clean IR |
| Shapes | **static only** — no dynamic shapes | shim pins shapes at convert time |
| Transport | USB3; ~ms per host↔stick round trip | one big inference per transaction; batch queries |
| Software | OpenVINO **2022.3.1 LTS only** (MYRIAD plugin removed in 2023.0), Python ≤3.10 | isolated venv, never in myriadforge main env |
| Performance ballparks | MobileNet-SSD ~6–24 FPS (host-dependent); ~1 W | calibration targets, corrected on arrival |

Workload class that fits: **static-shape, FP16-tolerant, linear-algebra or
stencil-dominant** — CNNs (2D *and* 1D), MLPs, kNN-as-GEMM, DSP filters,
decision-model heads. Does not fit: autoregressive LLM/VLM decoding,
ASR, dynamic shapes, pointer-chasing (HNSW), int-heavy sequential work.

---

## The two-layer simulation principle

No NCS2 is needed to build or validate any of this, because the simulation
problem splits cleanly:

1. **Functional layer — OpenVINO CPU plugin.** The OpenVINO 2022.3.1 runtime
   installed in the isolated venv accepts the *same FP16 IR* the MYRIAD plugin
   would. Every shim output is validated for numerical correctness against an
   onnxruntime/PyTorch reference **on CPU, today**. This is truth for
   *correctness*, zero simulation involved.
2. **Behavioral layer — `MYRIAD.SIM`.** A fake device implementing the adapter
   runner interface that enforces the hardware envelope as *gates* (memory
   budget, op-support matrix, static shapes, FP16) and *models* latency
   (per-layer FLOPs × SHAVE-scaling curve + fixed USB overhead). This is a
   model of *performance and failure behavior*. Synthetic constants get replaced
   by measured ones when the stick lands (calibration hook).

Anything that passes layer 1 compiles-and-runs; anything that passes layer 2
*fits and is worth benching*. Real hardware only adds: firmware/driver contact,
true latencies, thermal behavior, SHAVE sweep ground truth.

---

## Workstreams

### WS0 — Toolchain pin (day 1, no hardware)

| Item | Path | Acceptance |
|---|---|---|
| Isolated venv setup script | `tools/ncs2/setup_env.ps1` | `py -3.10` venv; installs `openvino==2022.3.1` + `openvino-dev[onnx]==2022.3.1` (Model Optimizer `mo` CLI) from PyPI, else the GitHub LTS archive wheels |
| CPU smoke test | `tools/ncs2/smoke_cpu.py` | convert a small ONNX → FP16 IR via `mo`, run on CPU plugin, compare logits to onnxruntime reference (rtol 1e-2 — FP16) |
| Env sanity | — | `ov.Core()` lists CPU/GPU; absence of MYRIAD is expected and logged, not fatal |

Note: native Windows only (the MYRIAD driver story is Windows-native; do not
route this lane through WSL2).

### WS1 — Op-support matrix (knowledge artifact feeding everything)

| Item | Path | Acceptance |
|---|---|---|
| Matrix extraction | `tools/ncs2/op_matrix/extract.py` | clone/tag `openvino` at `2022.3.1`, extract the MYRIAD/VPU plugin's supported-op enumeration programmatically (not by hand) |
| Matrix data | `tools/ncs2/op_matrix/ncs2_ops.yaml` | op → {supported, caveats, min/max operand constraints}; covers ONNX ops (post-MO lowering) + IR ops |
| Doc | `docs/NCS2_OPS.md` | human table + sources; states confidence and what needs ground-truth validation on arrival |

This matrix drives the shim (what to decompose) and the simulator (what to
reject). It is the doc Intel stopped maintaining in 2023.

### WS2 — The compatibility shim

| Item | Path | Acceptance |
|---|---|---|
| Rewriter core | `tools/ncs2/shim/rewrite.py` | onnx/graphsurgeon passes: decompose LayerNorm, GELU, SiLU, Einsum→MatMul, stable softmax; int64→int32; pin dynamic→static shapes; constant-fold; opset downgrade (target ≤10/11) |
| Matrix-driven decisions | — | passes consult `ncs2_ops.yaml`; unknown ops fail loudly with a report, not silently |
| Correctness harness | `tools/ncs2/shim/validate.py` | reference (onnxruntime) vs shimmed-onnx vs shimmed-IR-on-CPU-plugin; per-output max-abs-error report |
| CLI | `tools/ncs2/shim/convert.py` | `python convert.py model.onnx --shape 1,3,224,224 -o out/` → FP16 IR + conversion report (ops kept / decomposed / rejected) |

Test models: a SiLU CNN, a LayerNorm-containing tiny encoder, a pure-MLP —
all in `tools/ncs2/shim/testdata/` with expected-pass/expected-fail labels.

### WS3 — The `MYRIAD.SIM` simulator

| Item | Path | Acceptance |
|---|---|---|
| Device façade | `src/myriadforge/devices/sim_myriad.py` | implements the same load/infer interface as the real adapter; probe info: `MYRIAD.SIM (ma2485-emulated, N shaves)` |
| Envelope gates | — | rejects: weights+graph > 1 GB (computed FP16), FP32 tensors, dynamic shapes, unsupported ops (per matrix) — with MYRIAD-style error messages |
| Latency model | — | per-layer FLOP estimate from graph × synthetic FP16 rate × SHAVE-scaling curve (`shave_count^α`, α≈0.7 until measured) + fixed USB overhead (~0.8 ms/transaction) + load-time model (weight bytes over USB3) |
| Config dims | — | `shave_count` (1–16), `nireq` — modeled, so sweeps produce sensible Pareto curves today |

### WS4 — myriadforge adapter lane

| Item | Path | Acceptance |
|---|---|---|
| Adapter runner | `src/myriadforge/devices/ncs2.py` | mirrors `HostBenchmark` (`runner.py`): `_load/_prepare_io/_inference/run`, emits `myriadforge.result/1` with `backend.name="openvino-myriad"` |
| Device selector | — | `--device myriad|myriad-sim|openvino-cpu` behind one class — SIM→REAL swap is a string, not a code change |
| Result schema addition | `bench/schema.py` | `backend.simulated: bool`, `backend.shave_count`, `backend.usb_transport`; real-device results indistinguishable in shape |
| CLI wiring | `cli.py` | `myriadforge devices` lists SIM; `myriadforge bench --device myriad-sim` end-to-end |
| ONNX intake note | `ingest/hf_resolver.py` | NCS2 lane benches ONNX sources (HF has them); .tflite artifacts are not convertible to IR reliably — lane declares its own model registry entries |

### WS5 — Application pilots (runnable today, all hardware-independent)

| Item | Path | What it proves |
|---|---|---|
| kNN-as-MatMul engine | `apps/knn_engine/` | build ONNX with constant vector table (1M×256 FP16 ≈ 0.5 GB → fits); query = input, scores = GEMM, TopK on host; capacity calculator + batched-query bench on CPU plugin + SIM → **queries/sec/stick estimate** |
| Decision-model gate | `apps/gate/` | small CNN/MLP classifier + threshold + trigger contract (JSON event out); runs on OpenVINO CPU now, MYRIAD later |
| Cascade orchestrator | `apps/cascade/` | gate → host **Mirage** via `cactus-compute` (runs today, stick-independent) — the full always-on-gate→host-brain loop, live, with end-to-end latency chain printed |
| Pipeline skeletons | `apps/podcast/`, `apps/footage/` | frame/audio ingestion → gate interface → index writer (timestamps + labels + confidences); drive with sample media |

### WS6 — Docs and scope

| Item | Path | Acceptance |
|---|---|---|
| This roadmap | `docs/NCS2_ROADMAP.md` | kept current as phases land |
| SCOPE.md update | `../SCOPE.md` | NCS2 row: out-of-scope → Phase 5 adapter (frozen OpenVINO 2022.3, isolated venv, same result schema) |
| Ops doc | `docs/NCS2_OPS.md` | from WS1 |

---

## MVP — the demo that exists entirely before the stick

Three live components, no hardware:

1. **Simulated bench run.** Shim-convert one modern model (SiLU/LayerNorm
   present) → `myriadforge bench --device myriad-sim` → versioned
   `myriadforge.result/1` JSON with percentile metrics + the conversion report.
2. **kNN engine headline number.** 100k×256 FP16 vector table as constant
   MatMul weights; functional correctness on CPU plugin; simulated
   queries/sec/stick from `MYRIAD.SIM`.
3. **Live cascade.** CPU-gate classifies sample frames → trigger fires →
   host Mirage describes the frame → latency chain (gate_ms + mirage_ms)
   printed end-to-end.

Demo script: `tools/ncs2/demo_mvp.py` runs all three and writes one summary
report. On hardware arrival, the same script runs unchanged with
`--device myriad`.

---

## Explicitly blocked on hardware

- USB driver (inf) install + device enumeration/firmware contact
- True latencies, thermal throttling behavior, USB bandwidth reality
- Ground-truth validation of the op matrix (WS1 confidence levels)
- The real `MYRIAD_NUMBER_OF_SHAVES` 1→16 sweep (replaces synthetic curve)
- Multi-stick `MULTI` device scaling
- FP16 accuracy audits on real silicon

---

## Sequencing (solo pace; dependencies, not deadlines)

```
WS0 ──▶ WS1 ──▶ WS2 ──▶ WS3 ──▶ WS4 ──▶ MVP polish
                 │              ▲
                 └──▶ WS5 (parallel with WS2–WS4; pilots don't block on the shim)
WS6 spans everything
```

- WS0–WS1: toolchain + matrix (~2 sessions)
- WS2 shim core (~2–3 sessions), WS3+WS4 (~2 sessions)
- WS5 pilots in parallel (~2 sessions), MVP assembly (~1 session)
- WS4's adapter interface is the critical path for MVP — stub it early if needed
