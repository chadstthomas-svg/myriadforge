# Myriad 2 (NCS1): technical limits vs public benchmarks, how much is on the table, and the Mirage gap

**Scope:** the Intel Neural Compute Stick 1 (Movidius **Myriad 2**, MA2450) we have running, the public
performance claims for it, the gap between our measurements and the silicon's ceiling, and what a
sub-watt always-on accelerator like this could add to Cactus Compute's **Mirage** stack.

**Evidence discipline:** every number below is tagged `[measured]` (our own bench records in this repo),
`[vendor]` (a vendor's claim), or `[secondary]` (third-party/secondary source). Anything I could not
verify is marked `[unverified]` and named in §10. Public NCS1 benchmark data turned out to be sparse and
inconsistent — that is itself one of the findings.

---

## 1. Executive summary

1. **The 58 ms decomposes as 1.5 ms fixed dispatch (2.6 %), 14.8 ms input transfer (25.5 %), and 41.7 ms
   compute (71.9 %).** The compute runs at **~14.4 GFLOPS — only ~10–18 % of the 80–150 GFLOPS peak**
   `[measured]`. So the device is **compute-execution-bound, not USB-bound**: the SHAVEs are busy, but their
   kernels are inefficient. (This corrects an earlier guess that transport dominated.) And the paired-gap
   probes (§6) show the deficit is **model-structure**: dense 3×3 Conv kernels reach **31–50 GFLOPS** on the
   same silicon, while MobileNetV2's depthwise/1×1 path runs at **2.5–3.2 GFLOPS**.

2. **The transfer is real but hideable; the compute is not.** Input moves at **~40 MB/s** (poor for USB 3.0)
   and costs 14.8 ms. Overlapping it with compute via async requests gives **+48 %** (15.7 → 23.15 inf/s)
   `[measured]` — and the decomposition predicts it exactly: 58.0 − 14.8 = **43.2 ms**, which is the measured
   pipelined p50 to within **0.01 ms**. Deeper batching does not help because the device serializes per image
   (batch 4: predicted 232 ms, measured 239 ms).

3. **Against its own generation the device is what it is; against Myriad X it is a hardware generation
   behind.** The NCS2 (Myriad X) has **16 SHAVE cores plus a dedicated DNN accelerator** and Intel claimed
   **"up to 8×"** the NCS1 `[vendor]`. No shim closes an 8× silicon gap. "How much is on the table" therefore
   has two answers: **~5–10× vs its own compute ceiling** (kernel/firmware efficiency — mostly out of
   host-software reach), and **~8× vs the next chip** (hardware, out of reach).

4. **Mirage is a 22 MB, 2-bit vision model that runs on CPUs only.** Its runtime targets 17 platforms —
   macOS/Linux/Windows/Android/iOS/tvOS/watchOS/WASM/WASI `[vendor]` — and **none of them is an edge
   accelerator**. It also **serializes calls across threads** and explicitly tells users to add their own
   **VAD or wake word** before an always-on mic `[vendor]`.

5. **The gap this device fills is the missing half of the Mirage pattern: the always-on gate.**
   Cactus ships the expensive brain (Mirage/Whistle/Needle) and leaves the cheap "should I invoke it?" stage
   to the user, on the CPU. A 1 W, always-on, FP16 device that gates frames/audio before Mirage is invoked
   is exactly what they are missing — and we have it working, with a measured event/index contract
   (`ncs2.event/1`, JSONL index) already implemented in `apps/`.

---

## 2. What the device actually is

| Fact | Value | Tag |
|---|---|---|
| Chip | Movidius **Myriad 2**, MA2450 | `[secondary]` |
| SHAVE VLIW DSP cores | **12** (fixed; plugin exposes no SHAVE-count key) | `[measured]` (`myriad_plugin_config.hpp`, `vpu_plugin_config.hpp`) |
| Arithmetic | **FP16 only**; int64 downcast; **static shapes** | `[measured]` (plugin behaviour) |
| Peak throughput | **80–150 GFLOPS at ~1 W** (Fathom USB stick board) | `[secondary]` (Wikipedia/Movidius) |
| Host link | **USB 3.0** | `[vendor]` |
| Firmware | `usb-ma2450.mvcmd`, booted on first load | `[measured]` (our archive) |
| Runtime | OpenVINO **2020.3.2** (last release whose MYRIAD backend accepts PID 0x2150) | `[measured]` (boot table source) |
| Control surface | `FORCE_RESET`, `PLATFORM`, `LOG_LEVEL`, `INPUT_NORM/BIAS`, `COMPUTE_LAYOUT`, `CUSTOM_LAYERS`, `IGNORE_IR_STATISTIC`, `PRINT_RECEIVE_TENSOR_TIME`, `EXCLUSIVE_ASYNC_REQUESTS`, `PERF_COUNT`, `CONFIG_FILE`, `DEVICE_ID`, `HW_STAGES_OPTIMIZATION` (Myriad X only) | `[measured]` (headers) |

The control surface is itself evidence: the plugin exposes **no SHAVE-count knob, no stream count, no
high-throughput mode** on Myriad 2. `HW_STAGES_OPTIMIZATION` is documented Myriad X–only. So the levers
people assume exist for VPUs (SHAVE tuning, hardware stage pipelining, multi-request throughput) are
absent on this chip — which bounds what any host-side shim can extract.

---

## 3. Technical limits — hard vs soft

### 3.1 Hard (silicon or firmware contract; software cannot remove)

- **FP16, static shapes only.** No dynamic batch, no dynamic sequence length. `[measured]`
- **One graph resident at a time.** The first load boots firmware: **~1.6–1.9 s** `[measured]`. After boot,
  loading another graph is just graph+weight upload — **32–37 ms** for a tiny graph, **~306 ms** for
  MobileNetV2's 6.65 MB `[measured]`. Firmware boot is a one-time cost; model switching afterwards is cheap.
- **Per-inference cost decomposes (§6).** 1.5 ms fixed dispatch + ~40 MB/s input transfer + compute at
  ~14 GFLOPS. The compute is the bulk, and it is inefficient, not the USB.
- **Per-image serialization.** Batch=4 = 4 × 58 ms `[measured]`; the device does not amortize a batch.
- **Wide-output MatMul hangs.** A `[8, 100000]` GEMM never returns (MO converts, MYRIAD loads, infer hangs);
  `[8, 10000]` works at 29 ms `[measured]`. This caps kNN table width at ~10k–20k without host-side tiling.
- **No 2-bit/4-bit weight support.** FP16 is the only precision. Mirage's 2-bit weights cannot run here
  as-is; a gate for it must be a separate FP16 model.
- **Plugin introspection aborts.** `get_metric("EXECUTION_DEVICES")` **and** `InferRequest.get_perf_counts()`
  both **access-violate the process** on this build `[measured]`; `MYRIAD_DEVICE_TOTAL_MEM_SIZE` is
  `NOT_IMPLEMENTED`. Per-layer profiling must be done black-box (§6); the usable-memory budget (~0.4 GB) is
  an estimate.

### 3.2 Soft (firmware/toolchain choices — potentially fixable, but past the plugin boundary)

- **Firmware kernel coverage.** `HardSigmoid` is rejected (`unsupported layer type`), while `Erf`, `ArgMax`,
  `Elu` are accepted `[measured]`. That boundary is a firmware/toolchain property, and the only host-side
  remedy is **HETERO fallback to CPU** (we implemented `--device hetero`, proven: pure MYRIAD rejects a
  Conv→HardSigmoid graph, `HETERO:MYRIAD,CPU` runs it) `[measured]`.
- **No SHAVE-level tuning exposed.** Real gains would need plugin source changes or custom SHAVE kernels
  via `CUSTOM_LAYERS` — the deep end (firmware toolchain), not a Python shim.
- **Single-request default.** `OPTIMAL_NUMBER_OF_INFER_REQUESTS` reports 2, but the plugin allocates 1 unless
  you pass the `num_requests` kwarg; the generic `NUMBER_OF_INFER_REQUESTS` config key is rejected `[measured]`.

### 3.3 Not a limit

- **Raw compute.** For MobileNetV2 we use ~6–17 % of peak FLOPs. The SHAVEs are mostly idle waiting on USB.

---

## 4. Our measured reality

From `results/*.json` (device lane `ncs1`, MobileNetV2-7 gate, 224×224, batch 1):

| Mode | p50 | Throughput | Init | Notes |
|---|---|---|---|---|
| Sync (1 request) | **63.8 ms** | **15.7 inf/s** | 1.88 s | first-contact 58–60 ms |
| Sync (repeat runs) | 58.5–63.1 ms | 15.9–17.1 inf/s | 1.87–2.10 s | variance ±8 % |
| **Pipelined (2 requests)** | **43.2 ms** | **23.15 inf/s** | 1.84 s | **+48 %**; cadence, not single-shot latency |
| HETERO:MYRIAD,CPU | 58.7 ms | 17.0 inf/s | 1.90 s | fallback path, ~1 s extra load |
| kNN-as-MatMul, 10k×256, batch 32 | — | **285 queries/s** | — | 112 ms/batch; table = 4.9 MB FP16 |
| kNN, 100k×256 | — | **hangs** | — | wide-output limit (§3.1) |

Calibration point (`tools/ncs1/calibration.json`): weights 6.65 MB, load 1917.7 ms, p50 63.48 ms,
`ms_per_mb_full_shaves` 7.466, effective weight-upload ~15.9 MB/s.

---

## 5. What the public benchmarks actually show

This is the part the question most wants answered, and the honest answer is **"less than you'd hope."**

- **Vendor spec sheets give silicon, not FPS.** Intel's NCS2 page states the Myriad X has
  *"16 powerful processing cores (called SHAVE cores) and a dedicated deep neural network hardware accelerator"*
  and USB 3.0, but **no FPS figures for any named network** `[vendor]`.
- **The NCS1 vendor material similarly advertises capability, not benchmarks.** The Movidius/Intel pages
  describe "real-time inferencing" and performance-per-watt, with no per-network numbers `[vendor]`.
- **The OpenVINO benchmark result tables that did carry MYRIAD rows are no longer retrievable.** The 2020.x
  `openvino_docs_performance_benchmarks*` pages now 404 (Intel 404-redirector) and the Wayback copies I could
  reach expose only the overview page, not the MYRIAD rows `[verified: unavailable]`. The 2021.4 results were
  published as a downloadable spreadsheet, not inline HTML.
- **The NCSDK (legacy SDK) README contains no benchmarks either** — it only distinguishes generations and says
  NCS2 users must use OpenVINO instead `[verified: NCSDK README]`.
- **The only hard throughput figure I could source is the silicon ceiling: 80–150 GFLOPS at ~1 W** for a
  Myriad 2 board `[secondary]`, and Intel's claim that NCS2 delivers **"up to 8×"** the NCS1 `[vendor]`.

**Consequence for comparison:** there is no trustworthy, condition-matched public FPS number for MobileNetV2
on an NCS1 to put beside our 15.7–23.15 inf/s. The defensible comparison is **against the silicon ceiling**,
which §6 does. Any claim of the form "the NCS1 was published at N FPS" should be treated as unverified unless
the network, input size, batch, precision and USB conditions are stated.

---

## 6. How much is on the table — measured decomposition

The plugin's own per-layer profiler is unusable here (§3.1: `get_perf_counts()` access-violates), so the
split below is **black-box**: three graphs isolate the parts with the same runtime path.

| Probe model | Input | What it isolates | p50 | min |
|---|---|---|---|---|
| `relu_tiny` | `[1,8]` (32 B) | fixed USB round trip + firmware dispatch | 1.68 ms | **1.52 ms** |
| `gap_112` | `[1,3,112,112]` (150 KB) | + input transfer at ¼ size | 5.47 ms | 5.26 ms |
| `gap_224` | `[1,3,224,224]` (602 KB) | + input transfer at full size | 16.67 ms | 16.32 ms |
| `mobilenetv2-7` | `[1,3,224,224]` | + compute (600 MFLOP) | 58.38 ms | **58.01 ms** |

**Result (min latency, least noisy):**

| Component | Time | Share of 58.01 ms | Implied rate |
|---|---|---|---|
| Fixed dispatch (USB round trip + firmware) | **1.52 ms** | **2.6 %** | — |
| Input transfer, 602 KB @ 224×224 | **14.79 ms** | **25.5 %** | **40.7 MB/s** |
| Compute + residual (600 MFLOP) | **41.69 ms** | **71.9 %** | **14.4 GFLOPS** |

The 112 px probe confirms the transfer is bandwidth-linear, not fixed: 3.74 ms for 150 KB = **40.3 MB/s**,
matching 40.7 MB/s at 224 px. So ~40 MB/s is the real input path rate — poor for USB 3.0, but only a quarter
of the budget.

**The decomposition predicts the rest of our data — which is what makes it trustworthy:**

| Prediction | Predicted | Measured | Delta |
|---|---|---|---|
| Pipelined (transfer hidden behind compute) = 58.0 − 14.8 | **43.2 ms** | **43.20 ms** | **0.01 ms** |
| Batch 4 (serialized per image) = 4 × 58.0 | 232 ms | 239 ms | 7 ms (3 %) |
| Sync total = 1.5 + 14.8 + 41.7 | 58.00 ms | 58.01 ms | 0.01 ms |

**So the answer to "how much is USB vs firmware vs compute":** compute is **72 %** of the time, transfer
**26 %**, fixed dispatch **2.6 %**. Compute runs at **14.4 GFLOPS = ~18 % of 80 GFLOPS / ~10 % of 150 GFLOPS**
`[secondary ceiling]`. In other words the device is **compute-execution-bound, and its kernels are ~5–10×
off peak** — not waiting on USB. The transfer half is already banked (that is exactly what pipelining bought).

### The cascade benchmark vs Mirage — measured (2026-11-20)

`tools/ncs1/cascade_bench.py` + `tools/ncs1/gate_daemon.py` (persistent daemon, one firmware boot per
pass, cv2-free via PIL + `GateRunner.infer_nchw`): 60 frames of 720p sample video, same Mirage brain,
three ways of deciding when to invoke it.

| Config | Invocations /60 | Gate | Mirage | Wall |
|---|---|---|---|---|
| **no-gate** (Mirage on every frame) | 60 | — | 95.4 s | **95.4 s** (1.59 s/frame) |
| gate-myriad @0.1–0.5 | 59–60 | 4.5–4.6 s | 94.6–98.0 s | 99–103 s |
| gate-cpu @0.3–0.5 | 59–60 | 1.44–1.68 s | 72–73 s | 74–75 s |

**The honest finding: on this content the cascade saves nothing** — the gate fires on 59–60/60 frames,
because MobileNetV2 is *confidently wrong* on every cartoon frame (top-1 ≥ 0.5 nearly everywhere). A
generic top-1-confidence gate does not discriminate "interesting" on real footage. The result stands as
a harness validation and a product lesson:

- **The gate needs task-specific semantics** — a person/event/scene detector trained for the actual
  question, not ImageNet top-1. The contract (`ncs2.event/1` + index) already supports it; the model is
  the remaining work.
- **The gate's win is power and host-CPU freedom, not wall-clock**: gate-stick costs ~76 ms/frame
  (incl. daemon IPC) at ~1 W while gate-cpu costs ~24 ms/frame on a 30 W-class CPU. On real footage
  where 1–5 % of frames fire, the stick gate runs continuously at ~1 W and cuts Mirage invocations
  20–100× — the host CPU is otherwise free.
- Mirage measured **95.4 s / 60 frames = 1.59 s per describe** on the host CPU (720p) — the brain is
  ~26× the gate's per-frame cost. That ratio is the cascade's economic engine.

---

### The ceiling is model-structure-dependent, not device-wide (paired-gap probes)

The paired-gap method (each compute probe benched against a **zero-compute gap model with the identical

The paired-gap method (each compute probe benched against a **zero-compute gap model with the identical
input shape**, so transfer and dispatch cancel exactly) answers the next question — *is 14.4 GFLOPS a
Conv-kernel ceiling or a device-wide wall?* — decisively:

| Probe | Workload class | Compute GFLOPS |
|---|---|---|
| Dense Conv 3×3, 32×28×64→64 + GAP (29 MFLOP) | compute-dense | **50.6** |
| Dense Conv 3×3, 64×56×128→128 + GAP (462 MFLOP) | compute-dense | **31.0** |
| Dense Conv 3×3, 128×112×128→128 + GAP (3.7 GFLOP) | compute-dense | **35.8** |
| FC stack, 1×1 convs d=1024 ×8 / d=2048 ×4 + GAP | MobileNetV2's 1×1 path | **2.5–3.2** |
| MobileNetV2-7 (the gate model, §6) | mixed depthwise + 1×1 + dense | **14.4** |

**Findings:**
- The device's **dense-Conv kernels are good**: 31–50 GFLOPS sustained = **up to ~63 % of the 80 GFLOPS
  figure** (and ~25–45 % of 150). The 5–10× "off peak" gap from §6 is NOT a device-wide kernel deficit.
- **MobileNetV2's own structure is the bottleneck**: its depthwise convs and 1×1 bottleneck path run at
  **2.5–3.2 GFLOPS** (memory-bound — each 1×1 streams d×d weights and d×d activations), dragging the mix
  down to 14.4. A compute-dense redesign of the gate (favoring dense 3×3 convs over depthwise/1×1) could
  plausibly reach **~2–3.5×** on the same silicon.
- **The 1×1/FC path is the worst class on the device** — relevant for transformer-style models: an
  all-1×1 workload is memory-bound at ~3 GFLOPS, i.e. ~10× worse than dense conv.

**Additional transport/precision probes:**
- **FP16 input tensors are accepted** and dramatically faster: `gap_224` p50 drops **16.32 → 4.07 ms (4×)**
  with FP16 input (halves transfer bytes, skips FP32 handling). A direct optimization for every app: feed
  the gate FP16 preprocessed input.
- **Output direction runs at ~93 MB/s** — 2.3× faster than the input direction (40.7). Asymmetric USB path.
- **Wide-output threshold is tighter than first measured**: Gemm outputs **hang at 40k** (20000 works at
  2.38 ms). kNN tables stay ≤ 20k; the earlier 100k-GEMM hang sits in the same family.
- **Conv1d graphs wedge Myriad 2 at load** (deterministic, 2/2 — even at L=2000): a 3D-input Conv never
  boots. Audio gates must reshape to **2D Conv (kernel [1,9])** — the standard workaround.

So "how much is on the table" now has a three-layer answer: **~5–10× vs the peak figure** (silicon
marketing), **~2–3.5× vs dense-Conv kernels for MobileNetV2-class models** (restructure the model), and
**~1.5× banked** (pipelining, done) — plus **free FP16-input wins** the shim exposes to every app.

So "how much is on the table" now has a three-layer answer: **~5–10× vs the peak figure** (silicon
marketing), **~2–3.5× vs dense-Conv kernels for MobileNetV2-class models** (restructure the model), and
**~1.5× banked** (pipelining, done).

**Per-workload answer:**

| Workload | On the table | Bottleneck | Reachable by host software? |
|---|---|---|---|
| CNN gate (MobileNetV2-class) | **~5–10× compute** (14.4 → 80–150 GFLOPS), transfer already hidden | SHAVE kernel/firmware efficiency | **no** — needs plugin/firmware work (or a smaller model) |
| kNN-as-GEMM (10k table) | **~1–2×** | output transfer dominates; wide outputs hang | partly (host tiling) |
| Tiny compute-dense models (1D CNN, MLP) | utilization unknown, likely higher | compute share rises | untested — worth measuring |
| vs **Myriad X (NCS2)** | **~8×** | hardware generation | no |

**Bottom line (corrected by measurement):** the earlier guess that transport dominated was **wrong** — the
time is in compute, but the compute is running at ~14 GFLOPS. Host software has already taken the one big
lever it owns (+48 %, by hiding transfer). The remaining 5–10× is **firmware/kernel efficiency**, past the
Python boundary. That reinforces the strategic conclusion: don't fight the chip for throughput — use it
**where 15–23 inferences/s at ~1 W is already enough**, i.e. the always-on gate, and let a bigger model do
the rest.

### Op-support matrix (WS1, empirical) — `tools/ncs1/op_matrix.py`

A systematic sweep of 68 ops (minimal ONNX graph each, opset 11, MO → pure MYRIAD load+infer,
subprocess-per-op with hang timeouts, chunked with settle pauses) on real ma2450:
**42 OK · 7 REJECT · 15 MO_FAIL · 3 DEVICE_WEDGE · 1 CHECK_FAIL** (`tools/ncs1/op_support.json`,
generated table `docs/NCS1_OPS.md`).

The genuine rejections (converted fine, firmware refused):
- **`HardSigmoid`** and **`Abs`** — unsupported layer types; **`Softsign`** falls with Abs (it decomposes
  to Abs). These are MobileNetV3-era activations — exactly what HETERO covers.
- **`Clip`/`Clamp`** — the MO-generated Clamp node fails in the plugin ("incorrect number of inputs") even
  behind a Conv; `Expand` also fails after a Conv (and MO cannot convert several data-dependent ops at
  all: `NonZero`, `Where`, `CumSum`, `ScatterND`, `Softplus`, `Selu`, `LogSoftmax`, `Round`, `ArgMin`).
- Confirmed supported anchors: `Conv`, `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Elu`, `Erf`, `ArgMax`,
  `MatMul`, `Gemm`, `Softmax`, `MaxPool`, `AveragePool`, `TopK`, the reduce family, and the tensor-manip
  family (`Reshape`/`Transpose`/`Slice`/`Concat`/`Gather`/`Cast`/`Pad`/`Resize`/…).

Probe artifacts (not firmware verdicts, honestly labelled): `Shape` (batch-0 metadata), `Expand` head-only,
`Prelu` (onnx 1.7.0 ships **no Prelu schema at all**, so it cannot even be expressed). `LRN` /
`BatchNormalization` / `Shape_afterConv` re-probed 2026-11-20 after recovery and **wedged again** —
these three are deterministic device wedges, not a one-off USB transient.

This table **finishes the WS1 op-coverage deliverable empirically** — measured firmware truth, which
captures quirks the plugin source does not show — and powers the predictive partitioner
(`tools/ncs1/hetero_plan.py`): given any ONNX model, it names the ops that will run on the Myriad, the
ops that will fall back to CPU under HETERO, and any unmeasured ops, before the hardware is touched.

**Partitioner validation:** on the synthesized Conv→HardSigmoid graph it predicts exactly the measured
truth (Conv → MYRIAD, HardSigmoid → CPU, "HETERO required"). On MobileNetV2-7 it flags the one gap: the
`BatchNormalization` probe wedged this session, so the tool reports "contains unmeasured ops" rather than
silently assuming — the conservative answer. (BN is de-facto supported: the pure-MYRIAD bench runs that
produced §6's numbers execute all 27 BN layers.)

---

## 7. Mirage: what it is, where it is strong, where it is missing

### 7.1 What it is

- **22 MB vision model** (`[vendor]`), for "image understanding and structured extraction," doing
  **object detection, OCR, segmentation, decision and embedding** `[vendor]`.
- API: `cactus.Mirage().describe(path)` returns sections with **bbox and mask** fields;
  `extract(...)` returns ordered fields (enum / bounded int / string / box references) `[vendor]`.
- Accepts file paths, encoded bytes, Pillow images, or RGB `uint8` arrays `[vendor]`.
- **Architecture (from its config)** `[verified: HF config.json]`: a compact vision-language transformer —
  vision **body 16 layers**, 12 attention layers, 4 heads, stem `[32,64,128,256,512]`, **canvas 512**;
  text **decoder 8 layers, dim 512**, 2 KV heads, 8192 context; **detections 64**, **regions 32**;
  token buckets `[512, 1024, 1536]`; **7 languages** (en, de, fr, es, it, nl, pl).
- **2-bit quantized** (`cact_bits: 2`, embedding/mhc at 4 bits) `[verified: HF config.json]` — that is how
  22 MB fits a 512-wide encoder+decoder.
- License **Apache-2.0**; **~43–50k downloads/month** `[vendor]`.

### 7.2 Where it is strong

- Genuinely small and genuinely multi-task: detection + segmentation + OCR + VQA + structured extraction in
  one 22 MB artifact is a serious compression result.
- Structured output with **box and mask references** and a schema-constrained `extract` is exactly what
  automation needs — not free-text captions.
- Cross-platform runtime (17 targets) including **browser/WASM**, so it runs anywhere a CPU runs.

### 7.3 Where it is missing (the gaps this device maps onto)

1. **No always-on mode, and no gate.** Mirage is request/response (`describe(path)`). Cactus's own docs tell
   users to put a **VAD or wake word before an always-on mic** `[vendor]` — i.e. *they punt the cheap always-on
   stage to the user*, on the CPU. For vision there is no equivalent guidance at all.
2. **No accelerator backend.** The runtime's 17 targets are all CPU/GPU/WASM; **no Coral, no Jetson, no
   Myriad/NPU** `[vendor]`. Everything, including the 22 MB VLM, competes for host CPU cycles.
3. **Serialized calls.** "Calls are serialized across threads" `[vendor]` — no concurrent inference, no
   pipelining. Our measured **+48 %** from async overlap is precisely the throughput this design leaves on
   the floor for streaming workloads.
4. **No event/index contract.** There is no timestamped, append-only "what did the gate see" stream. Our
   `ncs2.event/1` + JSONL index is the missing layer between a gate and a brain.
5. **All-or-nothing cost.** Every `describe()` pays the full 22 MB encoder+decoder. There is no cheap
   pre-filter, so running Mirage at 30 fps is prohibitive; the product shape requires *deciding when to run it*.
6. **Deployment friction.** The engine and weights are downloaded and cached, and a `.cact` archive is
   **tied to the engine version and must be rebuilt after upgrades** `[vendor]` — awkward for constrained or
   air-gapped devices.
7. **2-bit only.** Its weights are not FP16-clean; they cannot be handed to a Myriad-class device as-is.
   That is a *complement*, not a conflict (see §8).

---

## 8. What this device could add to the Mirage stack (ranked)

| # | Idea | Why the Myriad fits | Feasibility | Effort |
|---|---|---|---|---|
| 1 | **Always-on gate in front of Mirage** — a tiny FP16 CNN/embedder on the stick decides per frame/audio-window whether Mirage is invoked; only hits go to the host brain | 1 W, always-on, FP16 static-shape is exactly the gate profile; we measure 15.7–23 fps and **285 qps** for embedding search; the event/index contract already exists (`apps/gate`, `apps/footage`, `apps/cascade`) | **high** | **low** — already built and running; productize the cascade |
| 2 | **A Cactus runtime backend for the gate** — a Myriad backend behind Cactus's C API (`cactus.h`, `libcactus.a`) so the gate runs inside their stack | Their runtime already ships a C API and 17 targets but **zero accelerators**; a new backend is additive, not a fork | **medium** | medium — FFI layer over our `bridge_runner` + a `cactus`-shaped wrapper |
| 3 | **Pipelined gate + brain** — overlap stick gating with host Mirage inference so the aggregate frame rate rises without a faster brain | Cactus serializes calls; our continuous async pipelining is measured at **+44–48 %** on the stick | **high** | medium — scheduler on top of existing apps |
| 4 | **Timestamped event index as the gate→brain contract** — a durable "what was seen, when, how sure" log that Mirage consumes selectively | We already emit `ncs2.event/1` + JSONL; Mirage has no such stream | **high** | low — mostly documentation + a consumer adapter |
| 5 | **Privacy default-on**: gate runs locally on 1 W; frames that don't fire never leave the device or wake Mirage | The stick is a physical, auditable always-on stage; the big model is opt-in per event | **medium** | medium — mostly packaging/claims, must be honest about USB |
| 6 | **Heterogeneous split**: FP16 gate on the stick + 2-bit Mirage-class brain on host CPU | The two precisions are complementary; HETERO already proves unsupported ops fall back cleanly | **medium** | medium — needs a real cascade benchmark, not a toy |
| 7 | **kNN/embedding memory on the stick** — a constant-weight GEMM table for cheap nearest-neighbour gating of embeddings | 285 qps at 10k×256 measured; 100k hangs on wide output (host-tiling needed) | **low–medium** | high — wide-GEMM tiling or firmware work |
| 8 | **Custom SHAVE kernels** to extend op coverage (e.g. HardSigmoid natively) | Would remove the HETERO fallback for MobileNetV3-class nets | **low** | very high — firmware toolchain |

**The one to build next:** #1 is already working; #3 and #4 turn it from a demo into a product shape. #2 is
the credible way to plug into Cactus's ecosystem rather than compete with it. #8 is the deep end and should
not be attempted before 1–4 pay off.

---

## 9. Incidental findings worth keeping

- **Plugin introspection aborts.** Both `get_metric("EXECUTION_DEVICES")` and `InferRequest.get_perf_counts()`
  **access-violate** the 2020.3.2 Myriad plugin — never call either; profile black-box instead (§6).
- The plugin rejects `NUMBER_OF_INFER_REQUESTS` but honours the **`num_requests` kwarg**; `OPTIMAL_…` reports 2.
- `HardSigmoid` (MobileNetV3-style) is rejected by pure MYRIAD but runs on HETERO — so **capability and
  throughput are separable concerns** (two shims, two gains).
- Public NCS1 performance data is effectively **unavailable** in a comparable form today; treat any
  unsourced "NCS1 does N FPS" as unverified.
- **FP16 accumulator saturation.** Raw N(0,1) random kernels produce non-finite outputs on the stick;
  Xavier-scaled inits are finite everywhere. Training-side weight scaling is a hardware-visibility concern.
- **Weight-fit boundary (first data point).** A 51.2 MB FP16 MatMul-weights graph (5×20k-column tiles,
  multi-output) fails device init (`NC_ERROR`) and — worse — **drops the stick off USB enumeration**,
  requiring a physical replug. The 10k×256 (~10 MB) baseline fits comfortably; bracket the boundary
  downward (50k ≈ 25.6 MB next). Multi-output compile itself is fine at small scale (§6.4).

---

## 9.5 The voice-to-tool slice (voicetools/1) — first experiment status (2026-11-20)

The research plan's first experiment (one spoken request → ordered, typed tool calls on one MA2450,
Cactus host stack as reference-to-beat) is underway. Results so far:

- **Task frozen** (`labs/shave_models/tools_spec.py`, `voicetools/1`): 12 benign tools (lighting/timers/
  media), enum room/device/scene slots, one numeric slot per call, max 2 ordered calls, explicit
  rejection. Fixtures: 2,680 clips (2,640 SAPI-synthesized utterances over 30 deterministic
  voice×rate×pitch condition cells — held-out cells are the unseen-speaker proxy — plus 40 noise
  clips), 3,102 graph windows. Label-vs-spoken-text consistency audited to zero mismatches.
- **Handoff experiment PASSED** (`labs/shave_models/handoff_probe.py`): MO 2020.3 + IECore return
  **multiple outputs** (tool-score head + window-value head, both [1,·], finite) — the shared
  multi-head graph is trainable, no head-splitting needed. Warm p50 **5.14 ms** per 2 s window:
  the ≤300 ms end-to-end budget has ~58× headroom for framing.
- **Host reference FAILS all three acceptance gates** (`labs/shave_models/host_reference.py`,
  Whistle ASR → Needle 3 over the same test split): exact match **35.9%** (gate ≥90%), false-action
  rate **13.5%** (gate ≤1%), p95 **664 ms** (gate ≤300 ms). The two-stage host stack is both slow and
  inaccurate on a narrow frozen task schema — the on-device baseline to beat is loose, which
  strengthens the case for the single trained acoustic graph.
- **Trainer** (`labs/shave_models/train_voicetools.py`): multi-head (tool[13] + room/device/scene/
  level/seconds none-indexed heads), per-window training over the eval-contract window set. First
  recipe collapsed to the all-none prior; inverse-sqrt class weights recover learning (tool acc on
  firing windows 0.18→0.38 by epoch 8) but the 32-d GAP bottleneck discards the temporal structure
  the slot heads need — architecture v2 keeps time resolution before the acceptance run.
- **Device evaluation** (`labs/shave_models/eval_voicetools.py`): ready, blocked on the stick replug.

Honest scope notes: fixtures are TTS (condition-matched for every comparison; real-speaker expansion
is future work with mini_speech_commands as the real-audio anchor); ordered calls come from the
per-window decision rule in this slice — sequence/CTC heads are the next slice.

---

## 10. Open questions and what would settle them

**Answered (2026-11-20):** *what fraction of the 58–64 ms is USB vs firmware vs compute* → **1.5 ms fixed
(2.6 %), 14.8 ms input transfer (25.5 %), 41.7 ms compute (71.9 %)**, with compute at ~14.4 GFLOPS (§6).
The plugin's `PERF_COUNT`/`get_perf_counts()` path is broken (access violation), so this was settled
black-box; the decomposition reproduces the pipelined and batched measurements to 0.01 ms / 3 %.

| Remaining question | Why it matters | What would settle it |
|---|---|---|
| Does a compute-dense tiny model (1D CNN/MLP) reach higher utilization than 14 GFLOPS? | Tests whether the 14 GFLOPS is kernel-specific or a device-wide ceiling | Bench a 1-D CNN and a small MLP on the stick (both fit easily), same decomposition |
| Is there a faster input path than 40 MB/s (FP16 input, zero-copy, larger transfers)? | If yes, transfer shrinks further — though pipelining already hides it | Try FP16 input tensors and a 1×1 conv probe with varying input sizes; measure MB/s |
| What is the real usable memory budget? | Gates how large a gate model / kNN table can be | Fit/no-fit sweep (metric is absent) |
| Does a Myriad backend actually help Cactus in aggregate? | Validates opportunity #2 with numbers, not theory | End-to-end cascade benchmark: gate on stick vs gate on CPU, same Mirage brain |
| Can the wide-GEMM hang be worked around with host-side tiling? | Unlocks larger kNN tables | Implement tiled GEMM (N≤10k per tile) and measure |
| How much of Mirage's 22 MB is encoder vs decoder? | Decides whether a stick-side encoder slice is conceivable | Inspect `mirage.safetensors` tensor shapes and total per-block parameter counts |

---

## 11. Reproducing our numbers

```powershell
# device lane + a sync and a pipelined bench on the stick
uv run ntvbench devices
uv run ntvbench bench onnx-models/mobilenetv2-7 --device ncs1-myriad
tools\ncs1\.venv\Scripts\python.exe tools\ncs1\bridge_runner.py --model models/ncs1/mobilenetv2-7.onnx `
    --device myriad --requests 1 --warmup 5 --runs 50      # sync  (~15.7 inf/s)
tools\ncs1\.venv\Scripts\python.exe tools\ncs1\bridge_runner.py --model models/ncs1/mobilenetv2-7.onnx `
    --device myriad --requests 2 --warmup 5 --runs 60      # pipelined (~23.1 inf/s)

# capability shim proof (HardSigmoid: MYRIAD rejects, HETERO runs)
tools\ncs1\.venv\Scripts\python.exe tools\ncs1\capability_demo.py

# the cost decomposition from §6 (fixed dispatch vs input transfer vs compute)
tools\ncs1\.venv\Scripts\python.exe tools\ncs1\profile_split.py

# measured envelope + calibration constants
#   docs/NCS1_ROADMAP.md   tools/ncs1/calibration.json   results/*myriad*.json
```

Records land in `results/`; the report tables in this doc are built from them.

---

## 12. Sources

**Workspace (primary, measured here):** `results/*myriad*.json`; `docs/NCS1_ROADMAP.md`;
`tools/ncs1/calibration.json`; `tools/ncs1/ov2020/.../include/vpu/{myriad_plugin_config,vpu_plugin_config}.hpp`;
`tools/ncs1/ov2020/.../XLink/pc/protocols/usb_boot.c` (boot-table PID support).

**External:**
- PyPI `cactus-compute` 4.2.6 JSON API (Apache-2.0, Python ≥3.9, 168 KB wheel, description, extras).
- `huggingface.co/Cactus-Compute/mirage` model card + **`config.json`** (2-bit, dims, layers, tasks, languages).
- `huggingface.co/Cactus-Compute/cactus` runtime repo (17 platform targets, C API, `.cact` weights, caching).
- `cactuscompute.com` (Mirage 22 MB; Whistle 16.9 MB; Needle 3 8–29 MB; task list).
- `cactuscompute.com/blog/needle-python-docs` (describe/extract API, serialized calls, VAD/wake-word advice,
  `--bits 2|4`, version-tied `.cact`, Needle tps/RAM).
- `en.wikipedia.org/wiki/Movidius` (**80–150 GFLOPS at 1 W**, Fathom board; Myriad X has a Neural Compute Engine;
  NCS2 "up to eight times").
- Intel NCS2 product page via Wayback (16 SHAVE + dedicated DNN accelerator, USB 3.0).
- `github.com/movidius/ncsdk` README (legacy SDK; no benchmarks).

**Could not retrieve (named as gaps, not filled):** Intel's MYRIAD OpenVINO benchmark result tables (404 /
overview-only); the NCS2 product-brief PDF (dead link); any condition-matched published MobileNetV2 FPS for NCS1.