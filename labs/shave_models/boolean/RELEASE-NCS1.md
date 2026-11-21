# MagicEightBall for Intel Neural Compute Stick

MagicEightBall evaluates staged Boolean rules over up to **131,072 Needle 3 tokens** and returns an answer, a category, and a proposed action. The physical NCS1 evaluates the compiled decision graph. The host counts tokens, extracts evidence, and decodes the result. The proposal is data for the caller to handle.

The compact native NCS1 executable is **480 bytes**, including **32 bytes** of FP16 coefficients. Its **699-byte** archive plus the **23,412-byte** shared tokenizer total **24,111 bytes**. The stock **512-byte** native executable also ships in a **735-byte** archive. Both passed replay and a 30-minute mixed-workload run on the physical NCS1. Each archive also carries a legacy Myriad X image compiled offline. Host code, Python, OpenVINO, and firmware have separate sizes. [Compact archive replay](measurements/compact-release-vector-ncs1.json) · [Stock release replay](measurements/release-vector-ncs1.json)

## Run on Windows

Use the original NCS1 with the working OpenVINO 2020.3.2 MYRIAD lane and a supported driver. The runner uses NumPy and `sentencepiece==0.1.99` in the lane's Python 3.7 environment. Keep `needle3-tokenizer.m8t.xz` next to `runtime.py`. The first result in the example proposes `review_billing`. The later veto clears that action.

```powershell
$toolchain = 'D:/NTVStudy/tools/ncs1'
$python = Join-Path $toolchain '.venv/Scripts/python.exe'
$b = 'labs/shave_models/boolean'
$archive = "$b/bundles/magiceightball-ncs1-compact.m8b"
& $python "$b/runtime.py" $archive --toolchain $toolchain --release-stick ncs1 --text 'duplicate charge. review authorized.'
@(
  '{"id":"open","append":"duplicate charge. review authorized."}'
  '{"id":"veto","append":" do not route"}'
) | & $python "$b/runtime.py" $archive --toolchain $toolchain --release-stick ncs1 --stream
```

## Matched CPU comparison

All seven backends shared the same tokenizer and evidence extraction and agreed on all 128 assignments, raw rule bits, two full-context fixtures and overflow rejection. Timings are **p50 / p95 milliseconds** across 31 warmed repeats per workload on the same i7-12650H Windows host. Each backend ran in its own session, with the workload order alternating within the session. The prepared head includes packing and unpacking. The full-text timings include host work, inference and the returned record. [Raw comparison](measurements/ncs1-cpu-comparison.json) · [Compiled C lookup](measurements/compare-compiled-table-cpu.json) · [Legacy OpenVINO CPU](measurements/compare-openvino-2020-cpu.json)

| Backend | Model or decision data | Prepared 128 | 128 short, one batch | Fresh 131K, authorized | Fresh 131K, late veto |
|---|---|---:|---:|---:|---:|
| Physical NCS1 Gather | 512-byte native executable | 2.28 / 2.77 | 6.43 / 10.73 | 44.24 / 55.55 | 41.48 / 55.04 |
| OpenVINO 2022.3.1 CPU Gather | Same 258-byte source ONNX, 72-byte IR weights | 0.52 / 0.98 | 3.91 / 5.99 | 44.25 / 55.24 | 43.94 / 58.16 |
| OpenVINO 2020.3.2 CPU Gather | Same ONNX and IR weights | 0.82 / 1.17 | 4.04 / 4.80 | 33.53 / 41.25 | 32.95 / 41.37 |
| CPU Boolean interpreter | 714-byte source policy | 0.99 / 1.58 | 4.57 / 8.26 | 39.26 / 49.48 | 35.80 / 44.25 |
| Python CPU lookup table | 128-byte generated table | 0.41 / 1.46 | 3.82 / 6.95 | 37.63 / 50.03 | 36.78 / 48.96 |
| Compiled C CPU lookup table | Same 128-byte table, separate 2,560-byte DLL | 0.015 / 0.032 | 3.29 / 4.02 | 33.58 / 38.09 | 32.91 / 37.84 |
| CPU decision diagram | 46-byte generated diagram | 0.55 / 0.97 | 4.03 / 7.37 | 37.92 / 52.50 | 38.95 / 52.47 |

The CPU table and diagram sizes count the generated decision data. Both benchmark paths also use the host policy manifest, output mapping, tokenizer, and runner. The diagram shares identical branches and encodes exact raw bits at the leaves. The legacy CPU control uses the same OpenVINO release as the NCS1. The C lookup uses the same table as Python. Its prepared timing includes the Python-to-C call, packing, lookup, and unpacking. [Reproduction script](benchmark_compare.py) · [Compiled C runner and build commands](benchmark_c_cpu.py)

Run `benchmark_compare.py` with the [pinned corpus](README.md#rebuild-and-benchmark), `--repeats 31 --warmups 4`, and a separate `--output` per backend. Select `--backend ncs1 --expect-cold-ncs1 --toolchain <path>` with the original stick freshly enumerated. Use the 2022.3.1 Python environment for `--backend openvino-cpu`. The 2020.3.2 Python environment also runs the `policy-cpu`, `table-cpu`, and `tree-cpu` controls.

The NCS1 took **223.70 ms** median to issue 128 short requests one by one. Packing the same decisions into one request took **6.43 ms**, about **35× faster**. The compiled C lookup completed the matched batch in **3.29 ms**. Host preparation dominated the fresh authorized 131K case, taking **41.82 ms** median on NCS1 and **33.48 ms** for the C lookup. On the append path, the same stick took **1.90 ms** median for short updates near the limit after an initial full prefix. The two-bank graph evaluated both policies in one 512-byte native graph and verified all 256 assignments on NCS1. [Matched batch results](measurements/ncs1-cpu-comparison.json) · [Compiled C timings](measurements/compare-compiled-table-cpu.json) · [Physical streaming](measurements/release-stream-ncs1.json) · [Physical policy banks](measurements/two-banks-ncs1.json)

## Portable runtime examples

[Run the ONNX radix, FP32 dense, and GGUF examples on CPU](portable_comparison/README.md). The smallest portable model file is 192 bytes and runs all 128 short decisions in 2.55 ms median under ONNX Runtime. The compact NCS1 graph handled the same 128 short decisions in one USB request at 5.31 ms median. All profiles agreed on the 128 assignments, the exact-length authorization and veto, and rejection at 131,073 tokens. These are application wall times from separate sessions on one host using identical fixtures. [CPU report](measurements/portable-cpu-comparison.json) · [Compact NCS1 report](measurements/compact-gather-physical-ncs1.json)

## Other models and workloads

Published compact systems give useful size context. fastText.zip evaluated learned text classifiers at 16 KiB. Bonsai evaluated 2 KB models on binary feature classification. FastGRNN reported a 1 KB wake-word model. These studies use different tasks and measure different storage layers. Use the matched controls above for MagicEightBall latency or decision-data claims. [fastText.zip](https://arxiv.org/abs/1612.03651) · [Bonsai](https://proceedings.mlr.press/v70/kumar17a.html) · [FastGRNN](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab013ca67cf2d50796b0c11d1b8bc95d-Abstract.html)

MyriadForge also has physical NCS1 results for learned voice-command classification, text intent, form localization and a browser-control ranker. Those models use different graphs, fixtures, and scoring. In the measured MobileNet pipeline, asynchronous requests raised stick throughput from 15.7 to 23.2 inferences/s. The current MagicEightBall runs document the exact decision offload and its batching benefit. They contain no system energy measurement. [Other physical model results](../README.md) · [MobileNet throughput](../../../docs/NCS1_ROADMAP.md)

The [competitor comparison](COMPETITORS.md) records which published workloads fit the original stick, the scope of the size claim, and the exact alternatives tested with the stock compiler.

## Industry starter packs

[Dealership service intake, manufacturing maintenance, and store-security review](industries/frontline/README.md) share one native graph. CPU and NCS1 matched all 27 labeled cases and all 384 raw Boolean assignments, including a late veto at 131,072 tokens for each bank. The staged policies made zero proposals in 15 no-action cases, while the ungated control made 12. The cases are hand-authored simulations. [CPU report](measurements/industry-simulation-frontline-cpu.json) · [Physical NCS1 report](measurements/industry-simulation-frontline-ncs1.json)

The logistics and fulfillment pack triages shipment and address exceptions. The software on-call pack triages network and capacity incidents. Both ran from one resident 512-byte native graph with named policy banks. The 18 hand-authored cases matched their labels on both CPU and physical NCS1. Among the ten no-action cases, there were zero wrong proposals with the staged policies and seven when the same baseline ignored approval and veto. [Policies, usage, and simulation reports](industries/README.md)

## Sustained run

Use `soak_ncs1.py` with the same pinned corpus from a cold NCS1. It keeps one model resident, validates a short 128-decision batch, a fresh full-context decision, and an exact-limit stream with a late veto on every cycle. It rejects the next token before inference and writes periodic latency summaries and decision counts.

The stock archive passed **30 minutes** on the physical NCS1 with **107,104 native requests**, **3,507,656 checked decisions** and **26,776 rejected overflows**. Across the mixed soak, the 128-decision short batch took 5.51 / 6.51 ms and the short append took 1.71 / 2.08 ms (p50 / p95). [Stock soak report](measurements/soak-release-30m-ncs1.json)

The compact archive passed **30 minutes** with **98,484 native requests**, **3,225,351 checked decisions** and **24,621 rejected overflows**. The short batch took 5.52 / 8.83 ms, and the short append took 1.78 / 2.33 ms (p50 / p95). [Compact soak report](measurements/soak-compact-30m-ncs1.json)

```powershell
& $python "$b/soak_ncs1.py" --archive $archive --toolchain $toolchain --corpus $corpus --duration 1800 --interval 30 --output soak-ncs1.json
```

## Provenance

The project code is MIT-licensed. The tokenizer is Cactus Compute's Apache-2.0 Needle 3 tokenizer, losslessly repackaged and verified against source SHA-256 `97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25`. The upstream model weights are outside this distribution. [Project license](../../../LICENSE) · [Pinned tokenizer](https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/tokenizer/tokenizer.model) · [Tokenizer license](../bundles/needle-ui/LICENSE)
