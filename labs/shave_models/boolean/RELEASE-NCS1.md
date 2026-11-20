# MagicEightBall for Intel Neural Compute Stick

MagicEightBall evaluates staged Boolean rules over up to **131,072 Needle 3 tokens** and returns an answer, a category, and a proposed action. The physical NCS1 evaluates the compiled decision graph. The host counts tokens, extracts evidence, and decodes the result. The proposal is data for the caller to handle.

The tested native NCS1 executable is **512 bytes**, including **64 bytes** of FP16 coefficients. The distribution data are the **735-byte** `.m8b` archive and the **23,412-byte** shared tokenizer. The archive also carries a legacy Myriad X image compiled offline. This release has physical NCS1 results. The NCS2 image has software and compiler evidence. Host code, Python, OpenVINO, and firmware have separate sizes. [Release replay](measurements/release-vector-ncs1.json)

## Run on Windows

Use the original NCS1 with the working OpenVINO 2020.3.2 MYRIAD lane and a supported driver. The runner uses NumPy and `sentencepiece==0.1.99` in the lane's Python 3.7 environment. Keep `needle3-tokenizer.m8t.xz` next to `runtime.py`. The first result in the example proposes `review_billing`. The later veto clears that action.

```powershell
$toolchain = 'D:/NTVStudy/tools/ncs1'
$python = Join-Path $toolchain '.venv/Scripts/python.exe'
$b = 'labs/shave_models/boolean'
$archive = "$b/bundles/magiceightball.m8b"
& $python "$b/runtime.py" $archive --toolchain $toolchain --release-stick ncs1 --text 'duplicate charge. review authorized.'
@(
  '{"id":"open","append":"duplicate charge. review authorized."}'
  '{"id":"veto","append":" do not route"}'
) | & $python "$b/runtime.py" $archive --toolchain $toolchain --release-stick ncs1 --stream
```

## Matched CPU comparison

All five backends shared the same tokenizer and evidence extraction and agreed on all 128 assignments, raw rule bits, two full-context fixtures and overflow rejection. Timings are **p50 / p95 milliseconds** across 31 warmed repeats per workload on the same i7-12650H Windows host. Each backend ran in its own session, with the workload order alternating within the session. The prepared head includes packing and unpacking. The full-text timings include host work, inference and the returned record. [Raw comparison](measurements/ncs1-cpu-comparison.json)

| Backend | Model or decision data | Prepared 128 | 128 short, one batch | Fresh 131K, authorized | Fresh 131K, late veto |
|---|---|---:|---:|---:|---:|
| Physical NCS1 Gather | 512-byte native executable | 2.28 / 2.77 | 6.43 / 10.73 | 44.24 / 55.55 | 41.48 / 55.04 |
| OpenVINO CPU Gather | Same 258-byte source ONNX, 72-byte IR weights | 0.52 / 0.98 | 3.91 / 5.99 | 44.25 / 55.24 | 43.94 / 58.16 |
| CPU Boolean interpreter | 714-byte source policy | 0.99 / 1.58 | 4.57 / 8.26 | 39.26 / 49.48 | 35.80 / 44.25 |
| CPU lookup table | 128-byte generated table | 0.41 / 1.46 | 3.82 / 6.95 | 37.63 / 50.03 | 36.78 / 48.96 |
| CPU decision diagram | 46-byte generated diagram | 0.55 / 0.97 | 4.03 / 7.37 | 37.92 / 52.50 | 38.95 / 52.47 |

The CPU table and diagram sizes count the generated decision data. Both benchmark paths also use the host policy manifest, output mapping, tokenizer, and runner. The diagram shares identical branches and encodes exact raw bits at the leaves. OpenVINO CPU uses 2022.3.1 and the NCS1 uses 2020.3.2. The timings compare those working paths on one host. [Reproduction script](benchmark_compare.py)

Run `benchmark_compare.py` with the [pinned corpus](README.md#rebuild-and-benchmark), `--repeats 31 --warmups 4`, and a separate `--output` per backend. Select `--backend ncs1 --expect-cold-ncs1 --toolchain <path>` with the original stick freshly enumerated. Use the 2022.3.1 Python environment for `--backend openvino-cpu`. The 2020.3.2 Python environment also runs the `policy-cpu`, `table-cpu`, and `tree-cpu` controls.

The NCS1 took **223.70 ms** median to issue 128 short requests one by one. Packing the same decisions into one request took **6.43 ms**, about **35× faster**. The CPU lookup remained faster in that matched batch at **3.82 ms**. NCS1 host preparation took **41.82 ms** median for the fresh authorized 131K case. On the append path, the same stick took **1.90 ms** median for short updates near the limit after an initial full prefix. The two-bank graph evaluated both policies in one 512-byte native graph and verified all 256 assignments on NCS1. [Matched batch results](measurements/ncs1-cpu-comparison.json) · [Physical streaming](measurements/release-stream-ncs1.json) · [Physical policy banks](measurements/two-banks-ncs1.json)

## Other models and workloads

Published compact systems give useful size context. fastText.zip evaluated learned text classifiers at 16 KiB. Bonsai evaluated 2 KB models on binary feature classification. FastGRNN reported a 1 KB wake-word model. These studies use different tasks and measure different storage layers. Use the matched controls above for MagicEightBall latency or decision-data claims. [fastText.zip](https://arxiv.org/abs/1612.03651) · [Bonsai](https://proceedings.mlr.press/v70/kumar17a.html) · [FastGRNN](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab013ca67cf2d50796b0c11d1b8bc95d-Abstract.html)

MyriadForge also has physical NCS1 results for learned voice-command classification, text intent, form localization and a browser-control ranker. Those models use different graphs, fixtures, and scoring. In the measured MobileNet pipeline, asynchronous requests raised stick throughput from 15.7 to 23.2 inferences/s. The current MagicEightBall runs document the exact decision offload and its batching benefit. They contain no system energy measurement. [Other physical model results](../README.md) · [MobileNet throughput](../../../docs/NCS1_ROADMAP.md)

## Provenance

The project code is MIT-licensed. The tokenizer is Cactus Compute's Apache-2.0 Needle 3 tokenizer, losslessly repackaged and verified against source SHA-256 `97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25`. The upstream model weights are outside this distribution. [Project license](../../../LICENSE) · [Pinned tokenizer](https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/tokenizer/tokenizer.model) · [Tokenizer license](../bundles/needle-ui/LICENSE)
