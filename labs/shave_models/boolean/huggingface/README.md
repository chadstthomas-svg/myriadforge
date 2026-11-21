---
language:
  - en
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE
tags:
  - intel-ncs1
  - myriad-2
  - openvino
  - fp16
  - decision-model
  - rule-engine
---

# MagicEightBall for Intel NCS1

![Eight, a small black billiard-ball automaton, and Relay, a floating green microchip, inspect a glowing circuit on a violet workbench.](./assets/workbench.webp)

MagicEightBall reads up to **131,072 Needle 3 tokens** and returns a Boolean answer, a category, and a proposed action. The host counts tokens and extracts evidence. This private development preview includes the **512-byte** native build that passed the sustained NCS1 run and a **480-byte** candidate that passed direct and packaged physical replay.

## Three decisions at the workbench

![Eight and Relay show three support cases. An authorized duplicate charge proposes a billing review. A later hold clears the proposal. An unrelated request leaves category and action empty.](./assets/relay-decisions.webp)

- **Approve a billing review.** `Duplicate charge. Review authorized.` returns `true`, `billing`, and `review_billing`.
- **Clear a proposed action.** `Duplicate charge. Review authorized. Do not route.` returns `false`, `billing`, and `null`.
- **Abstain when evidence is absent.** `General help requested.` returns `false`, `null`, and `null`.

The three fields are `may_route`, `category`, and `action`. The caller receives the proposal as data. [Support policy](./support.json) · [Still image](./assets/relay-decisions-still.webp)

## Run on the original stick

Use a Windows NCS1 toolchain with OpenVINO 2020.3.2, Python 3.7, NumPy, and SentencePiece 0.1.99. Download or clone these files together and keep the tokenizer beside `runtime.py`. Replace the toolchain path with your installation. [MyriadForge setup guide](https://github.com/chadstthomas-svg/myriadforge/blob/main/docs/NCS1_ROADMAP.md)

```powershell
$toolchain = 'D:/NTVStudy/tools/ncs1'
$python = Join-Path $toolchain '.venv/Scripts/python.exe'
& $python runtime.py bundles/magiceightball.m8b --toolchain $toolchain --release-stick ncs1 --text 'duplicate charge. review authorized.'
```

For a batch, use `--serve` and send one JSON line containing a `documents` array. For updates to an existing document, use `--stream` and send `{"append":"next fragment"}` per line. The final accepted document may contain exactly 131,072 tokens. An append that would exceed the limit is rejected and preserves the prior document state. [Runtime source](./runtime.py) · [Physical stream replay](./measurements/release-stream-ncs1.json)

## Measurements

The release passed all 128 support-policy assignments on a physical NCS1, including full-context fixtures, a late veto, and rejection at 131,073 tokens. The matched comparison used 31 warmed repeats per workload on the same Intel Core i7-12650H Windows host. All rows below use the same tokenizer and evidence extraction. [Physical release replay](./measurements/release-vector-ncs1.json) · [Complete comparison report](./measurements/ncs1-cpu-comparison.json)

| Backend | Decision artifact | 128 short decisions, p50 / p95 | Fresh 131,072-token authorization, p50 / p95 |
|---|---:|---:|---:|
| Physical NCS1, one request | 512-byte native executable | 6.43 / 10.73 ms | 44.24 / 55.55 ms |
| OpenVINO CPU, matched Gather | 258-byte source ONNX | 3.91 / 5.99 ms | 44.25 / 55.24 ms |
| CPU Boolean interpreter | 714-byte source policy | 4.57 / 8.26 ms | 39.26 / 49.48 ms |
| CPU lookup | 128-byte generated table | 3.82 / 6.95 ms | 37.63 / 50.03 ms |
| CPU decision diagram | 46-byte generated diagram | 4.03 / 7.37 ms | 37.92 / 52.50 ms |
| Compiled CPU lookup | 128-byte table and 2,560-byte native library | 3.29 / 4.02 ms | 33.58 / 38.09 ms |
| OpenVINO 2020.3.2 CPU Gather | Same 258-byte source ONNX | 4.04 / 4.80 ms | 33.53 / 41.25 ms |

The CPU table and diagram figures cover decision data. Their runners also use the policy, tokenizer, and output mapping. The NCS1 took 223.70 ms median for the same 128 decisions issued one by one, about 35 times as long as the batch. In the streaming replay, short appends near the context limit took **1.90 ms median** after ingesting the initial prefix. Fresh full-context requests spend most of their time in host preparation. [Complete comparison report](./measurements/ncs1-cpu-comparison.json) · [Physical stream replay](./measurements/release-stream-ncs1.json)

The **480-byte radix Gather** is the smallest native artifact verified in this comparison. It passed all 128 assignments, two exact-limit documents and overflow rejection on NCS1. It took 5.31 ms median for one batch of 128 short decisions, compared with 5.37 ms for its 512-byte padded control. The packaged candidate also passed the vector and mixed-context replay. Its separate sustained run is in progress. [Physical compact comparison](./measurements/compact-gather-physical-ncs1.json) · [Packaged candidate replay](./measurements/compact-release-vector-ncs1.json) · [Size and competitor comparison](./COMPETITORS.md)

A **30-minute** mixed workload on the physical NCS1 completed **107,104 native requests** and **3,507,656 checked decisions**, with **26,776 overflows rejected**. The run exercised full documents, incremental vetoes and 128-decision batches on the resident release. [Physical soak report](./measurements/soak-release-30m-ncs1.json)

## Industry use cases

Two starter packs are ready to try on one resident NCS1 graph. **Logistics and fulfillment** triages missing shipments and address exceptions. **Software on-call** triages network and capacity incidents. The record-mode examples take evidence from a declared field, so an archived approval or hold stays outside the decision. [Run the starter packs](./industries/README.md) · [Labeled cases](./industries/scenarios.json)

Both CPU and physical NCS1 agreed with all **18 of 18** simulated outcomes, including two 131,072-token cases with late vetoes. MagicEightBall issued zero proposals on the ten expected no-action cases. A baseline using the same rules without approval or veto checks issued seven. These results are from hand-authored simulations. [CPU simulation](./measurements/industry-simulation-cpu.json) · [Physical NCS1 simulation](./measurements/industry-simulation-ncs1.json)

## Scope and downloads

The shipped English support policy recognizes specific phrases and evaluates authored Boolean rules. Use it for deterministic routing and action proposals within that policy. General support chat, interpretation of new intents, and execution of actions belong in the surrounding application. The native NCS1 evidence applies to the original stick and the pinned runtime. The archive carries a separately compiled legacy NCS2 image.

| Artifact | Bytes |
|---|---:|
| [Native release archive](./bundles/magiceightball.m8b), including both targets | 735 |
| [Compact candidate archive](./bundles/magiceightball-ncs1-compact.m8b), with the 480-byte NCS1 image | 699 |
| [Shared tokenizer](./needle3-tokenizer.m8t.xz) | 23,412 |
| Stock archive plus tokenizer | **24,147** |
| Compact archive plus tokenizer | **24,111** |

Each total covers one archive and the shared tokenizer. Python, OpenVINO, firmware, runner source and the artwork have separate sizes. [Checksums](./SHA256SUMS.txt)

MyriadForge source and policy use the MIT license. The shared tokenizer restores the original Apache-2.0 Needle 3 tokenizer bytes and verifies the pinned SHA-256. [License](./LICENSE) · [Tokenizer provenance](./NOTICE.txt)
