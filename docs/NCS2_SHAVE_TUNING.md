# NCS2 SHAVE tuning and language-model memory

Myriad X exposes SHAVE cores, CMX scratchpad slices, and stream count through the OpenVINO 2022.3.1 MYRIAD plugin. Allocations are per stream. Set at most 16 SHAVEs and 16 CMX slices across streams, with at least one CMX slice per allocated SHAVE. This is a compiler configuration. The hardware also has a dedicated neural compute engine. [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/140109/intel-neural-compute-stick-2/specifications.html) · [Runtime support](https://www.intel.com/content/www/us/en/support/articles/000091265/software.html)

## Tune a model

Use the isolated NCS2 Python environment. Compare latency with one request at a time and throughput with two requests in flight. The default grid sweeps equal allocations across one and two streams and samples extra CMX allocations. Add `--alloc SHAVES:CMX:STREAMS` for a custom grid. The conservative default remains 16:16:1.

```powershell
$ncs2 = 'tools/ncs2/.venv/Scripts/python.exe'
$model = 'models/ncs2/mobilenetv2-12.onnx'
& $ncs2 tools/ncs2/tune.py $model --check top1 --out results/ncs2-mobilenet.json
& $ncs2 tools/ncs2/bridge_runner.py --model $model --device myriad --tuning-profile results/ncs2-mobilenet.json --objective latency
```

Profiles bind to the model SHA-256, including an IR's adjacent BIN, the device family, and the OpenVINO runtime version. The bridge applies the selected allocation. Its existing benchmark submits synchronous requests. Use the tuner's throughput figure for the two-request queue or submit two asynchronous requests in an application.

The tuner compares outputs with OpenVINO CPU before timing. Numeric parity is the default. Supply `--feeds` as a JSON array of named input mappings for representative cases, especially for integer inputs with a constrained range. Top-1 checking is available for classification logits and records the largest observed absolute logit difference. The tuner keeps individual timings and repeats finalists and the stock profile in two randomized confirmation rounds.

## Measured MobileNetV2 results

Tested on one Myriad X with OpenVINO 2022.3.1. Twenty-nine of 32 initial configurations passed. The other three had more SHAVEs than CMX slices, which the runtime rejected. The revised shim validates that rule before compiling. The classifier agreed with the CPU's top choice on three fixed tensors. Its logits differed numerically.

| Allocation per stream | One-request p95 | Two-request throughput |
| --- | ---: | ---: |
| 16 SHAVEs, 16 CMX, 1 stream | **23.30 ms** | 47.71 inferences/s |
| 4 SHAVEs, 8 CMX, 2 streams | 24.30 ms | **70.35 inferences/s** |

These are medians of three independently compiled runs, each with 50 timed requests per metric. Two streams increased measured throughput by **47%**, with a roughly 1 ms trade-off in single-request p95. This is the best confirmed allocation on these inputs and queue settings. Use a new sweep for a different graph or workload.

MagicEightBall's prepared Gather head also passed exact CPU comparison at every tested setting. Its two-request throughput rose from 655.5 to 673.8 requests/s, with overlapping trial ranges. The measured difference is too small to establish a speedup on this workload. These timings cover device requests with 128 prepared inputs, and exclude document scanning and tokenization. [MobileNetV2 trials](../tools/ncs2/measurements/mobilenetv2-tuning.json) · [MagicEightBall trials](../tools/ncs2/measurements/magiceightball-tuning.json)

## Language-model budget

Treat 1,024 MiB as a planning budget. The tested MYRIAD plugin reports device identity and thermal telemetry. Its supported metrics provide no allocator free-byte counter. On-device limits also depend on the compiler, transformed weights, activation buffers, and the largest operator.

```powershell
python tools/ncs2/memory_budget.py estimate path/to/config.json --context 2048 --budget-mib 1024 --reserve-mib 128
& $ncs2 tools/ncs2/memory_budget.py probe 64 --out results/ncs2-64-mib.json
```

The estimator computes bias-free Llama weights and batch-one key/value cache with the model's actual grouped-query dimensions. With 2,048 tokens and an assumed 128 MiB runtime reserve:

| Model | FP16 weights | FP16 KV cache | Headroom within 1,024 MiB |
| --- | ---: | ---: | ---: |
| SmolLM2-135M | 256.6 MiB | 45 MiB | 594.4 MiB |
| SmolLM2-360M | 690.1 MiB | 80 MiB | 125.9 MiB |
| SmolLM2-1.7B | 3,264.2 MiB | 384 MiB | Over budget by 2,752.2 MiB |

These are analytical storage estimates. The FP16 compiler and operator results determine whether a model runs. An INT4 file estimate assumes ideal packing, while the 2022.3.1 MYRIAD runtime would need kernels that keep those weights compressed on the device. [135M config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json) · [360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/f8027fd0eaeea54caa13c31d31b9fdc459c38b49/config.json) · [1.7B config](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/effd688a12921b4cc83e3312b6feb579f70f9c71/config.json)

### Physical weight residency

One NCS2 graph completed six inferences with **224 MiB of embedded FP16 constants**, agreeing with the NumPy control. The seven 4,096-wide dense layers reached a 28.19 ms warm median. A single 256 MiB graph timed out twice. A pair of 96 MiB graphs worked together, while the pair of 128 MiB graphs timed out while loading the second graph. The probe uses a bounded child process so an overdue allocation attempt returns a result.

The evidence establishes 224 MiB for this dense workload. Graph layout, firmware buffers, and runtime loading can shift the boundary. SmolLM2-135M's FP16 weights alone exceed this tested payload. Prioritize a smaller trained decoder or a partition with more headroom while narrowing down the operator and loading limits. [Dense probes](../tools/ncs2/measurements/llm-residency.json)

## Trained model replay

The follow-up cached decoder runs all transformer blocks and the vocabulary projection on MYRIAD. See the [full-decoder benchmark and instructions](NCS2_FULL_DECODER.md).

The static decoder uses a 32-token window and recomputes the prefix each step. The host handles tokenization, embedding lookup, and sampling. The vocabulary-projection partition also keeps attention and transformer blocks on the CPU.

| Checkpoint and placement | Greedy agreement on two 12-token prompts | Median warm step |
| --- | ---: | ---: |
| TinyStories 260K, NCS2 vocabulary projection | 24/24 | 2.23 ms |
| TinyStories 15M, NCS2 vocabulary projection | 24/24 | 10.90 ms |
| TinyStories 260K, entire transformer and projection on NCS2 | 0/24 at the CPU prefixes | 55.89 ms, invalid output |
| TinyStories 15M, entire transformer and projection on NCS2 | Trial stopped after more than 120 seconds | — |

The 15M vocabulary projection places 17.58 MiB of FP16 weights on the NCS2 and takes about 4.00 ms per warm request. The full CPU step took about 7.1 ms with the same window and inputs. The physical offload proves that part of a trained decoder works. CPU delivers the lower latency for this partition. A useful next experiment is a cache-aware decoder block with a small vocabulary and one device request per token.

```powershell
# Use the NCS2 environment with SentencePiece installed.
& $ncs2 tools/ncs2/tiny_llama.py models/stories260K.bin models/tok512.model --partition head --out results/tiny260k.json
& $ncs2 tools/ncs2/tiny_llama.py models/stories15M.bin models/tokenizer.model --partition head --out results/tiny15m.json
```

Use the pinned model revision and matching tokenizer for each checkpoint. The evidence contains file hashes, generated token IDs, CPU comparisons, and timings. [TinyStories checkpoints](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add) · [32K tokenizer and checkpoint format](https://github.com/karpathy/llama2.c) · [Measured replay](../tools/ncs2/measurements/llm-residency.json)
