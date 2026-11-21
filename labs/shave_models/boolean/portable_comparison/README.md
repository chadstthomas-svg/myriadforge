# MagicEightBall on CPU

The three portable profiles share the seven-input Boolean policy and the Needle tokenizer. They accept up to 131,072 source tokens per document. The host counts tokens and extracts facts from the full document. The small model file encodes the decision head that maps those facts to answers, categories, and actions.

| Profile | Model file | CPU runner | Head capacity |
| --- | ---: | --- | ---: |
| ONNX Radix Gather | 192 B | ONNX Runtime CPUExecutionProvider | 128 |
| ONNX FP32 dense reference | 1,832 B | ONNX Runtime CPUExecutionProvider | 1 |
| GGUF I8 | 1,024 B | MagicEightBall table reader using `gguf` and NumPy | 1 |

The displayed sizes cover model files. Each runnable profile also requires the tokenizer and host code. The ONNX profiles also read their JSON manifests. The GGUF uses the custom `magiceightball` architecture and table layout described by the included reader.

## Measured results

Latency is p50 / p95 milliseconds per workload on an Intel Core i7-12650H running Windows 11, Python 3.13.7, ONNX Runtime 1.24.1, SentencePiece 0.2.2, NumPy 2.5.4 and GGUF 0.19.0.

| Profile | 128 short, sequential | 128 short, native capacity | 128 prepared, native capacity | Fresh 131,072, authorized | Fresh 131,072, late veto |
| --- | ---: | ---: | ---: | ---: | ---: |
| ONNX Radix Gather | 7.93 / 8.84 | 2.55 / 3.01 | 0.25 / 0.30 | 23.21 / 24.60 | 23.32 / 25.20 |
| ONNX FP32 dense reference | 4.80 / 5.25 | 4.80 / 5.25 | 2.06 / 2.23 | 22.70 / 25.14 | 22.54 / 25.04 |
| GGUF I8 | 2.90 / 3.61 | 2.90 / 3.61 | 0.35 / 0.39 | 22.55 / 24.17 | 22.39 / 25.70 |

All three agreed on all 128 prepared and short-text assignments, both exact full-context outcomes, and rejection at 131,073 tokens. The Radix profile took 3.11× as long to handle the same 128 short documents sequentially as in a native batch. Host preparation accounted for more than 98% of the corresponding median application wall times for fresh full-context inputs.

## Workloads

- **128 sequential short documents** submits the same shuffled 128 documents one at a time in all three runners.
- **128 short documents at native capacity** submits the same documents in one call on Radix Gather. The dense reference and GGUF use their sequential result because they expose one decision per call.
- **128 prepared assignments** gives the heads the same precomputed Boolean inputs. The measurement includes packing, inference or lookup, and output decoding. Radix Gather makes one call while the other profiles make 128 calls.
- **One 131,072-token document** prepares and decides a fresh document each time. One case grants authorization at the end and another cancels it at the end. A separate correctness check verifies rejection at 131,073 tokens.

Application wall time includes token counting, policy fact extraction, head execution, and result construction. Each case runs after model loading. Fixture generation, validation, file and console I/O are outside the measurements. The benchmark warms up four times, takes 31 samples per case, and alternates the order of profiles and cases by round. It records latency in milliseconds per workload, along with host prep time and decision/decode time for application requests. Raw samples, p50, p95, min and max are included in JSON. ONNX Runtime uses one intra-op thread. These results describe runtime behavior on the measured CPU and exact fixtures.

## Reproduce

Use the [same pinned corpus](../README.md#rebuild-and-benchmark) as the NCS1 release benchmark. From the repository root, run these in the modern Python environment with `numpy`, `sentencepiece`, `onnxruntime`, and `gguf` installed.

```powershell
python labs/shave_models/boolean/export_portable.py ./exports --corpus ./decision-corpus.txt
python labs/shave_models/boolean/portable_comparison/portable_benchmark.py --exports ./exports --corpus ./decision-corpus.txt --output ./portable-cpu.json
```

Run the second command with `--check-only` to verify all profiles without collecting a latency series. The corpus and tokenizer hashes, profile hashes, shared fixture contract hash, exact full-context outcomes, and overflow check are in the report.

Each profile can evaluate the same text from the command line after export. All three propose `review_billing` with this input.

```powershell
$text = 'Duplicate charge. Review authorized.'
python labs/shave_models/boolean/portable_runtime.py ./exports/onnx/model.onnx --text $text
python labs/shave_models/boolean/portable_runtime.py ./exports/fp32/model.onnx --text $text
python labs/shave_models/boolean/portable_runtime.py ./exports/gguf/model.gguf --text $text
```
