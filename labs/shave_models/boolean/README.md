# Boolean decision model for Intel Neural Compute Stick

The packed support model accepts one UTF-8 document of up to **131,072 Needle 3 tokens** and returns a yes/no answer, a category and an optional proposed action. It runs the decision lookup on the VPU in one request. The host counts tokens, extracts literal evidence and decodes the result. The proposed action is returned as data.

## Edit the policy

[`support.json`](support.json) combines evidence in successive named rules. JSON-quoted phrases are evidence predicates. A reference such as `@billing` uses an earlier rule. Operators bind in the order `NOT`, `AND`, `OR`. Parentheses override that order. The first matching category and action win in their respective groups. Each named answer returns a Boolean. Absent categories and actions return `null`.

```text
billing     = "charged twice" OR "duplicate charge"
case_found  = @billing OR @delivery
ready       = @case_found AND @permission AND NOT @hold
may_route   = @ready
```

Evidence may occur anywhere in the document. A phrase matches adjacent lexemes after Unicode case folding. Whitespace may vary and punctuation counts as evidence. Document-wide matching treats repeated and quoted occurrences alike and carries every match through the last token. Policies that depend on chronology, the speaker or the entity need evidence scoped to those concepts. Over-limit input returns an error before inference.

## Model size and latency

The support policy has seven phrase predicates and seven reachable output vectors. The compiler numbers the output vectors with three-bit codes. Four codes fit in an exactly representable signed FP16 integer in `[-2048, 2047]`. The host uses five raw evidence bits to address a row and two bits to select a lane of the returned word. A single 1×1 convolution selects that word from 32 resident coefficients. The codebook maps result codes to answer, category and action bits.

This is the smallest artifact measured so far for this input and output contract. It matches the compiled size of a control model with one input, one output and one affine operation on the same NCS1 compiler. That control computed `2x + 3` and its FP16 IR weights occupied four bytes. A worldwide claim requires a comparable benchmark. Published [fastText.zip classification benchmarks](https://arxiv.org/abs/1612.03651) measure accuracy on other tasks and report a different storage boundary.

| Same policy | ONNX bytes | FP16 IR weight bytes | NCS1 blob bytes | VPU calls | Full-input p50 | Full-input p95 |
|---|---:|---:|---:|---:|---:|---:|
| **Packed** | **314** | **66** | **1,536** | **1** | **32.8–33.0 ms** | **38.4–38.8 ms** |
| Lookup | 700 | 258 | 1,728 | 1 | 33.0–36.1 ms | 37.5–42.9 ms |
| Gates | 563 | 66 | 3,776 | 1 | 32.2–34.3 ms | 36.7–41.8 ms |
| Dense | 1,832 | 512 | 4,480 | 1 | 32.8–33.4 ms | 36.9–38.4 ms |
| VPU phrase scan | 4,520 | 1,548 | 8,128 | 21 | 198.8–211.1 ms | 204.8–219.6 ms |

Ranges cover three separately loaded NCS1 sessions with 35 warm runs per input length in each session. The host had an i7-12650H and used OpenVINO 2020.3.2 with a Myriad 2 MA2450 USB stick. Each session checked all 128 Boolean assignments, 13 targeted fixtures and seven full-window documents. Tests cover evidence at the start, middle and end, evidence thousands of tokens apart, a late veto, a late category change, absent evidence and rejection at exactly 131,073 tokens. [Raw measurements](measurements/)

For the packed model, the VPU request and output decoding took 2.2–2.3 ms at p50. Counting the 397,562-byte input took 27.3–27.6 ms at p50. Model load took 1.62–1.82 s. The CPU-only implementation of the same policy took 30.5–31.7 ms at p50. The VPU scan variant also performs phrase matching on the stick. Its host counts Needle tokens, maps lexemes and passes the accumulated evidence bits between tiles.

The [1,536-byte blob](bundles/packed/ncs1.blob), 1,755-byte manifest and 45,492-byte losslessly compressed tokenizer total **48,783 bytes** for the NCS1 import path. Host code, Python, OpenVINO and firmware are additional. The tokenizer is decompressed and checked once per resident process. For NCS2, use the common IR and OpenVINO 2022.3.1 to compile a device-specific blob. All five variants also passed the 128 assignments on the [2022.3.1 CPU plugin](measurements/openvino-2022.3-cpu.json). Physical Myriad X measurements are pending. [Intel runtime support](https://www.intel.com/content/www/us/en/support/articles/000091265/software.html)

## Run on Windows

Set up the chosen stick using `tools/ncs1/setup_env.ps1` or `tools/ncs2/setup_env.ps1`. Install SentencePiece in its isolated environment. The runner selects a visible MYRIAD device. Use `--target` when multiple devices appear on the selected lane.

```powershell
$b = 'labs/shave_models/boolean'
$ncs1 = 'tools/ncs1/.venv/Scripts/python.exe'
$ncs2 = 'tools/ncs2/.venv/Scripts/python.exe'
& $ncs1 -m pip install sentencepiece==0.1.99
& $ncs1 "$b/runtime.py" "$b/bundles/packed" --lane ncs1 --blob "$b/bundles/packed/ncs1.blob" --text 'package missing. review authorized.' --explain

& $ncs2 -m pip install sentencepiece==0.1.99
& $ncs2 "$b/runtime.py" "$b/bundles/packed" --lane ncs2 --text 'package missing. review authorized.' --explain
```

Use `--file` for UTF-8 files. Use `--serve` to keep the model loaded and send one JSON object per line, each containing `text` and optionally `id`. The response includes that `id`. For the example above, `result` is `{"answers":{"may_route":true},"category":"delivery","action":"review_delivery"}`. The `--explain` flag adds literal matches and named rule values.

### Rebuild and benchmark

Use a current host Python with `numpy` and `onnx` to compile the rules. Packed and lookup forms support up to ten evidence predicates. Use `gates` for a larger policy or `scan` for on-stick evidence matching. The runner creates an FP16 IR from ONNX if the bundle lacks one. If your NCS1 toolchain lives outside this checkout, pass its directory with `--toolchain`.

```powershell
python "$b/build.py" "$b/support.json" "$b/bundles/my-policy" --variant packed
$corpus = Join-Path (Get-Location) 'decision-corpus.txt'
Invoke-WebRequest 'https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt' -OutFile $corpus
& $ncs1 "$b/benchmark.py" "$b/bundles/packed" --lane ncs1 --corpus $corpus --output ncs1-packed.json --export ncs1-packed.blob --fast-count --repeats 35
& $ncs2 "$b/benchmark.py" "$b/bundles/packed" --lane ncs2 --corpus $corpus --output ncs2-packed.json --export ncs2-packed.blob --fast-count --repeats 35
```

The benchmark verifies the SHA-256 of its pinned copy of Jane Austen’s public-domain *Pride and Prejudice*. Token lengths use the Apache 2.0 [Cactus Compute Needle 3 SentencePiece model](https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/tokenizer/tokenizer.model), SHA-256 `97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25`. The [license text](../bundles/needle-ui/LICENSE) also applies to this unmodified tokenizer. Tests compare optimized and original counts on Unicode, special tokens, newlines, repeated spaces and the full token limit.

Run the differential CPU tests with `python -m pytest labs/shave_models/boolean/test_boolean.py` in a host environment with `onnx`, `onnxruntime` and `sentencepiece`.

## Learned Boolean rules

[`learn.py`](learn.py) converts trained decision-tree paths into Boolean strings and gates proposals using a separate calibration set. On a held-out [SLURP](https://aclanthology.org/2020.emnlp-main.588/) replay, NCS1 matched the compiled policy on all 2,967 utterances. It classified 1,346 of 2,967 correctly (45.4%) and proposed 216 actions (7.3% coverage). Of those, 203 were correct (94.0% precision). When an utterance was embedded in 131,072 tokens, the learned policy preserved its clean decisions in one of 12 probes. The learned bundle currently serves as an utterance experiment. [Training selection](selection.json) · [Stick replay](measurements/learned-ncs1.json)

To reproduce, put SLURP's `train.jsonl`, `devel.jsonl` and `test.jsonl` in the same directory and pass it as `--cache` to `learn.py`. The trainer uses `scikit-learn` and writes a deduplicated test replay to `boolean-learned-test.jsonl` in that directory. Pass it as `--samples` to `evaluate_learned.py` in the stick's environment, along with the learned bundle, `--corpus` and `--output`. [Pinned SLURP annotations and CC BY 4.0 attribution](https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1)
