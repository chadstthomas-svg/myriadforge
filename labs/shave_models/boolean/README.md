# MagicEightBall

MagicEightBall is an Intel Neural Compute Stick decision model. It accepts one UTF-8 document of up to **131,072 Needle 3 tokens** and returns a yes/no answer, a category and an optional proposed action. It runs the decision lookup on the VPU in one request. The host counts tokens, extracts literal evidence and decodes the result. The proposed action is returned as data. The model identifier is `magiceightball`.

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

The support policy has seven phrase predicates and seven reachable output vectors. The compiler numbers the output vectors with three-bit codes. Four codes fit in an exactly representable signed FP16 integer in `[-2048, 2047]`. The host uses five raw evidence bits to address a row and two bits to decode the selected word. In the smallest build, one VPU `Gather` reads that word from 32 resident coefficients using one INT32 index. The codebook maps result codes to the Boolean answer, category and action.

The **512-byte Gather** is the smallest artifact measured here for this input and output contract. It matches the offline compiler's size for a two-entry scalar Gather on both device targets. A worldwide claim requires a comparable benchmark. Published fastText.zip classification benchmarks measure accuracy on other tasks and report a different storage boundary. [Offline compiler controls](measurements/offline-compiler-controls.json) · [fastText.zip benchmarks](https://arxiv.org/abs/1612.03651)

| Same policy | ONNX bytes | FP16 IR weight bytes | NCS1 blob bytes | VPU calls | Full-input p50 | Full-input p95 |
|---|---:|---:|---:|---:|---:|---:|
| **Gather, native FP16 output** | **256** | **72** | **512** | **1** | **53.2 ms** | **62.4 ms** |
| Gather, cast to FP32 output | 256 | 72 | 640 | 1 | 54.0 ms | 62.9 ms |
| Packed convolution | 314 | 66 | 1,536 | 1 | 32.8–33.0 ms | 38.4–38.8 ms |
| Lookup | 700 | 258 | 1,728 | 1 | 33.0–36.1 ms | 37.5–42.9 ms |
| Gates | 563 | 66 | 3,776 | 1 | 32.2–34.3 ms | 36.7–41.8 ms |
| Dense | 1,832 | 512 | 4,480 | 1 | 32.8–33.4 ms | 36.9–38.4 ms |
| VPU phrase scan | 4,520 | 1,548 | 8,128 | 21 | 198.8–211.1 ms | 204.8–219.6 ms |

Each Gather row covers one NCS1 session. Ranges for the earlier variants cover three sessions. Each session used 35 warm runs per input length on an i7-12650H and a Myriad 2 MA2450 stick with OpenVINO 2020.3.2. Every session checked all 128 Boolean assignments, 13 targeted fixtures and seven full-window documents. Tests cover evidence at the start, middle and end, evidence thousands of tokens apart, a late veto, a late category change, absent evidence and rejection at exactly 131,073 tokens. [Raw measurements](measurements/)

For the packed model, the VPU request and output decoding took 2.2–2.3 ms at p50. Counting the 397,562-byte input took 27.3–27.6 ms at p50. Model load took 1.62–1.82 s. The CPU-only implementation of the same policy took 30.5–31.7 ms at p50. The VPU scan variant also performs phrase matching on the stick. Its host counts Needle tokens, maps lexemes and passes the accumulated evidence bits between tiles.

The [512-byte NCS1 blob](bundles/magiceightball/ncs1-native.blob), 1,755-byte Windows manifest and 45,492-byte losslessly compressed tokenizer total **47,759 bytes**. Host code, Python, OpenVINO and firmware are additional. The tokenizer is decompressed and checked once per resident process.

The [512-byte Myriad X target blob](bundles/magiceightball/ncs2-2020.3.blob) was compiled offline by Intel's 2020.3.2 `myriad_compile` for `VPU_MYRIAD_2480`. Blob format is toolchain-specific. OpenVINO 2022.3.1 rejects the 2020.3 binary version, so use the common IR on the NCS2 lane. The Gather IR passed all 128 combinations on the [2022.3.1 CPU plugin](measurements/gather-2022.3-cpu.json). Physical Myriad X correctness and speed are still pending. [Intel runtime support](https://www.intel.com/content/www/us/en/support/articles/000091265/software.html)

## Run on Windows

Set up the chosen stick using `tools/ncs1/setup_env.ps1` or `tools/ncs2/setup_env.ps1`. Install SentencePiece in its isolated environment. The runner selects a visible MYRIAD device. Use `--target` when multiple devices appear on the selected lane.

```powershell
$b = 'labs/shave_models/boolean'
$ncs1 = 'tools/ncs1/.venv/Scripts/python.exe'
$ncs2 = 'tools/ncs2/.venv/Scripts/python.exe'
& $ncs1 -m pip install sentencepiece==0.1.99
& $ncs1 "$b/runtime.py" "$b/bundles/magiceightball" --lane ncs1 --blob "$b/bundles/magiceightball/ncs1-native.blob" --text 'package missing. review authorized.' --explain

& $ncs2 -m pip install sentencepiece==0.1.99
& $ncs2 "$b/runtime.py" "$b/bundles/magiceightball" --lane ncs2 --text 'package missing. review authorized.' --explain
```

Use `--file` for UTF-8 files. Use `--serve` to keep the model loaded and send one JSON object per line, each containing `text` and optionally `id`. The response includes that `id`. For the example above, `result` is `{"answers":{"may_route":true},"category":"delivery","action":"review_delivery"}`. The `--explain` flag adds literal matches and named rule values.

## Adaptive batches

[`adaptive.py`](adaptive.py) maintains a latency estimate for each combination of device lane, vertical, policy, document structure and length range. It compares three exact host preparation paths. They use projected lexemes, regular expressions with uncached token counting, or regular expressions with a bounded cache of exact token counts. Each document still submits its decision to the physical stick.

The router checks the first batch in each profile and repeats the check every eight batches. It compares alternate token and evidence paths on an excerpt, then checks the chosen lane's full-document evidence and VPU outputs against the references. A disagreeing or failing lane is quarantined for 16 batches before a fresh check. A device exception triggers one reload and retry. Set `--batch-timeout` to supervise the native worker, return a batch error on a stall and start a new worker for the next batch. If firmware stalls during shutdown, the supervisor bounds its cleanup wait. SQLite retains optional profile timings, health and aggregate scores from labeled inputs. Document text and token-cache entries remain process-local.

Each vertical can point to a different compiled policy. With `evidence_field`, the runner checks the token limit on the whole JSON document and evaluates the Boolean evidence only in the named top-level string. The [sample config](adaptive-support.json) uses full-text evidence for support notes and the `evidence` field for structured cases.

```powershell
'{"id":"batch-1","vertical":"case_records","documents":[{"id":"case-9","text":"{\"evidence\":\"package missing. review authorized.\",\"archive\":\"review cancelled\"}"}]}' |
  & $ncs1 "$b/adaptive.py" "$b/adaptive-support.json" --state routing.db --batch-timeout 30 --explain
```

For a labeled replay, optionally supply an `expected` object on each document with the canonical `answers`, `category` and `action` fields. The batch reports agreement for each output, joint agreement, and proposals issued and correct. Labels measure policy quality. The contextual bandit uses only preparation timings and exactness checks to tune execution. The deployed Boolean rules change when a newly compiled policy is selected.

### Measured batch routing on NCS1

Ten two-document batches per profile used overlapping excerpts of the same source corpus, with distinct documents of exactly 131,072 tokens each. Both routes used the same evidence scope per profile and passed all 40 decisions. The adaptive runner used its cache for 19 of 20 documents in each profile. Values below are per batch and include periodic checks. [Raw matched-scope measurements](measurements/adaptive-ncs1-matched-scope.json)

| Profile | Fixed p50 | Adaptive p50 | Fixed p95 | Adaptive p95 |
|---|---:|---:|---:|---:|
| Full-text support notes | 76.3 ms | 52.0 ms | 91.8 ms | 114.9 ms |
| JSON records with a declared evidence field | 49.8 ms | 29.9 ms | 60.9 ms | 45.5 ms |

Full-document evidence checks added latency to the tail for prose. The dictionary cache benefits workloads with repeated vocabulary.

### Rebuild and benchmark

Use a current host Python with `numpy` and `onnx` to compile the rules. Gather, packed and lookup forms support up to ten evidence predicates. Use `gates` for a larger policy or `scan` for on-stick evidence matching. The runner creates an FP16 IR from ONNX if the bundle lacks one. If your NCS1 toolchain lives outside this checkout, pass its directory with `--toolchain`.

```powershell
python "$b/build.py" "$b/support.json" "$b/bundles/my-policy" --variant gather
$corpus = Join-Path (Get-Location) 'decision-corpus.txt'
Invoke-WebRequest 'https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt' -OutFile $corpus
& $ncs1 "$b/benchmark.py" "$b/bundles/magiceightball" --lane ncs1 --corpus $corpus --output ncs1-gather.json --export ncs1-gather.blob --fast-count --repeats 35
& $ncs2 "$b/benchmark.py" "$b/bundles/magiceightball" --lane ncs2 --corpus $corpus --output ncs2-gather.json --export ncs2-gather.blob --fast-count --repeats 35
& $ncs1 "$b/benchmark_adaptive.py" "$b/bundles/magiceightball" --corpus $corpus --output adaptive.json
```

The benchmark verifies the SHA-256 of its pinned copy of Jane Austen’s public-domain *Pride and Prejudice*. Token lengths use the Apache 2.0 [Cactus Compute Needle 3 SentencePiece model](https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/tokenizer/tokenizer.model), SHA-256 `97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25`. The [license text](../bundles/needle-ui/LICENSE) also applies to this unmodified tokenizer. Tests compare optimized and original counts on Unicode, special tokens, newlines, repeated spaces and the full token limit.

Run the differential CPU tests with `python -m pytest labs/shave_models/boolean/test_boolean.py` in a host environment with `onnx`, `onnxruntime` and `sentencepiece`.

## Learned Boolean rules

[`learn.py`](learn.py) converts trained decision-tree paths into Boolean strings and gates proposals using a separate calibration set. On a held-out [SLURP](https://aclanthology.org/2020.emnlp-main.588/) replay, NCS1 matched the compiled policy on all 2,967 utterances. It classified 1,346 of 2,967 correctly (45.4%) and proposed 216 actions (7.3% coverage). Of those, 203 were correct (94.0% precision). When an utterance was embedded in 131,072 tokens, the learned policy preserved its clean decisions in one of 12 probes. The learned bundle currently serves as an utterance experiment. [Training selection](selection.json) · [Stick replay](measurements/learned-ncs1.json)

Placing those utterances in a declared JSON evidence field preserved their clean decisions in all 12 full-window probes. NCS1 classified 9 of 12 by the original labels and all six proposed actions were correct. This measures field isolation when the caller supplies the evidence location. [Scoped stick replay](measurements/learned-ncs1-scoped.json)

To reproduce, put SLURP's `train.jsonl`, `devel.jsonl` and `test.jsonl` in the same directory and pass it as `--cache` to `learn.py`. The trainer uses `scikit-learn` and writes a deduplicated test replay to `boolean-learned-test.jsonl` in that directory. Pass it as `--samples` to `evaluate_learned.py` in the stick's environment, along with the learned bundle, `--corpus` and `--output`. Add `--scoped` for the JSON evidence-field replay. [Pinned SLURP annotations and CC BY 4.0 attribution](https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1)
