# MagicEightBall

MagicEightBall is an Intel Neural Compute Stick decision model. It accepts one UTF-8 document of up to **131,072 Needle 3 tokens** and returns a yes/no answer, a category and an optional proposed action. It runs the decision lookup on the VPU in one request. The host counts tokens, extracts literal evidence and decodes the result. The proposed action is returned as data. The model identifier is `magiceightball`.

[NCS1 release guide](RELEASE-NCS1.md) · [Frontline starter packs](industries/frontline/README.md) · [Hugging Face model family](https://huggingface.co/tablinum/MagicEightBall)

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

The portable CPU exports provide a 192-byte ONNX radix Gather, a 1,832-byte FP32 dense reference, and a 1,024-byte GGUF profile with a 128-byte decision table. All three use the pinned Needle tokenizer and the support policy. The GGUF profile uses the included `magiceightball` reader. Run `export_portable.py <output> --corpus <pinned-corpus>` with NumPy, SentencePiece, ONNX Runtime, and the `gguf` package to build and verify the three bundles. Run `portable_runtime.py <artifact> --text <document>` from a bundle. These exports use the CPU. The NCS1 release uses its native archive. [CPU command examples and matched benchmark](portable_comparison/README.md)

The support policy has seven phrase predicates and seven reachable output vectors. The compiler numbers the output vectors with three-bit codes. Four codes fit in an exactly representable signed FP16 integer in `[-2048, 2047]`. The host uses five raw evidence bits to address a row and two bits to decode the selected word. In the smallest build, one VPU `Gather` reads that word from 32 resident coefficients using one INT32 index. The codebook maps result codes to the Boolean answer, category and action.

The **480-byte radix Gather** is the smallest artifact physically verified here for this input and output contract. The stock compiler emits the radix graph at **512 bytes**. Removing 32 bytes of terminal zero padding leaves the section starts, executable stage and all policy data intact. NCS1 passed all 128 assignments, both full-context cases and overflow rejection with the compact artifact. It took 5.31 ms median for one 128-document batch, compared with 5.37 ms for the padded radix control. [Physical size and speed comparison](measurements/compact-gather-physical-ncs1.json) · [Competitors](COMPETITORS.md)

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

The 128-slot [release archive](bundles/magiceightball.m8b) and [compact tokenizer](needle3-tokenizer.m8t.xz) total **24,147 bytes**. The 735-byte release contains the policy manifest and both native 2020.3 executable images. The 23,412-byte tokenizer is shared between targets. Host code, Python, OpenVINO and firmware are additional.

The tokenizer archive groups vocabulary strings by byte length. It records a leading SentencePiece space marker in one bit per vocabulary entry and regenerates byte fallback entries and scores. The reader restores rank order, verifies all 126,520 original bytes by SHA-256 and passes them to SentencePiece. The [rebuild tool](compact_tokenizer.py) accepts the pinned raw `.model`, the previous 26,136-byte archive and earlier archives.

Across 60 alternating initializations per archive in one Python 3.7 process, median startup through adaptive vocabulary initialization was **14.0 ms** with the length-bucket archive and **13.9 ms** with the previous archive. Median startup through SentencePiece initialization alone was **7.1 ms** and **6.9 ms**. An earlier physical NCS1 replay with the prior compact format passed the full truth table and mixed-length inputs, with one 128-decision request taking 5.4 ms median. [Length-bucket startup measurements](measurements/tokenizer-length-buckets.json) · [Earlier physical device replay](measurements/compact-batch-ncs1.json)

### Native release format

The release restores the manifest and both 512-byte native graphs byte for byte. The runtime checks the graph layout and selects the requested physical target through the 2020.3 plugin. The current release and tokenizer passed physical replay on NCS1, including all 128 Boolean assignments and the mixed 131,072-token replay. NCS2 uses the common IR in the standard 2022.3 lane and has software verification. [Current release replay](measurements/release-vector-ncs1.json)

```powershell
python "$b/release.py" "$b/bundles/magiceightball-batch" "$b/bundles/magiceightball.m8b"
& $ncs1 "$b/runtime.py" "$b/bundles/magiceightball.m8b" --release-stick ncs1 --text 'package missing. review authorized.'
```

On the 2020.3 runtime with an NCS2, use `--release-stick ncs2`. The hardware benchmark can select the corresponding release image with `--expect-stick ncs2` when run inside that runtime.

The [128-slot NCS1 blob](bundles/magiceightball-batch/ncs1-native.blob) also occupies 512 bytes. Both builds use a single rank-one Gather over the same 64-byte table. The 2020.3.2 serializer aligns each section to 64 bytes. These blobs contain 192 bytes of headers, 64 each for input and output descriptors, 128 for the Gather stage and 64 for the table. The compiler emits the same byte count for a two-entry bank. This establishes the size floor for this resident rank-one Gather layout under that compiler. [Intel serializer source](https://github.com/openvinotoolkit/openvino/blob/2020.3.2/inference-engine/src/vpu/graph_transformer/src/backend/serialize.cpp#L150-L175) · [Intel Gather stage source](https://github.com/openvinotoolkit/openvino/blob/2020.3.2/inference-engine/src/vpu/graph_transformer/src/stages/gather.cpp#L67-L82) · [Parsed artifact layouts](measurements/batch-head-compatibility.json)

In a matched release comparison, NCS1 handled 128 short decisions in one request in **6.43 ms median**, down from **223.70 ms** when sent individually. A CPU lookup took **3.82 ms** for the same batch. Fresh 131,072-token requests were dominated by host tokenization and evidence extraction. [NCS1 release and matched CPU results](RELEASE-NCS1.md)

ReLU, Clip and Sigmoid controls compiled to **448 bytes** on both targets with 128-element FP16 vectors. These controls have no resident constants section and measure container overhead for simpler operations. The compact radix graph uses the same 448 bytes before its complete 32-byte resident policy table. [Release and stock-stage controls](measurements/native-release-and-stock-controls.json)

An exact rank-zero Gather with a scalar index and output matched all 16 packed words on the CPU plugin. The 2020.3.2 offline compiler exited with a Windows access violation on both MYRIAD targets. The rank-one vector control rebuilt byte for byte at 512 bytes on each target using the same offline configuration. [Rank-zero experiment](measurements/native-release-and-stock-controls.json) · [Intel descriptor serializer](https://github.com/openvinotoolkit/openvino/blob/2020.3.2/inference-engine/src/vpu/graph_transformer/src/model/data.cpp#L447-L484)

MaxPool and ReduceMax also reproduced all 128 assignments on the CPU plugin by evaluating 16 distinct Boolean conjunctions in parallel. The host supplied individual signed facts and decoded which conjunctions matched each rule. These graphs compiled without resident constants. The smallest result was 512 bytes on NCS1 and 832 bytes on NCS2. Both used 24,576 bytes of input and output tensors per full batch versus 768 for Gather. [Pooling experiment and layouts](measurements/constant-free-pooling.json) · [Intel MaxPool serializer](https://github.com/openvinotoolkit/openvino/blob/2020.3.2/inference-engine/src/vpu/graph_transformer/src/stages/pooling.cpp)

The [512-byte Myriad X target blob](bundles/magiceightball/ncs2-2020.3.blob) was compiled offline by Intel's 2020.3.2 `myriad_compile` for `VPU_MYRIAD_2480`. Blob format is toolchain-specific. OpenVINO 2022.3.1 rejects the 2020.3 binary version, so use the common IR on the NCS2 lane. The Gather IR passed all 128 combinations on the [2022.3.1 CPU plugin](measurements/gather-2022.3-cpu.json). Physical Myriad X correctness and speed are still pending. [Intel runtime support](https://www.intel.com/content/www/us/en/support/articles/000091265/software.html)

The [128-slot Myriad X blob](bundles/magiceightball-batch/ncs2-2020.3.blob) also compiled offline to 512 bytes. OpenVINO 2022.3.1 accepted all 128 decisions in one vector with the common IR on its CPU plugin. [Batch compatibility record](measurements/batch-head-compatibility.json)

### Radix packing

The optional [radix bundle](bundles/magiceightball-radix/model.json) stores the same decisions in **32 bytes of constants**, down from 64. Its compiler groups outcomes by the four category facts and uses the three authorization and veto facts to select among eight lanes. Each row contains no more than two possible results. The stick returns both result codes and the selector bits in one exact FP16 word. All row contents reside in the graph. Host decoding permutes the raw fact bits, selects a slot in the returned word and resolves the output labels.

The stock blob remains 512 bytes because the compiler reserves a whole 64-byte constants section. On NCS1, the compact radix variant preserves the 32 live bytes of that section and stops at 480. For an eight-fact control with three reachable decisions, the same codec used 52 constant bytes and reduced the compiled size from **576 to 512 bytes** on each target. The 2022.3.1 CPU plugin passed all 128 MagicEightBall assignments and all 256 control assignments. It also passed the 131,072-token mixed-length replay. [Radix compatibility and size evidence](measurements/radix-gather-compatibility.json) · [Physical radix results](measurements/compact-gather-physical-ncs1.json)

Rebuild this variant with `--variant gather --codec radix --slots 128`. The default batch bundle has the physical NCS1 benchmark record.

### Two policies in one graph

The [banked bundle](bundles/magiceightball-banks/model.json) holds customer support and on-call triage in a single Gather graph. Each policy contributes a distinct 32-byte table. The combined graph compiled to **512 bytes** for each target. The caller names a bank and the runtime adds that bank's fixed offset to each address. A single request can carry up to 128 documents across both banks. The compiler shares storage for byte-identical tables. [Both compiled layouts and CPU verification](measurements/two-bank-compatibility.json)

The physical NCS1 replay passed all 256 decisions, three mixed 131,072-token inputs and overflow rejection. Across 13 warmed trials, one mixed request for 128 short documents took 6.8 ms median. Separate requests for the same two banks took 8.2 ms median. The on-call policy requires both capacity signals before proposing a page for a capacity-only incident. Each policy ignores distractor terms from the other policy. [Physical two-bank replay](measurements/two-banks-ncs1.json)

```powershell
python "$b/build.py" "$b/policy-banks.json" "$b/bundles/magiceightball-banks" --variant banks --slots 128
'{"id":"mixed","documents":[{"id":"s1","bank":"support","text":"duplicate charge; review authorized"},{"id":"o1","bank":"oncall","text":"queue backed up; paging approved"}]}' |
  & $ncs1 "$b/runtime.py" "$b/bundles/magiceightball-banks" --blob "$b/bundles/magiceightball-banks/ncs1-native.blob" --serve
```

The top-level `device_requests` counts native submissions. Each document gets its own result or error in input order. The optional [adaptive configuration](adaptive-banks.json) routes each vertical to a named bank and shares one resident graph across them. The mixed JSONL request above uses the direct runtime.

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

### Append to a live document

Use `--stream` when new text arrives over time. Each JSONL request appends its string verbatim and returns the updated yes/no answer, category and proposed action. Use `reset: true` to start a document. The worker holds one document per process and uses its selected policy bank for every fragment. Rejected requests leave the last accepted document intact.

```powershell
@(
  '{"id":"start","reset":true,"append":"duplicate charge"}'
  '{"id":"approve","append":" review authorized"}'
  '{"id":"veto","append":" do not route"}'
) | & $ncs1 "$b/runtime.py" "$b/bundles/magiceightball.m8b" --release-stick ncs1 --stream
```

The host reuses the exact token count and literal facts for the prefix ending at a stable ASCII-space boundary. It keeps the last unfinished segment and enough preceding lexemes to match phrases across appends. Very long spans without ASCII spaces can require recounting that span. `--explain` retains the full text to include rules that are used only in explanations.

In the current NCS1 release replay, the first 130,432 tokens took 51.8 ms. Across 24 short appends near the limit, median total time fell from **52.1 ms** for complete-document requests to **1.9 ms** for incremental requests. The final veto cleared the proposed action at exactly 131,072 tokens. An append that would reach 131,073 tokens was rejected. The earlier CPU control measured host preparation falling from **28.9 ms** to **0.056 ms**. [Physical NCS1 append replay](measurements/release-stream-ncs1.json) · [CPU append control](measurements/stream-context-cpu.json)

## Adaptive batches

[`adaptive.py`](adaptive.py) maintains a latency estimate for each combination of device lane, vertical, policy, document structure and length range. It compares three exact host preparation paths. They use an independent literal substring search with lexical boundary checks, regular expressions with uncached token counting, or regular expressions with a bounded cache of exact token counts. The independent matcher falls back to full lexical projection when a document has many hits for an anchor.

The router checks the first batch in each profile and repeats the check every eight batches. It compares alternate token and evidence paths on an excerpt, then checks the chosen lane's full-document evidence and VPU outputs against the references. A disagreeing or failing lane is quarantined for 16 batches before a fresh check. A device exception triggers one reload and retry. Set `--batch-timeout` to supervise the native worker, return a batch error on a stall and start a new worker for the next batch. If firmware stalls during shutdown, the supervisor bounds its cleanup wait. SQLite retains optional profile timings, health and aggregate scores from labeled inputs. Document text and token-cache entries remain process-local.

Each vertical can point to a different compiled policy. With `evidence_field`, the runner checks the token limit on the whole JSON document and evaluates the Boolean evidence only in the named top-level string. The [sample config](adaptive-support.json) uses full-text evidence for support notes and the `evidence` field for structured cases.

With the 128-slot bundle, the runner sends up to 128 valid documents from one batch in a single Gather request. It preserves response order, skips over-limit or invalid documents, pads unused slots and discards padded results. The response's `device_requests` counts submissions including any retry. Each document's `request_documents` reports how many documents shared its native request.

```powershell
'{"id":"batch-1","vertical":"case_records","documents":[{"id":"case-9","text":"{\"evidence\":\"package missing. review authorized.\",\"archive\":\"review cancelled\"}"}]}' |
  & $ncs1 "$b/adaptive.py" "$b/adaptive-support.json" --state routing.db --batch-timeout 30 --explain
```

For a labeled replay, optionally supply an `expected` object on each document with the canonical `answers`, `category` and `action` fields. The batch reports agreement for each output, joint agreement, and proposals issued and correct. Labels measure policy quality. The contextual bandit uses only preparation timings and exactness checks to tune execution. The deployed Boolean rules change when a newly compiled policy is selected.

### Measured batch routing on NCS1

On the 128-slot graph, 128 short documents spanning the entire truth table took **5.4 ms median** in one request versus **221.2 ms** when the same graph processed them individually. These timings include text processing and result decoding. The batched p95 was 6.2 ms across 13 warmed trials. A mixed-length batch also returned correct results for three 131,072-token documents and an empty document in one request, while rejecting a 131,073-token document. [Native vector benchmark](measurements/batch-head-ncs1.json)

Ten two-document batches per profile used overlapping excerpts of the same source corpus, with distinct documents of exactly 131,072 tokens each. Both routes used the same evidence scope per profile and passed all 40 decisions. The adaptive runner used its cache for 19 of 20 documents in each profile and cut USB requests from 20 to 10 per profile. Values below are per batch and include periodic checks. [Raw vector measurements](measurements/adaptive-ncs1-vector-gather.json)

| Profile | Fixed p50 | Adaptive p50 | Fixed p95 | Adaptive p95 |
|---|---:|---:|---:|---:|
| Full-text support notes | 83.3 ms | 50.5 ms | 89.0 ms | 84.2 ms |
| JSON records with a declared evidence field | 52.5 ms | 29.1 ms | 60.5 ms | 46.4 ms |

The dictionary cache benefits workloads with repeated vocabulary. The earlier implementation used a full lexical projection during each prose audit and measured 114.9 ms at p95. [Earlier matched-scope run](measurements/adaptive-ncs1-matched-scope.json)

### Rebuild and benchmark

Use a current host Python with `numpy` and `onnx` to compile the rules. Gather, packed and lookup forms support up to ten evidence predicates. Use `gates` for a larger policy or `scan` for on-stick evidence matching. The runner creates an FP16 IR from ONNX if the bundle lacks one. If your NCS1 toolchain lives outside this checkout, pass its directory with `--toolchain`.

```powershell
python "$b/build.py" "$b/support.json" "$b/bundles/my-policy" --variant gather --slots 128
$corpus = Join-Path (Get-Location) 'decision-corpus.txt'
Invoke-WebRequest 'https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt' -OutFile $corpus
& $ncs1 "$b/benchmark.py" "$b/bundles/magiceightball" --lane ncs1 --corpus $corpus --output ncs1-gather.json --export ncs1-gather.blob --fast-count --repeats 35
& $ncs2 "$b/benchmark.py" "$b/bundles/magiceightball" --lane ncs2 --corpus $corpus --output ncs2-gather.json --export ncs2-gather.blob --fast-count --repeats 35
& $ncs1 "$b/benchmark_adaptive.py" "$b/bundles/magiceightball-batch" --corpus $corpus --output adaptive.json
& $ncs1 "$b/benchmark_batch_heads.py" "$b/bundles/magiceightball-batch" --corpus $corpus --output gather-vector.json
& $ncs1 "$b/benchmark_stream.py" "$b/bundles/magiceightball.m8b" --expect-stick ncs1 --corpus $corpus --output stream-ncs1.json
& $ncs2 "$b/benchmark_stream.py" "$b/bundles/magiceightball-batch" --cpu --corpus $corpus --output stream-cpu.json
& $ncs1 "$b/benchmark_policy_banks.py" "$b/bundles/magiceightball-banks" --blob "$b/bundles/magiceightball-banks/ncs1-native.blob" --expect-stick ncs1 --corpus $corpus --output gather-banks.json
```

On a Windows host with one freshly connected NCS2, this acceptance run identifies the stick from its cold USB product ID before loading the real MYRIAD plugin. It runs the 128-case truth table, measures warmed batches, verifies mixed inputs at the 131,072-token boundary and exports the graph from 2022.3.1.

```powershell
& $ncs2 "$b/benchmark_batch_heads.py" "$b/bundles/magiceightball-batch" --lane ncs2 --expect-stick ncs2 --corpus $corpus --output ncs2-verified.json --export ncs2-verified.blob
```

The benchmark verifies the SHA-256 of its pinned copy of Jane Austen’s public-domain *Pride and Prejudice*. Token lengths use the Apache 2.0 [Cactus Compute Needle 3 SentencePiece model](https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/tokenizer/tokenizer.model), SHA-256 `97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25`. The [license text](../bundles/needle-ui/LICENSE) also applies to this unmodified tokenizer. Tests compare optimized and original counts on Unicode, special tokens, newlines, repeated spaces and the full token limit.

Run the differential CPU tests with `python -m pytest labs/shave_models/boolean/test_boolean.py` in a host environment with `onnx`, `onnxruntime` and `sentencepiece`.

## Learned Boolean rules

[`learn.py`](learn.py) converts trained decision-tree paths into Boolean strings and gates proposals using a separate calibration set. On a held-out [SLURP](https://aclanthology.org/2020.emnlp-main.588/) replay, NCS1 matched the compiled policy on all 2,967 utterances. It classified 1,346 of 2,967 correctly (45.4%) and proposed 216 actions (7.3% coverage). Of those, 203 were correct (94.0% precision). When an utterance was embedded in 131,072 tokens, the learned policy preserved its clean decisions in one of 12 probes. The learned bundle currently serves as an utterance experiment. [Training selection](selection.json) · [Stick replay](measurements/learned-ncs1.json)

Placing those utterances in a declared JSON evidence field preserved their clean decisions in all 12 full-window probes. NCS1 classified 9 of 12 by the original labels and all six proposed actions were correct. This measures field isolation when the caller supplies the evidence location. [Scoped stick replay](measurements/learned-ncs1-scoped.json)

To reproduce, put SLURP's `train.jsonl`, `devel.jsonl` and `test.jsonl` in the same directory and pass it as `--cache` to `learn.py`. The trainer uses `scikit-learn` and writes a deduplicated test replay to `boolean-learned-test.jsonl` in that directory. Pass it as `--samples` to `evaluate_learned.py` in the stick's environment, along with the learned bundle, `--corpus` and `--output`. Add `--scoped` for the JSON evidence-field replay. [Pinned SLURP annotations and CC BY 4.0 attribution](https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1)
