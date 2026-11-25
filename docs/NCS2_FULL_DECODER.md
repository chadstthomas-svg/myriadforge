# Full Llama decoder on Intel NCS2

The full decoder runs every trained transformer block and the vocabulary projection on one physical Myriad X. The host handles tokenization, embedding lookup, greedy sampling, and FP16 KV storage. The static graph consumes one token and one cache window at a time. Its last attention slot holds the current token, and a mask exposes only the populated past rows and that slot. All K/V rows are committed together after the device returns finite results.

The original prefix-recomputation graph remains available as a diagnostic control in tiny_llama.py. The cached path uses static two-dimensional attention operations and compares them with an independent NumPy decoder. The Hugging Face checkpoint loader supports bias-free Llama with standard RoPE and SiLU, including grouped-query attention and tied or untied embeddings.

## Run and verify

Use the OpenVINO 2022.3.1 Python environment with SentencePiece for .model tokenizers or tokenizers for Hugging Face .json tokenizers. Place large checkpoint files outside the source tree. The TinyStories release has a 512-token vocabulary for 260K and the llama2.c 32K tokenizer for its 15M, 42M, and 110M checkpoints. [Pinned TinyStories files](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add) · [llama2.c tokenizer](https://github.com/karpathy/llama2.c/blob/master/tokenizer.model)

~~~powershell
$ncs2 = 'tools/ncs2/.venv/Scripts/python.exe'
$weights = 'models/tinyllamas/stories260K.bin'
$tokenizer = 'models/tinyllamas/tok512.model'
& $ncs2 tools/ncs2/decoder_validation.py $weights $tokenizer --out results/decoder-128.json
& $ncs2 tools/ncs2/decoder_tune.py $weights $tokenizer results/decoder-128.json --out-dir results/decoder-tune
& $ncs2 tools/ncs2/decoder_validation.py $weights $tokenizer --mode generate --profile results/decoder-tune/profile.json --out results/generated.json
~~~

The model and tokenizer must match the profile. The profile also verifies OpenVINO version, device family, context, graph ranges, and the cached decoder source fingerprint. Set the context to 256 or 512 to qualify a new window within the checkpoint’s trained limit before tuning it. Use `--graphs 2` to try two contiguous graphs. Both stay loaded while the host relays the hidden vector between them.

To prepare pinned Hugging Face weights, download the checkpoint, tokenizer and configuration at a fixed revision. Run the following conversion with the modern host Python and use the resulting directory as the checkpoint argument in the legacy runner. Conversion reads safetensors or uses PyTorch’s weights-only loader for the original Llama 68M archive.

~~~powershell
python tools/ncs2/decoder_checkpoint.py models/source models/prepared --revision MODEL_COMMIT
~~~

## How a model qualifies

- Twelve fixed prompts cover 8-, 64-, and 112-token prefixes at the 128-token window. Larger windows test a short prefix, the midpoint, and the final 16 token positions. The device must match at least 99% of 192 teacher-forced CPU top-1 decisions. Any disagreement where the CPU’s top-two margin is at least 0.10 fails the candidate.
- The harness checks repeat generation, reset, the configured context boundary, and finite device logits and cache updates. Teacher-forced performance includes host embedding lookup, KV movement, and greedy choice. Time to first token includes prompt prefill.
- The tuner checks CPU control prefixes before timing, ranks allocations by warm token p95, repeats the finalists and stock allocation in three independent loads, and reruns full qualification before a ten-minute soak. Two-session throughput is measured separately.

## Physical NCS2 results at 128 tokens

| Checkpoint | Trained parameters | CPU top-1 agreement | Disagreements at margin ≥ 0.10 | Qualification |
| --- | ---: | ---: | ---: | --- |
| TinyStories 260K | 260,032 | 191/192 | 0 | Passed |
| TinyStories 15M | 15,191,712 | 176/192 | 8 | Failed |
| TinyStories 42M | 41,689,600 | 181/192 | 4 | Failed |
| Llama 68M | 68,030,208 | 184/192 | 0 | Failed |
| TinyStories 110M | 109,529,856 | 186/192 | 2 | Failed |
| SmolLM2-135M | 134,515,008 | 0/192 | 180 | Failed |

TinyStories 110M loaded 208.91 MiB of FP16 weight equivalents in one graph and produced 5.82 device-path tokens per second. It missed the parity gate. The focused checks changed cache layout, normalization, attention graph, SHAVE/CMX allocation, and graph split. The tested alternatives did not resolve the mismatches. Llama 68M's eight errors were all below the margin cutoff, and it still fell short of the 99% accuracy gate.

SmolLM2-135M loaded and produced token 0 throughout the full 192-comparison test. Splitting it across two resident graphs reproduced the failure in a 16-comparison probe. This differs from the earlier dense-weight memory timeout. The model reached inference but failed correctness. The 360M stretch checkpoint was conditional on 135M qualification. [Evidence and hashes](../tools/ncs2/measurements/decoder-search-v1.json)

## TinyStories 260K tuning

All 21 allocations passed the screening comparison. The stock allocation and the three fastest candidates were then measured in three independent loads each.

| SHAVEs | CMX slices | Streams | Confirmed warm token p50 | Confirmed warm token p95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 30.44 ms | **31.11 ms** |
| 2 | 2 | 1 | 31.43 ms | 32.03 ms |
| 3 | 3 | 1 | 32.78 ms | 33.88 ms |
| 16 | 16 | 1 | 40.45 ms | 41.31 ms |

One SHAVE and one CMX slice cut confirmed p95 by 24.7%. The allocation matched 191/192 CPU choices in full qualification, with zero above the margin cutoff. The full qualification replay measured 32.92 warm tokens/s on the NCS2. The NumPy CPU reference was faster on this checkpoint, at 1.45 ms p95 per warm token versus 30.98 ms for the device path.

The selected allocation completed a 600-second soak with 258 repeated generations and 3,870 timed decode steps. It reproduced the saved sequences and matched all 48 CPU control comparisons used before the soak. Warm p95 was 31.39 ms. [Verified 128-token profile](../tools/ncs2/measurements/tinystories-260k-128-profile.json)

## Context and concurrency

| Context | CPU top-1 agreement | Clear disagreements | NCS2 warm tokens/s | NCS2 median first token | KV input and output per step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | **191/192** | 0 | 32.92 | 1.94 s | 80 KiB |
| 256 | 189/192 | 0 | 31.43 | 4.08 s | 160 KiB |
| 512 | 189/192 | 1 | 28.91 | 8.86 s | 320 KiB |

All three windows honored the context cap and cache reset. The profile qualifies at 128 tokens. The 256- and 512-token runs fell below the accuracy gate. The saved profile successfully generated a continuation through the command-line runner.

Two concurrent cached sessions preserved their serial control sequences at 4 SHAVEs and 8 CMX slices per stream. They produced 31.13 aggregate tokens/s in the separate throughput probe. Other tested concurrent allocations changed generated tokens. The verified single-session profile remains the fastest usable configuration in this test. [Context, concurrency, and generation evidence](../tools/ncs2/measurements/decoder-search-v1.json)
