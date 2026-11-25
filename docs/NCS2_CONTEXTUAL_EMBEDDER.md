# Contextual retrieval on Intel NCS2

This experiment adds a 32-dimensional contextual residual to the 5,120-dimensional static and keyword encoder. The host groups 128 WordPieces into eight ordered groups and builds signed unigram and bigram counts. A single FP16 MYRIAD graph projects those groups, applies masked attention, pools the result, and combines it with the existing vector. The contextual channel contributes 20% of the cosine score when both channels are nonzero.

Four adapter variants share the same training data and feature construction. They use a feature-only projection, one attention pass, two passes with shared weights, or two independently weighted passes. The recurrent block uses one attention head, a width-64 feed-forward layer, and gates initialized to smaller updates on the first pass. The MYRIAD export uses a rank-three view for normalization and explicit per-channel scale constants for the final weighted vectors. The latter avoids a numerical error observed with scalar broadcast on this runtime.

## Physical results

| Encoder | NFCorpus nDCG@10 | Industry pair accuracy | NCS2 p95 across three runs |
|---|---:|---:|---:|
| Frozen static and keyword | 0.317 | 0.0% | 11.7–12.1 ms |
| Feature-only adapter | 0.297 | 23.3% | 12.9–13.4 ms |
| One pass | 0.311 | 33.3% | 14.9–15.4 ms |
| Two shared passes | 0.297 | 31.7% | 16.6–17.3 ms |
| Two independent passes | 0.304 | 34.4% | 16.6–17.0 ms |
| EmbeddingGemma 2, 768 dimensions, CPU | 0.357 | — | — |

All four adapters passed the physical checks and the 25 ms gate. The shared two-pass adapter had worst cosine agreement above 0.9996 with its FP32 CPU reference. It scored 0.800 on SciFact, down from 0.864, and trailed the one-pass adapter on both primary quality metrics. The release decision is to retain the static and keyword encoder. [Machine-readable results and timing samples](../tools/ncs2/measurements/contextual-embedder.json)

The industry set is a targeted synthetic challenge. The frozen baseline tied 120 of 180 pairs and a tie counts as a miss. The industry gains describe this benchmark's precedence, negation, and state-change examples.

## Data and training

- Train on 8,192 SQuAD 1.1 question and passage pairs with hard negatives selected by lexical overlap from different articles. Retain examples whose answer fits the 128-WordPiece window. Hold out 512 pairs from disjoint articles for checkpoint selection. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- Train on 216 synthetic industry pairs and select checkpoints with 108 additional pairs. The 180 test pairs use separate workflows and wording, balanced by vertical and by precedence, negation, and state changes. Record paired bootstrap intervals clustered by the underlying industry workflow and phenomenon. A tied score does not count as a correct retrieval.
- Train for three epochs with AdamW at 0.001, mixing 24 public pairs and eight industry pairs per step. Use 70% supervised pair ranking and 30% teacher similarity distillation. Select the earliest checkpoint with the highest mean pair accuracy across the two development sets. The training cache uses the full 768-dimensional text teacher with CUDA BF16 autocast. The independent public comparison uses the FP32 CPU teacher.
- Evaluate on the 323 BEIR NFCorpus test queries against all 3,633 documents with graded relevance. Use the same catalog-derived keyword weights for the baseline and every adapter for a given set. Replay the frozen SciFact and HotpotQA samples as regression checks. [BEIR NFCorpus corpus and queries](https://huggingface.co/datasets/BeIR/nfcorpus/tree/b5026a0e96e8a7ac4f95f482a596389289d46269) · [BEIR test judgments](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/blob/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65/test.tsv)

## Reproduce

Use the static encoder assets and pinned EmbeddingGemma 2 snapshot from the [static encoder guide](NCS2_STATIC_EMBEDDER.md). Create a scratch directory and download the [SQuAD 1.1 training split](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) as `squad.json`. Download the pinned BEIR corpus and query Parquets as `nfc-corpus.parquet` and `nfc-queries.parquet`, and save the pinned test judgments as `nfc-test.tsv`.

Run from the repository root. Use a modern Python environment with PyTorch, NumPy, Tokenizers, PyArrow and Transformers 5.24.0 for preparation and training. Use the existing Python 3.10 runtime with OpenVINO 2022.3.1 and Tokenizers 0.22.2 for hardware tests. Make `src` importable in both environments.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m tools.ncs2.prepare_contextual_embedder $scratch $staticAssets $teacher --only fit --device cuda
python -m tools.ncs2.prepare_contextual_embedder $scratch $staticAssets $teacher --only public --device cpu --batch 8
python -m tools.ncs2.train_contextual_embedder $scratch $staticAssets
foreach ($variant in 'baseline','features','one','tied','untied') {
    & $legacyPython -m tools.ncs2.measure_contextual_embedder $scratch $staticAssets --variant $variant
}
& $legacyPython -m tools.ncs2.measure_contextual_embedder $scratch $staticAssets --summarize
```

The training commands also support `--device cpu`. Sample latency with the host otherwise idle. Use `--retime` with the same device and model to refresh latency and industry vector checks without rescoring the public corpus. The report includes all three warmed runs, each with 50 queries and 50 documents, CPU vector agreement and compiled blob size. Blob size describes the serialized device program.

Use a trained adapter with the following opt-in command. Its fingerprint covers the tokenizer, static table, keyword weights and adapter bundle. Re-embed a catalog with the same fingerprint before searching the 5,152-dimensional vectors.

```powershell
& $legacyPython -m apps.static_embedder "$staticAssets/tokenizer.json" "$staticAssets/base1024-fp16.npy" "Find the recall verification procedure" --backend myriad --lexical-weights "$scratch/nfcorpus-lexical.npy" --context-weights "$scratch/context-tied.npz"
```

## Decision rules

Qualify an adapter when it gains at least 0.01 NFCorpus nDCG@10 and five percentage points of industry pair accuracy over the frozen baseline, with at most 0.01 nDCG@10 loss on either regression set. Require all three full-encoder p95 measurements to be at most 25 ms. The hardware vectors must meet the cosine, norm, padding and truncation checks.

Promote two shared passes when they qualify and improve over one pass by at least 0.005 nDCG@10 or three industry percentage points without a loss on the other measure. Otherwise use the qualifying one-pass adapter or retain the frozen baseline. A 95% paired interval with a lower bound of at least −0.03 against the 768-dimensional teacher is required for a parity claim on NFCorpus.
