# Full text embeddings on Intel NCS2

The Intel NCS2 runs the full learned encoder in one FP16 MYRIAD graph. The host tokenizes English text and counts tokens. The stick applies the learned embedding table, pools token contributions, applies optional corpus-frequency weights, and returns a normalized vector. Cosine search runs on the host. The runtime targets MYRIAD directly.

The 256-dimensional distilled model contains 7,813,632 parameters in a 14.9 MiB weights file. The higher-quality model uses the original 1,024-dimensional table and adds a balanced 4,096-bucket keyword channel. It uses 59.6 MiB of FP16 semantic weights and 16 KiB of frequency weights and returns a 5,120-dimensional vector. These measurements are file sizes. VPU allocator memory and power are unmeasured.

## Physical results

| Encoder | Device | SciFact nDCG@10 | HotpotQA nDCG@10 |
|---|---|---:|---:|
| EmbeddingGemma 2 at 768 dimensions | CPU | 0.919 | 0.967 |
| EmbeddingGemma 2 at 256 dimensions | CPU | 0.878 | 0.954 |
| Static table plus keyword weighting | NCS2 | 0.864 | 0.944 |
| Static table at 1,024 dimensions | NCS2 | 0.808 | 0.915 |
| Distilled table at 256 dimensions | NCS2 | 0.725 | 0.802 |

The combined model trailed the full teacher by 0.055 on the exploratory SciFact sample and 0.023 on HotpotQA. The HotpotQA point estimate fell within the 0.03 target. Its paired 95% bootstrap interval for the difference ran from −0.054 to +0.006, so the experiment leaves statistical parity unresolved. The same model returned the expected first result on all six Rack Lab automotive, manufacturing, and retail smoke queries.

| Encoder | SciFact median / p95 | HotpotQA median / p95 |
|---|---:|---:|
| EmbeddingGemma 2 on CPU | 142.57 / 198.27 ms | 102.86 / 206.11 ms |
| Combined model on NCS2 | 10.61 / 11.66 ms | 10.29 / 10.71 ms |
| Combined model on CPU | 0.45 / 1.05 ms | 0.23 / 0.72 ms |

The combined model's worst cosine agreement with its CPU reference was above 0.999999 across the two samples. The CPU and NCS2 versions returned the same aggregate retrieval scores on both. The 5,120-dimensional vectors require 6.7 times as many stored values per document as the full teacher. [Raw hardware reports and timing samples](../tools/ncs2/measurements/static-embedder/)

## Method

The semantic input is capped at 128 WordPieces. The keyword input uses the first 128 matched words. A deterministic signed hash turns word counts into features. The frequency weights are fitted from unlabeled catalog documents and remain fixed for query encoding. An index and its query encoder must use the same table, tokenizer, and frequency weights.

The first experiment fit a ridge projection to EmbeddingGemma 2 targets from 256 FiQA documents picked by SHA-256 of their IDs. It combined the fitted projection with a residual from the frozen English retrieval table. The hardware graph produced finite near-unit vectors, and the adapted weights reduced nDCG relative to the frozen 256-dimensional control. The second experiment chose the keyword channel on SciFact and checked it on separate HotpotQA queries.

Each retrieval sample included all 50 queries, every judged document, and SHA-256-selected distractors to 384 documents. Both samples use binary judgments. The CPU teacher used its retrieval prompts, a 128-token cap, and four CPU threads. Paired bootstrap intervals quantify uncertainty over query selection on these samples. Latencies include tokenization and warmed batch-one encoding on the same i7-12650H host. The timed calls comprise 12 documents and 12 queries per dataset. Ranking and compilation are outside the request timing.

Native high token IDs in the first Gather graph produced incorrect vectors during the hardware probe. The deployed graph uses a token histogram and a learned matrix multiply, which computes the same sum of token vectors. OpenVINO 2022.3.1 on Windows compiled and ran the graph on a physical Myriad X, using 16 SHAVEs and 16 CMX slices in one stream.

## Reproduce

Run from the repository root. The fit step uses a modern Python environment with PyTorch, Transformers 5.24.0, PyArrow, Safetensors, NumPy, and Tokenizers. The device step uses the Python 3.10 legacy runtime from the [NCS2 tuning guide](NCS2_SHAVE_TUNING.md) with Tokenizers 0.22.2. Make the repository's `src` importable in both environments.

Keep these pinned inputs in a scratch directory.

| Input | Local files |
|---|---|
| [Sentence Transformers static retrieval table](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/tree/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding) | Save `model.safetensors` as `static-table.safetensors` and keep `tokenizer.json`. |
| [NanoFiQA2018](https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018/tree/4163ba032953d5044a7a6244261413f609c14342) | Save `corpus/train-00000-of-00001.parquet` as `fiqa-corpus.parquet`. |
| [NanoSciFact](https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact/tree/309f1d1ae3ae2e092444a8a0c25bed59b82318bc) | Save the corpus, queries, and qrels parquet files as `scifact-corpus.parquet`, `scifact-queries.parquet`, and `scifact-qrels.parquet`. |
| [NanoHotpotQA](https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA/tree/d79c0cdda980aba54842756770928035e1b61a51) | Save the three parquet files as `hotpot-corpus.parquet`, `hotpot-queries.parquet`, and `hotpot-qrels.parquet`. |
| [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2/tree/f9972ac22e696a105c482202646979e33e4b5a1e) | Pass the downloaded checkpoint directory as `$teacher`. The comparison loads its text components. |

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m tools.ncs2.fit_static_embedder $scratch $teacher
python -m tools.ncs2.fit_static_embedder $scratch $teacher --dataset hotpot --skip-fit
& $legacyPython -m tools.ncs2.measure_static_embedder $scratch --variant hybrid
& $legacyPython -m tools.ncs2.measure_static_embedder $scratch --dataset hotpot --variant hybrid
& $legacyPython -m apps.static_embedder "$scratch/tokenizer.json" "$scratch/base1024-fp16.npy" "Find the service procedure for a brake rotor" --backend myriad --lexical-weights "$scratch/scifact-lexical4096.npy"
```

Use `--variant distilled` for the fitted 256-dimensional model and `--variant static1024` for the semantic-only control. To build a keyword channel for a new catalog, pass the catalog's document strings to `fit_lexical_weights` from `apps.static_embedder`, save the returned NumPy array, and use it to encode both documents and queries.
