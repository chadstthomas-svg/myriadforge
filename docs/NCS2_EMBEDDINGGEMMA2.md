# EmbeddingGemma 2 projection offload on Intel NCS2

EmbeddingGemma 2 is a Gemma 4-based multimodal embedder. The pinned text model contains 271,002,624 trained parameters, about 517 MiB in FP16 before runtime overhead. The full checkpoint contains about 1.39 GiB of weight data. The tested NCS2 runs two learned projections from the same text checkpoint with OpenVINO 2022.3.1. [Google model overview](https://ai.google.dev/gemma/docs/embeddinggemma) · [Pinned checkpoint](https://huggingface.co/google/embeddinggemma-2/tree/f9972ac22e696a105c482202646979e33e4b5a1e)

| Mode | Device work | Host work |
| --- | --- | --- |
| `cpu` | — | Text-only model, pooling, normalization, retrieval |
| `head` | Learned 512 × 768 output projection | Text backbone, pooling, normalization, retrieval |
| `ple-head` | Output projection and learned 512 × 12,288 per-layer embedding projection | Tokenizer, text blocks, per-layer RMS normalization, pooling, normalization, retrieval |

The output head has no bias, so linear projection commutes with mask-aware mean pooling. The shim pools the 512-wide representation on the host and projects one row per text on the stick. The per-layer projection consumes a row per token and returns 12,288 values each time. Both device graphs stay loaded across the batch. Query and document vectors are truncated to the same Matryoshka dimension and normalized in FP32 before cosine search. [Transformers model implementation](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py) · [Pooling and normalization](https://huggingface.co/docs/transformers/model_doc/embedding_gemma2)

## Run it

Use a modern host Python environment with PyTorch, NumPy, and Transformers 5.24.0. Keep the separate OpenVINO 2022.3.1 Python 3.10 environment for the stick. Download the pinned Hugging Face snapshot to a local directory. The shim disables the audio and vision towers when loading the checkpoint.

~~~powershell
$ncs2 = 'tools/ncs2/.venv/Scripts/python.exe'
$checkpoint = 'models/embeddinggemma2'
$fixture = 'tools/ncs2/measurements/embeddinggemma2-industry.jsonl'
python -m apps.embeddinggemma2 $checkpoint $fixture --mode head --ncs2-python $ncs2 --dimension 256 --compare --repeats 5 --max-tokens 128 --out results/embeddinggemma2.json
~~~

Supply JSONL rows with unique string IDs and text. Set the kind to query to search the document rows in the same file. Documents may include a title. The output includes normalized embeddings, top-ranked documents, and timing data. Comparison mode uses the full CPU text path as its reference and checks all four supported dimensions.

~~~jsonl
{"id":"recall","title":"Recall lookup","text":"Use the vehicle identification number to find open safety campaigns."}
{"id":"query","kind":"query","text":"Does this VIN have an outstanding recall?"}
~~~

## Measured on Myriad X

The fixture contains nine documents and six queries across automotive service, manufacturing, and retail security. Each mode used one warm-up and five measured passes over the same 15 texts. The host used two CPU threads, batches of eight, and a 128-token limit.

| Offload | Static device rows | Learned device weights | CPU median | Hybrid median |
| --- | ---: | ---: | ---: | ---: |
| Output projection | 16 | 393,216 | 992 ms | 957 ms |
| Output and per-layer projections | 16 | 6,684,672 | 987 ms | 2,040 ms |
| Output and per-layer projections | 64 | 6,684,672 | 951 ms | 2,101 ms |

Every mode preserved all six CPU top-ranked documents and matched all six authored labels at 128, 256, 512, and 768 dimensions. The lowest embedding cosine across the runs was 0.9999965. Output-only offload showed overlapping CPU and hybrid timings. The larger offload moved 12.75 MiB of FP16 weight equivalents to the device and took about twice as long end to end. Each token's wider per-layer result must return to the CPU before the transformer can use it. [Fixture](../tools/ncs2/measurements/embeddinggemma2-industry.jsonl) · [Raw benchmark evidence](../tools/ncs2/measurements/embeddinggemma2-ncs2-v1.json)

## Scope

The tokenizer, attention and transformer blocks, and the retrieval index remain on the host. The vision and audio towers, including the video path, are outside this text shim. Full-model execution on the stock NCS2 runtime remains unverified. The shim supports explicit CPU mode for low-overhead text search and explicit NCS2 modes for measured device offload.
