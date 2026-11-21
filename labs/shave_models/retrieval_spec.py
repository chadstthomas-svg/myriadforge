"""Frozen task spec for the retrieval suite (retrieval/1).

Text embedding student + kNN-as-MatMul retrieval on-stick. The engine is
measured (10kx256 FP16, ~285 qps); this suite trains the ENCODER that makes
it a model: a small text encoder producing L2-normalized 128-d embeddings.

Training signal is our own paraphrase structure: template pairs with identical
tool+slots are positives; different calls are in-batch negatives (InfoNCE).
Cactus references: nomic-embed / Needle retrieval (host, HNSW).

Gates compare against the strongest non-learned baseline we ship
(`features.text_features`, the intent bundle's hashed-gram encoder).
"""

TASK_SPEC_VERSION = "retrieval/1"

EMBED_DIM = 128
TABLE_ROWS_MAX = 10_000      # the measured-good kNN engine shape (fit boundary below 25.6 MB)
TOP_K = 10

# Paraphrase-positive rule: two utterances are positives iff identical ordered
# tool calls (tool + slot values). Negatives: everything else in batch.
POSITIVE_RULE = "identical ordered tool calls"

ACCEPTANCE = {
    "recall10_min": 0.90,          # held-out slot-filled queries, 10k-row table
    "recall10_vs_hashedgram": 2.0, # ...and within 2 pts of the hashed-gram baseline
    "query_p95_ms_max": 4.0,       # on-stick batched query, engine envelope
    "min_heldout_queries": 300,
}
