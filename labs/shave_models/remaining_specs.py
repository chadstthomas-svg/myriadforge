"""Frozen task specs for the remaining suites (docqa, asr, confidence, generative).

One module because these are DECLARED contracts - each becomes its own
`<id>_spec.py` with templates/fixtures when its lane starts (per SUITES.md
build order: after retrieval and vision-gate have device evidence).
"""

# --- docqa/1 ----------------------------------------------------------------
# Local document QA: span reader over retrieval-suite spans; host chunks, the
# stick reads. Cactus reference: Mirage text/OCR + Needle extraction.
DOCQA = {
    "version": "docqa/1",
    "input": "question tokens + retrieved span tokens (static-shape packed window)",
    "heads": "span start, span end, unanswerable (3-way over span positions)",
    "acceptance": {
        "em_min": 0.60,               # exact match on held-out docs
        "f1_min": 0.75,
        "unanswerable_rejection_min": 0.90,
        "min_heldout_docs": 100,
    },
    "gates_declared_before_sealed_set": True,
}

# --- asr/1 ------------------------------------------------------------------
# Windowed conv CTC dictation student. Cactus reference: Whistle / whisper.
ASR = {
    "version": "asr/1",
    "input": "[1,1,40,200] log-mel windows (the shared framing harness), CTC over a fixed vocab",
    "vocab": "English lowercase + space + apostrophe, ~40 symbols",
    "acceptance": {
        "wer_max": 0.35,              # constrained-domain dictation first
        "realtime_factor_max": 1.0,   # end-to-end including windowing
        "min_heldout_minutes": 20,
    },
    "gates_declared_before_sealed_set": True,
}

# --- confidence/1 (cross-cutting) ------------------------------------------
# Calibrated abstention + handoff for EVERY suite: thresholds frozen from dev
# sweeps (voicetools 0.8, vad 0.9 are the first instances), risk-coverage
# curves published per suite, every abstention counted.
CONFIDENCE = {
    "version": "confidence/1",
    "contract": "each suite reports (risk, coverage) pairs on held-out data; "
                "operating points frozen in the suite spec; abstentions never hidden",
    "acceptance": {
        "risk_coverage_curve": True,
        "abstentions_counted": True,
    },
}

# --- generative/1 (deliberately last) --------------------------------------
# Tiny static decoder (1-4M params, causal conv / recurrent), fixed vocab
# buckets, greedy decode on host. Entered only after docqa/asr have device
# evidence (SUITES.md build order).
GENERATIVE = {
    "version": "generative/0-draft",
    "status": "deferred",
    "reason": "hardest 2020.3-compiler fit, least differentiated vs host brains",
}
