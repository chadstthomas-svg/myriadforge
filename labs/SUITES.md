# MyriadForge model suites

A **suite** is one first-party model family covering one capability row, built
from scratch on our own data, with Cactus models as references to beat - not
dependencies. Every suite ships the same artifact contract, so "the model
exists" always means the same thing regardless of which suite you are looking
at. The machine-readable registry is [`suites.json`](suites.json) (schema
`myriadforge.suites/1`); this document is the contract and the build order.

## The suite contract

Every suite ships, in this order:

1. **`<id>_spec.py`** - the frozen task spec: versioned (`<id>/1`), enums,
   templates, windowing contract, and acceptance gates declared **before** any
   sealed set is opened. Changing anything bumps the version and re-records
   every downstream artifact.
2. **Fixtures + manifest** - live outside the repo (`D:/models/ncs-shave-research`),
   built by a deterministic seeded script, SHA-256 verified, with a
   label-vs-input consistency audit at zero.
3. **Trainer** (`train_<id>.py`) - honest metrics in `model.json` (arch,
   recipe, history); the exported digest is what the device run replays.
4. **Bundle** (`bundles/<id>/`) - ONNX + manifest + FP16 IR (or blob) with
   SHA-256 digests; registered in `models/registry.json` once it compiles.
5. **Device evaluation** (`eval_<id>.py`) - isolated run dirs (prior reports
   survive), digests, timing boundary, finiteness check, and the
   negative-count gate reported explicitly. Raw calls per fixture ID so
   device/CPU agreement is computable from the reports alone.
6. **Release path** - GitHub evidence first (digest-matched results on the
   lab README + Pages), Hugging Face only where licensing is clear (see the
   boolean suite for the working example: MIT code + Apache-2.0 attribution
   for derived tokenizers).

**Strict-stick claim rule:** every learned stage on the identified physical
stick (cold USB identity recorded); CPU, simulator, HETERO and network runs
are recorded as *distinct* measurements, never blended.

## The suites

Coverage maps to the Cactus capability roadmap plus suites that have **no
Cactus analog** (novel). Statuses: `spec` → `fixtures` → `trained` →
`device-evaluated` → `release-lane`.

### Decisions and tools (Cactus row 1: Needle)

| Suite | Capability | Status | Evidence |
|---|---|---|---|
| **boolean (MagicEightBall)** | Staged Boolean policy → yes/no + category + action, up to 131,072 tokens | **release-lane evidence complete** | 128/128 physical, 30-min soak (107,104 requests), 7-backend matched comparison, cold identity recorded |
| **voicetools** | Voice → ordered typed tool calls (acoustic Needle) | **sealed-evaluated v6** (LibriSpeech real-speaker negatives) | iterations: 19.0 → 16.9 → 11.88 (slot abstention) → **11.40% sealed false**; diagnostic **exact 77.6% / false 2.7%** (both improved by real-speaker data); 90/1 gates = data-scale lane; p95 20.2 ms PASS |
| voice | Isolated command classification (wake/kws primitive) | evidence complete | 90.7% @ p95 3.89 ms |
| intent | Text intent routing (91 SLURP labels) | evidence complete | 80.1% @ p95 3.08 ms |
| grounding | Browser control ranking | experimental | 7.4% — first bottleneck named (32-candidate retrieval limit) |

### Perception (Cactus rows 2-3: retrieval, vision)

| Suite | Capability | Status | Evidence |
|---|---|---|---|
| forms | UI control localization from pixels | evidence complete | 89.2% @ p95 45.49 ms |
| retrieval | Encoder student + kNN-as-MatMul search on-stick | **trained v2** (contrastive, 1 MB FP16 encoder) | **beats the hashed-gram baseline +23 pts** (0.68 vs 0.45 Recall@10); 0.90 absolute gate = richer inputs/capacity/corpus iteration; engine measured 285 qps |
| vision-gate | Task-specific image gate — fixes the measured cascade failure | v0 sealed **FAIL** → v1 task corrected | diagnostic (BBB vs COCO) all PASS; sealed ToS unseen animations fire 100% — the gate keys on cartoon-ness, not semantics. v1: trigger = ALL footage, non-trigger = desktop UI (forms renders); matches the real triage need |
| docqa | Span reader over retrieval spans + unanswerable rejection | spec (after retrieval) | gates: EM/F1 on held-out docs |

### Speech (Cactus row 4)

| Suite | Capability | Status | Evidence |
|---|---|---|---|
| vad | Voice activity + audio events (the always-on podcast gate) | **device-evaluated — ALL GATES PASS** | paired-gap method (single session, 600 windows): 96.75% acc, 98.2% recall, 0.0 false wakes/h, absolute p95 **3.89 ms**, ratio **1.59×** vs the 2.45 ms gap floor — the earlier 9.8 ms was session/USB variance, now instrumented away |
| asr | Windowed conv CTC dictation | spec (after vad) | gates: real-time factor ≤1 at acceptable WER |

### Novel (no Cactus analog)

| Suite | Capability | Status | Evidence |
|---|---|---|---|
| cascade-gate | Always-on gate orchestrating host brains — the composition itself is the suite | measured (mismatched gate) | cascade bench on real content; fix arrives via vision-gate + voicetools |
| confidence | Calibrated abstention + handoff, cross-cutting | spec | risk-coverage curves per suite; every abstention counted |

### Generative breadth (Cactus row 5)

Deliberately **last**: tiny static decoders (1-4M params, causal conv) are the
hardest fit for the 2020.3 compiler and the least differentiated vs host
brains. Entered as a suite only after vad/retrieval/docqa have device
evidence.

## Build order

1. **boolean** → GitHub release prep (model card, LICENSE, Boolean suite in CI) — the release template for every other suite
2. **voicetools** → device eval of `ebc7ba2ce98e…` (needs a replug), then utterance-level aggregation + abstention calibration against the 65% call-count error class; sealed ≥300-negative set
3. **vad** → first fully from-scratch suite through the whole contract (reuses the voicetools pipeline end to end: features, windows, trainer, evaluator)
4. **retrieval** → encoder student makes the measured kNN engine a real model suite
5. **vision-gate** → replaces the ImageNet gate; cascade v2 numbers become honest
6. **docqa → asr → generative** in that order

## What "from scratch" means here

Own data (synthesized or public-domain with SHA-256 provenance), own training
recipes, own FP16 graphs through the 2020.3 compiler — Cactus checkpoints are
probed for reference scores and teacher selection only, and any derived
artifact (like the Needle 3 tokenizer in the boolean suite) carries explicit
upstream license attribution.
