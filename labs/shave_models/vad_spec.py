"""Frozen task spec for the VAD suite (vad/1).

Voice-activity and audio-event gating: the always-on podcast/footage gate.
Host framing only; every learned stage on the stick. Reuses the voicetools
pipeline end to end (features.py mel filterbank, 2 s windows, trainer and
evaluator patterns) - vad is the first suite taken through the full contract
from scratch and the template for retrieval/docqa/asr.

Labels: speech / noise / silence. The v1 model is a tiny conv on the same
[1,1,40,200] window producing per-window class scores; events are decoded
host-side from window sequences (min-duration hysteresis is a harness rule,
not a graph rule - same split as voicetools' decision rule).

Bootstrap data (v0): voicetools speech fixtures (TTS conditions, held-out
cells) + its noise/silence clips + mini_speech_commands held-out speakers as
real-audio speech. Sealed set: new noise sources + unseen conditions.
"""

TASK_SPEC_VERSION = "vad/1"

# Windowing contract: identical to voicetools so one framing harness serves both.
WINDOW_SECONDS = 2.0
WINDOW_STRIDE = 1.0
MEL_BINS = 40
SAMPLE_RATE = 16000

# Class enums (index 0 reserved so "none/absent" stays a per-head convention).
CLASSES = ("speech", "noise", "silence")
CLASS_IDX = {name: i for i, name in enumerate(CLASSES)}

# Event decoding (host-side, from window sequences):
MIN_SPEECH_WINDOWS = 1        # a single speech window fires an event at v1
HYSTERESIS_SILENCE_WINDOWS = 2  # event closes after 2 consecutive non-speech windows
# Speech-fire threshold, re-frozen for the gap-head model (2026-11-20 device
# sweep): 0.9 = recall 98.2% (gate 98) AND 0.0 false wakes/hour; 0.8 leaks
# 3.9/hour. Note: frozen from the diagnostic-split device sweep - the sealed
# run uses fresh unseen noise, so this is the diagnostic operating point.
FIRE_THRESHOLD = 0.9

# Fixture bootstrap (v0, all already built and digest-verified):
#   speech  = voicetools single/pair utterance windows (TTS, 30 condition cells)
#   noise   = voicetools neg_silence clips (low-level deterministic noise)
#   silence = voicetools neg_silence trailing windows + pure zero windows
#   real-speech anchor = mini_speech_commands held-out-speaker clips (real audio)
V0_SOURCES = {
    "speech": "voicetools manifest kinds single+pair (windows)",
    "noise": "voicetools manifest kind neg_silence",
    "silence": "zero-pad windows",
    "real_speech": "mini_speech_commands held-out speakers (SHA-256 in fetch_data.py)",
}

# Acceptance gates (declare BEFORE opening any sealed set). The false-wake
# budget is expressed per hour of held-out non-speech at 1 s hop.
# p95 budget note (revised twice, 2026-11-20 - falsified hypotheses recorded):
# (1) 2.5 ms was below the dispatch+transfer physics floor. (2) 6.0 ms assumed
# the torch-export path cost the extra ~3 ms - FALSIFIED: a builder-built
# opset-11 transplant of the SAME weights measured 8.8/14.8 ms with byte-
# identical outputs, so the variance is SESSION/USB-bound (the handoff 5.0 ms
# was a different session). Final gate: absolute p95 <= 12 ms sync AND a
# session-relative paired-gap check (vad p95 <= 2x a zero-compute gap model of
# the same input shape, measured in the same session) - the compute_probe
# method that cancels host-USB state.
ACCEPTANCE = {
    "window_accuracy_min": 0.95,      # all three classes, held-out conditions
    "speech_recall_min": 0.98,        # speech windows correctly gated IN
    "false_wake_per_hour_max": 1.0,   # non-speech windows misclassified as speech
    "p95_window_ms_max": 12.0,        # device inference per window, warm, batch 1
    "paired_gap_ratio_max": 2.0,      # vad p95 <= 2x same-session gap-model p95
    "min_held_out_noise_clips": 300,  # independent noise sources in the sealed set
}
