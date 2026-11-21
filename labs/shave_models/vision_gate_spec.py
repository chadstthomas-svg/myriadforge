"""Frozen task spec for the vision-gate suite (vision-gate/1).

Task-specific image gate: replaces the ImageNet MobileNetV2 gate whose
confident mis-fires on cartoon content saved NOTHING in the measured cascade
(59-60/60 fired at every threshold). Design follows the paired-gap ceiling:
compute-DENSE convs (35.8 GFLOPS sustained) beat the memory-bound 1x1/FC
pattern (2.9 GFLOPS) - a small stack of 3x3 convs, no depthwise.

v1 task (frozen): distinguish natural-photo content from non-trigger content
(animation, screenshots, text slides, static frames) - the cascade's actual
need on our footage pipeline. Host framing; the gate runs on-stick at batch 1
with async pipelining (the +48% throughput shim).

Cactus reference: Mirage's implicit gate role (it punts wake/gate to the user).
"""

TASK_SPEC_VERSION = "vision-gate/1"

# Input contract: the footage pipeline's frames (MobileNetV2-7 preprocessing).
INPUT_SHAPE = [1, 3, 224, 224]
NORMALIZE = "scale 1/255, then (x-0.5)/0.5"

# Design rule from docs/RESEARCH_MYRIAD2_LIMITS_AND_MIRAGE.md paired-gap data:
# 3x3 convs ~35.8 GFLOPS sustained; 1x1/FC ~2.9; depthwise = MobileNetV2's
# trap. Keep channels even, shapes static, opset <= 11, FP16.
DESIGN_RULE = "stack of 3x3 convs (dense), no depthwise, no 1x1-heavy head"

ACCEPTANCE = {
    "accuracy_min": 0.90,            # trigger vs non-trigger, held-out footage
    "fire_rate_on_nontrigger_max": 0.10,  # the measured MobileNetV2 failure mode
    "recall_on_trigger_min": 0.95,   # cascade usefulness
    "p50_ms_max": 60.0,              # sync batch 1 (the measured MobileNet envelope)
    "min_nontrigger_fixtures": 300,
}
