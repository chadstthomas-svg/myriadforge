"""WS5 footage-triage pilot: video -> sampled frames -> NCS2 gate -> index.

The 2026 footage application shape: one pass over a clip, the gate (MYRIAD
on the stick, ~1 W) classifies sampled frames, and every threshold crossing
lands in an ``ncs2.index/1`` JSONL (timestamps + labels + confidences) that
downstream search/editing consumes. Runs on any OpenVINO device;
``--device myriad`` = the stick.
"""
