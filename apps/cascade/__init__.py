"""WS5 cascade pilot: NCS2 gate fires -> host Mirage describes the frame.

The always-on-gate -> host-brain loop from the roadmap MVP, promoted from
the demo layer: the gate now runs on the stick (MYRIAD, ~1 W) instead of the
LiteRT stand-in, and only threshold crossings wake the host brain
(cactus-compute Mirage, 22 MB VLM on CPU). The latency chain
(gate_ms + mirage_ms) prints end-to-end; the Mirage leg is an honest skip
when cactus-compute is not installed.
"""
