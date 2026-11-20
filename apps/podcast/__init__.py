"""WS5 podcast pilot: WAV -> windows -> gate -> index.

Audio side of the 2026 applications. The NCS2's fit class covers small 1D
CNNs (VAD / keyword-spot class); ASR does not fit (autoregressive,
pointer-chasing - see the roadmap hardware envelope). This skeleton drives
the full pipeline (windowing -> gate -> ``ncs2.index/1``) with an honest
energy/RMS gate, and accepts a trained 1D-CNN gate model through the same
GateRunner contract when one exists.
"""
