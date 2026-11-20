"""Timing statistics and memory measurement."""

from __future__ import annotations

import ctypes
import math
import time
from collections.abc import Callable
from typing import Any


def percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an ascending-sorted list."""
    if not sorted_values:
        raise ValueError("percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def timed_loop(fn: Callable[[], Any], warmup_runs: int, measured_runs: int) -> dict[str, float]:
    """Run ``fn`` warmup_runs + measured_runs times; return latency stats in ms.

    Uses perf_counter_ns; every call is timed individually so percentiles
    are real per-iteration latencies, not batch averages.
    """
    for _ in range(warmup_runs):
        fn()

    samples_ms: list[float] = []
    for _ in range(measured_runs):
        start = time.perf_counter_ns()
        fn()
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    samples_ms.sort()
    mean = sum(samples_ms) / len(samples_ms)
    variance = sum((s - mean) ** 2 for s in samples_ms) / (len(samples_ms) - 1)
    return {
        "mean": mean,
        "p50": percentile(samples_ms, 50),
        "p90": percentile(samples_ms, 90),
        "p99": percentile(samples_ms, 99),
        "stddev": math.sqrt(variance),
        "min": samples_ms[0],
        "max": samples_ms[-1],
        "runs": float(len(samples_ms)),
    }


class ProcessMemory:
    """Peak + current working set for this process, via GetProcessMemoryInfo."""

    def __init__(self) -> None:
        self._kernel32 = ctypes.windll.kernel32
        self._psapi = ctypes.windll.psapi

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        self._counters_cls = _PROCESS_MEMORY_COUNTERS

    def snapshot_mb(self) -> dict[str, float]:
        counters = self._counters_cls()
        counters.cb = ctypes.sizeof(self._counters_cls)
        # GetCurrentProcess must return a real 64-bit HANDLE; without the
        # restype declaration ctypes truncates the pseudo-handle (-1) and
        # GetProcessMemoryInfo fails with ERROR_INVALID_HANDLE.
        self._kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        self._psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
        ]
        handle = self._kernel32.GetCurrentProcess()
        if not self._psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return {"peak_mb": 0.0, "current_mb": 0.0}
        # Field values are KiB on current Windows builds (verified against
        # allocation deltas) despite MSDN documenting bytes.
        return {
            "peak_mb": round(counters.PeakWorkingSetSize / 1024**2, 1),
            "current_mb": round(counters.WorkingSetSize / 1024**2, 1),
        }
