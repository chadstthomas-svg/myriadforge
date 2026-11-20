"""Capacity calculator: what vector table fits on the stick.

The Myriad X envelope gives ~1 GB of usable LPDDR4 for graph + weights
(roadmap hardware envelope; the hard budget gate MYRIAD.SIM enforces). This
module is the pre-flight arithmetic for kNN table sizing: FP16 bytes per
vector, max table size, and a go/no-go report.
"""

from __future__ import annotations

from dataclasses import dataclass

# Roadmap hardware envelope: ~1 GB usable for graph + weights (FP16 budget).
USABLE_BUDGET_BYTES = 1024**3
# Leave headroom for the graph, I/O buffers, and firmware arena.
HEADROOM_FRACTION = 0.10


@dataclass
class CapacityReport:
    n_vectors: int
    dim: int
    table_bytes_fp16: int
    budget_bytes: int
    fits: bool
    max_vectors: int
    utilization: float

    def as_dict(self) -> dict:
        return {
            "n_vectors": self.n_vectors,
            "dim": self.dim,
            "table_mb_fp16": round(self.table_bytes_fp16 / 1024**2, 1),
            "budget_mb": round(self.budget_bytes / 1024**2, 1),
            "fits": self.fits,
            "max_vectors_at_dim": self.max_vectors,
            "budget_utilization": round(self.utilization, 3),
            "note": "TopK runs on the host; only the GEMM lives on-device",
        }


def max_vectors(dim: int, budget_bytes: int = USABLE_BUDGET_BYTES) -> int:
    """Largest N for an N x dim FP16 table under the usable budget."""
    usable = int(budget_bytes * (1.0 - HEADROOM_FRACTION))
    bytes_per_vector = dim * 2  # FP16
    return max(0, usable // bytes_per_vector)


def check(n_vectors: int, dim: int, budget_bytes: int = USABLE_BUDGET_BYTES) -> CapacityReport:
    table_bytes = n_vectors * dim * 2
    budget = int(budget_bytes * (1.0 - HEADROOM_FRACTION))
    return CapacityReport(
        n_vectors=n_vectors,
        dim=dim,
        table_bytes_fp16=table_bytes,
        budget_bytes=budget,
        fits=table_bytes <= budget,
        max_vectors=max_vectors(dim, budget_bytes),
        utilization=table_bytes / budget if budget else float("inf"),
    )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="kNN table capacity vs the NCS2 budget")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()
    print(json.dumps(check(args.n, args.dim).as_dict(), indent=2))
