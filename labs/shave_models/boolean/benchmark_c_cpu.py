"""Run the matched MagicEightBall benchmark with an optional compiled CPU lookup.

From an x64 Visual Studio developer shell, build the kernel in a scratch directory.

    cl /nologo /O2 /std:c17 /LD /GS- cpu_table_kernel.c /link /NOENTRY /NODEFAULTLIB /Brepro /OUT:cpu_table_kernel.dll

On Linux, use `cc -O3 -fPIC -shared cpu_table_kernel.c -o cpu_table_kernel.so`.
Pass the compiled library with --kernel. The C API accepts binary facts in
row-major order and writes the seven raw output bits as floats for each row.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from benchmark_compare import LocalRuntime, SUPPORT, TOKENIZER, digest, evaluate, file_details


WIDTH = 7


class CompiledTableRuntime(LocalRuntime):
    """Reuse the text and result handling of the existing CPU control."""

    def __init__(self, spec, kernel):
        super().__init__(spec, backend="table-cpu")
        if len(self.policy.literals) != WIDTH or len(self.policy.outputs) != WIDTH:
            raise ValueError("The compiled CPU baseline requires seven facts and seven raw outputs")

        self.library = ctypes.CDLL(str(kernel.resolve()))
        self.table_buffer = (ctypes.c_ubyte * len(self.table)).from_buffer_copy(self.table)
        self.batch = self.library.magic8ball_table_batch
        matrix = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
        self.batch.argtypes = (ctypes.POINTER(ctypes.c_ubyte), matrix, ctypes.c_uint, matrix)
        self.batch.restype = None

    def gather_heads(self, facts):
        features = np.ascontiguousarray(facts, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != WIDTH:
            raise ValueError("Expected one row of seven facts per decision")
        outputs = np.empty_like(features)
        self.batch(self.table_buffer, features, len(features), outputs)
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--include-fixtures", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be nonnegative")

    spec = json.loads(SUPPORT.read_text(encoding="utf8"))
    started = time.perf_counter()
    engine = CompiledTableRuntime(spec, args.kernel)
    initialized_ms = (time.perf_counter() - started) * 1000
    report = evaluate(engine, engine.policy, args.corpus, args.repeats, args.warmups, args.include_fixtures)
    report.update(
        backend="compiled-table-cpu",
        device="CPU",
        artifacts={
            "generated_uint8_truth_table": {"bytes": len(engine.table), "sha256": digest(engine.table)},
            "compiled_kernel_library": file_details(args.kernel),
            "kernel_c_source": file_details(Path(__file__).with_name("cpu_table_kernel.c")),
            "source_policy_json": file_details(SUPPORT),
            "shared_tokenizer_archive": file_details(TOKENIZER),
        },
        host=dict(processor=platform.processor(), platform=platform.platform(), logical_cpus=os.cpu_count(),
                  python=platform.python_version()),
        runner_initialization_ms=initialized_ms,
        timing_scope="The shared harness includes text preparation and result records in wall time. "
                     "Prepared head timing includes the Python-to-C call, packing, lookup, and unpacking.",
        artifact_scope="Truth table bytes count decision data. The compiled library includes native CPU code and "
                       "loader metadata, and receives the table from the host. The Python runner is separate.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({name: result["wall_ms"] for name, result in report["timings"].items()}))


if __name__ == "__main__":
    main()
