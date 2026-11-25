"""Serve static EmbeddingGemma 2 linear projections from the legacy Myriad runtime."""

from argparse import ArgumentParser
import base64
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
from openvino.runtime import Core, Model, get_version, opset8 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from myriadforge.myriad_x import allocation_properties, require_myriad_x

MARKER = "MYRIAD_EMBEDDING "
WIDTH = 512
OUTPUTS = {"head": 768, "ple": 12288}


def reply(data):
    print(MARKER + json.dumps(data, allow_nan=False), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--shaves", type=int, default=16)
    args = parser.parse_args()
    if args.block < 1:
        parser.error("Block size must be positive")
    try:
        core = Core()
        device = require_myriad_x(core)
        props = allocation_properties(args.shaves, args.shaves, 1)
        stages = {}
        timings = {}
        with np.load(args.weights, allow_pickle=False) as weights:
            if "head" not in weights or any(name not in OUTPUTS for name in weights.files):
                raise ValueError("The bundle must contain the head and optionally the PLE projection")
            for name in weights.files:
                array = weights[name]
                if array.shape != (OUTPUTS[name], WIDTH) or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {name} projection weights")
                x = op.parameter([args.block, WIDTH], np.float32)
                y = op.matmul(x, op.constant(array), False, True)
                start = perf_counter_ns()
                compiled = core.compile_model(Model([y], [x], name), "MYRIAD", props)
                timings[name] = (perf_counter_ns() - start) / 1e6
                stages[name] = (compiled, compiled.create_infer_request())
        reply({"status": "ready", "device": device, "runtime": get_version(), "compile_ms": timings})
    except (OSError, ValueError, RuntimeError) as error:
        reply({"status": "error", "detail": str(error)})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            name, rows = request["stage"], request["rows"]
            if name not in stages or type(rows) is not int or not 1 <= rows <= args.block:
                raise ValueError("Invalid projection stage or row count")
            raw = base64.b64decode(request["data"], validate=True)
            matrix = np.frombuffer(raw, dtype="<f4").reshape(rows, WIDTH)
            values = np.zeros((args.block, WIDTH), dtype=np.float32)
            values[:rows] = matrix
            if not np.isfinite(matrix).all():
                raise ValueError("Projection input must be finite")
            compiled, infer = stages[name]
            start = perf_counter_ns()
            result = infer.infer({compiled.input(0): values})[compiled.output(0)]
            device_ms = (perf_counter_ns() - start) / 1e6
            result = np.ascontiguousarray(result[:rows], dtype="<f2")
            if not np.isfinite(result).all():
                raise ArithmeticError("MYRIAD returned a non-finite projection")
            reply({"status": "ok", "device_ms": device_ms, "data": base64.b64encode(result).decode("ascii")})
        except (KeyError, TypeError, ValueError, ArithmeticError, RuntimeError) as error:
            reply({"status": "error", "detail": str(error)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
