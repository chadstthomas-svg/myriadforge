"""Run the MagicEightBall portable ONNX or GGUF profile on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from policy import LIMIT, load_tokenizer
from runtime import Runtime


ARCHITECTURE = "magiceightball"
PROFILE_REVISION = 1
TABLE = "decision.table"


class OnnxRequest:
    def __init__(self, model):
        from onnxruntime import InferenceSession, SessionOptions

        options = SessionOptions()
        options.intra_op_num_threads = 1
        self.session = InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
        self.outputs = [item.name for item in self.session.get_outputs()]

    def infer(self, tensors):
        return dict(zip(self.outputs, self.session.run(self.outputs, tensors)))


def read_gguf(path):
    from gguf import GGMLQuantizationType, GGUFReader

    model = GGUFReader(path)

    def metadata(key):
        field = model.get_field(key)
        if field is None:
            raise ValueError("Missing MagicEightBall GGUF metadata: " + key)
        return field.contents()

    if (metadata("general.architecture") != ARCHITECTURE
            or metadata("magiceightball.profile_revision") != PROFILE_REVISION
            or metadata("magiceightball.max_tokens") != LIMIT):
        raise ValueError("Unsupported MagicEightBall GGUF profile")
    manifest = {"name": ARCHITECTURE, "variant": "gguf",
                "policy": json.loads(metadata("magiceightball.policy_json"))}
    if len(model.tensors) != 1:
        raise ValueError("MagicEightBall GGUF needs exactly one decision tensor")
    tensor = model.tensors[0]
    if tensor.name != TABLE or tensor.tensor_type != GGMLQuantizationType.I8 or tensor.data.ndim != 1:
        raise ValueError("Expected a rank-one I8 decision.table tensor")
    return manifest, tensor.data.copy()


class PortableRuntime(Runtime):
    def __init__(self, artifact):
        path = Path(artifact).resolve()
        self.folder = path.parent
        self.lane = "cpu"
        self.target = "CPU"
        self.fast_count = True
        self.tokenizer = load_tokenizer()
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            self.bundle_info, self.table = read_gguf(path)
            self._use_bank(None)
            outputs = len(self.policy.outputs)
            if (not 1 <= outputs <= 7 or len(self.table) != 1 << len(self.policy.literals)
                    or np.any(self.table < 0) or np.any(self.table >= 1 << outputs)):
                raise ValueError("Decision tensor does not match the embedded policy")
        elif suffix == ".onnx":
            self.bundle_info = json.loads(path.with_name("model.json").read_text(encoding="utf8"))
            self._use_bank(None)
            model = path.read_bytes()
            if self.info["variant"] not in ("gather", "dense") or (
                    hashlib.sha256(model).hexdigest() != self.info["onnx_sha256"]):
                raise ValueError("Unexpected MagicEightBall ONNX artifact")
            self.output = self.info["decisions"]
            self.request = OnnxRequest(model)
        else:
            raise ValueError("Choose a MagicEightBall .onnx or .gguf file")

    def head(self, atoms):
        if self.info["variant"] != "gguf":
            return super().head(atoms)
        address = sum(int(bit) << index for index, bit in enumerate(atoms))
        word = int(self.table[address])
        return np.array([(word >> index) & 1 for index in range(len(self.policy.outputs))], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--explain", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    args = parser.parse_args()
    text = args.text if args.file is None else args.file.read_bytes().decode("utf-8-sig")
    runtime = PortableRuntime(args.artifact)
    try:
        outcome = runtime.decide(text, explain=args.explain)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(outcome, backend=runtime.target, format=args.artifact.suffix.lower()[1:])))


if __name__ == "__main__":
    main()
