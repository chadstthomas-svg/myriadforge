"""Persistent MYRIAD runner for document-wide Boolean policies."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

import numpy as np
from policy import Policy, count_tokens, load_tokenizer, validate_count


@dataclass
class Prepared:
    tokens: int
    atoms: Optional[np.ndarray]
    projected: Optional[bytes]
    started: float
    counted: float
    ready: float


class Runtime:
    def __init__(self, bundle, toolchain=None, fast_count=True, lane="ncs1", target=None, blob=None):
        self.folder = Path(bundle).resolve()
        self.info = json.loads((self.folder/"model.json").read_text(encoding="utf8"))
        self.model_name = self.info["policy"].get("name", self.folder.name)
        self.policy = Policy(self.info["policy"])
        self.tokenizer = load_tokenizer()
        self.fast_count = fast_count
        self.lane = lane
        self.output = self.info["decisions"]
        if lane == "ncs1":
            if toolchain is None:
                toolchain = Path(__file__).resolve().parents[3] / "tools" / "ncs1"
            sys.path.insert(0, str(Path(toolchain).resolve()))
            import ncs1_paths
            ncs1_paths.setup()
            from openvino.inference_engine import IECore, get_version
            self.core = IECore()
        elif lane == "ncs2":
            from openvino.runtime import Core, get_version
            self.core = Core()
        else:
            raise ValueError("Unknown MYRIAD lane: " + lane)
        self.openvino_version = get_version()
        self.devices = list(self.core.available_devices)
        myriad = [name for name in self.devices if name.split(".", 1)[0] == "MYRIAD"]
        if not myriad:
            raise RuntimeError("No physical MYRIAD device visible to %s (%s)" % (lane, self.openvino_version))
        if target is None and len(myriad) != 1:
            raise ValueError("Choose --target from %s" % myriad)
        self.target = target if target is not None else myriad[0]
        if self.target not in myriad:
            raise ValueError("Expected a visible MYRIAD device from %s" % myriad)

        self.ir = None if blob is not None else self.resolve_ir()
        start = time.perf_counter()
        if lane == "ncs1":
            if blob is not None:
                self.compiled = self.core.import_network(str(blob), self.target, num_requests=1)
            else:
                net = self.core.read_network(model=str(self.ir), weights=str(self.ir.with_suffix(".bin")))
                if self.output not in net.outputs and not (self.info["variant"] == "gather" and len(net.outputs) == 1):
                    raise ValueError("Unexpected IR outputs: %s" % list(net.outputs))
                if self.info["variant"] == "gather":
                    net.outputs[next(iter(net.outputs))].precision = "FP16"
                self.compiled = self.core.load_network(network=net, device_name=self.target, num_requests=1)
            self.request = self.compiled
        else:
            self.compiled = (self.core.import_model(blob.read_bytes(), self.target) if blob is not None
                             else self.core.compile_model(str(self.ir), self.target))
            self.output = self.compiled.output(0 if self.info["variant"] == "gather" else self.output)
            self.request = self.compiled.create_infer_request()
        self.load_ms = 1000*(time.perf_counter()-start)

    def resolve_ir(self):
        source = self.folder/"model.onnx"
        model_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if model_hash != self.info["onnx_sha256"]:
            raise ValueError("Unexpected ONNX artifact")
        ir = self.folder/"ir"/model_hash
        xml = ir/"model.xml"
        binary = ir/"model.bin"
        if not xml.exists() or not binary.exists():
            if self.lane == "ncs1":
                import ncs1_paths
                ir.mkdir(exist_ok=True, parents=True)
                subprocess.run([sys.executable, str(ncs1_paths.MO/"mo_onnx.py"),
                                "--input_model", str(source), "--data_type", "FP16",
                                "--output_dir", str(ir)], timeout=120, check=True,
                               env=dict(os.environ, PYTHONPATH=str(ncs1_paths.MO)))
            else:
                from openvino.runtime import serialize
                from openvino.tools.mo import convert_model
                # Cache separately so a new toolchain never overwrites the IR10 pair.
                xml = ir / "2022.3" / "model.xml"
                if not xml.exists() or not xml.with_suffix(".bin").exists():
                    xml.parent.mkdir(exist_ok=True, parents=True)
                    serialize(convert_model(str(source), compress_to_fp16=True), str(xml))
        return xml

    def export_blob(self, path):
        if self.lane == "ncs1":
            self.compiled.export(str(path))
        else:
            path.write_bytes(self.compiled.export_model())

    def head(self, atoms):
        variant = self.info["variant"]
        if variant == "gather":
            return self.gather_heads([atoms])[0]
        if variant in ("lookup", "packed"):
            mask = sum(int(bit) << j for j, bit in enumerate(atoms))
            features = np.zeros(self.info["input_shapes"]["features"], dtype=np.float32)
            features[0, mask >> self.info.get("lane_bits", 0), 0, 0] = 1
            packed = self.request.infer({"features": features})[self.output].reshape(-1)
            self.validate_words(packed)
            if variant == "packed":
                return self.unpack(packed[0], mask)
            return np.array([(int(packed[k//10]) >> (k%10)) & 1
                             for k in range(len(self.policy.outputs))],dtype=np.float32)
        result = self.request.infer({"features": atoms.reshape(1,-1,1,1)})
        return result[self.output].reshape(-1).copy()

    @property
    def batch_capacity(self):
        return self.info["input_shapes"]["features"][0] if self.info["variant"] == "gather" else 1

    @staticmethod
    def validate_words(packed):
        if not np.isfinite(packed).all() or not np.array_equal(packed, packed.round()):
            raise RuntimeError("Invalid output from exact lookup head")

    def unpack(self, number, mask):
        lane = mask & ((1 << self.info["lane_bits"]) - 1)
        width = self.info["code_width"]
        offset = self.info["offset"]
        word = int(number) + offset
        code = (word >> (lane * width)) & ((1 << width) - 1)
        codebook = self.info["codebook"]
        if not 0 <= word < 2 * offset or code >= len(codebook):
            raise RuntimeError("Invalid codeword from exact packed head")
        return np.asarray(codebook[code], dtype=np.float32)

    def gather_heads(self, facts):
        if not 1 <= len(facts) <= self.batch_capacity:
            raise ValueError("Gather needs 1–%d decisions per request" % self.batch_capacity)
        masks = [sum(int(bit) << j for j, bit in enumerate(row)) for row in facts]
        addresses = np.zeros(self.batch_capacity, dtype=np.int32)
        addresses[:len(masks)] = [mask >> self.info["lane_bits"] for mask in masks]
        outputs = self.request.infer({"features": addresses})
        # The legacy ONNX converter renames the sole Gather output.
        if len(outputs) != 1:
            raise RuntimeError("Expected exactly one packed Gather result")
        words = next(iter(outputs.values())).reshape(-1)
        if words.size != self.batch_capacity:
            raise RuntimeError("Unexpected Gather result width")
        if self.lane == "ncs1" and words.dtype == np.int16:
            # IECore 2020.3 exposes native FP16 buffers as signed words.
            words = words.view(np.float16)
        used = words[:len(masks)]
        self.validate_words(used)
        return [self.unpack(word, mask) for word, mask in zip(used, masks)]

    def scan(self, projected):
        side = self.info["tile_side"]
        halo = max(map(len, self.policy.patterns))-1
        count = max(1, (len(projected)+side*side-1)//(side*side))
        storage = np.zeros(halo + count*side*side, dtype=np.float32)
        storage[halo:halo+len(projected)] = np.frombuffer(projected, dtype=np.uint8)
        prior = np.zeros((1,len(self.policy.literals),1,1), dtype=np.float32)
        # Each row includes the real preceding lexemes. A phrase can cross either
        # a row or a USB request boundary. Padding code 0 matches no literal.
        rows = np.lib.stride_tricks.as_strided(storage, shape=(count*side, side+halo),
                                               strides=(side*storage.itemsize,storage.itemsize),
                                               writeable=False)
        for i in range(count):
            tile = np.ascontiguousarray(rows[i*side:(i+1)*side])[None,None]
            result = self.request.infer({"lexemes": tile, "prior": prior})
            values = result[self.output].reshape(-1)
            n = self.info["state_channels"]
            prior = values[:n].reshape(1,n,1,1).copy()
        return values[n:].copy(), prior.reshape(-1), count

    def prepare_decision(self, text, prepare=None):
        start = time.perf_counter()
        projected = atoms = None
        scanning = self.info["variant"] == "scan"
        if prepare is not None:
            if scanning:
                raise ValueError("Batch lane selection needs a decision head that accepts fact bits")
            tokens, atoms, counted = prepare(text)
            validate_count(tokens)
        else:
            tokens = count_tokens(self.tokenizer, text, fast=self.fast_count)
            counted = time.perf_counter()
            if scanning or not self.fast_count:
                projected = self.policy.project(text)
                if not scanning:
                    atoms = self.policy.atoms(projected)
            else:
                atoms = self.policy.text_atoms(text)
        return Prepared(tokens, atoms, projected, start, counted, time.perf_counter())

    def decide_prepared(self, documents):
        if not 1 <= len(documents) <= self.batch_capacity:
            raise ValueError("Unexpected number of prepared documents")
        start = time.perf_counter()
        variant = self.info["variant"]
        if variant == "scan":
            document = documents[0]
            bits, atoms, calls = self.scan(document.projected)
            decisions = [(bits, atoms)]
        elif variant == "gather":
            facts = [document.atoms for document in documents]
            decisions = list(zip(self.gather_heads(facts), facts))
            calls = 1
        else:
            atoms = documents[0].atoms
            decisions = [(self.head(atoms), atoms)]
            calls = 1
        results = []
        for document, (bits, atoms) in zip(documents, decisions):
            result = self.policy.result(bits > 0.5)
            end = time.perf_counter()
            lexemes = len(document.projected) if document.projected is not None else None
            results.append(dict(
                model=self.model_name, result=result, bits=bits.tolist(), atoms=atoms.tolist(),
                needle_tokens=document.tokens, lexemes=lexemes, requests=calls, request_documents=len(documents),
                tokenization_ms=(document.counted-document.started)*1000,
                host_projection_ms=(document.ready-document.counted)*1000,
                inference_path_ms=(end-start)*1000, total_ms=(end-document.started)*1000,
            ))
        return results

    def decide(self, text, explain=False, prepare=None):
        record = self.decide_prepared([self.prepare_decision(text, prepare)])[0]
        if explain:
            record["explanation"] = self.policy.explain(text, record["atoms"])
        return record


def json_output_stream():
    # Legacy MYRIAD prints firmware messages to native stdout, including
    # during teardown. Save the caller's stream and redirect diagnostics
    # before loading the plugin so stdout remains a clean JSON protocol.
    sys.stdout.flush()
    stdout_fd = sys.stdout.fileno()
    output = os.fdopen(os.dup(stdout_fd), "w", encoding="utf8")
    os.dup2(sys.stderr.fileno(), stdout_fd)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt
        set_handle = ctypes.windll.kernel32.SetStdHandle
        set_handle.argtypes = (wintypes.DWORD, wintypes.HANDLE)
        set_handle.restype = wintypes.BOOL
        if not set_handle(0xFFFFFFF5, msvcrt.get_osfhandle(stdout_fd)):  # STD_OUTPUT_HANDLE
            output.close()
            raise OSError("Could not redirect native stdout")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--toolchain", type=Path, help="Override the NCS1 toolchain directory")
    parser.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs1")
    parser.add_argument("--target", help="Explicit OpenVINO name when several MYRIAD devices are visible")
    parser.add_argument("--blob", type=Path, help="Load a binary exported by the matching device/toolchain")
    parser.add_argument("--explain", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    source.add_argument("--serve", action="store_true", help="Read one JSON request with text and optional id per line")
    args = parser.parse_args()
    with json_output_stream() as output:
        engine = Runtime(args.bundle, args.toolchain, lane=args.lane, target=args.target, blob=args.blob)
        if args.serve:
            for line in sys.stdin.buffer:
                request_id = None
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("Expected an object with a text string")
                    request_id = request.get("id")
                    if not isinstance(request.get("text"), str):
                        raise ValueError("Expected an object with a text string")
                    response = engine.decide(request["text"], args.explain)
                except ValueError as exc:
                    response = {"error": str(exc)}
                print(json.dumps(dict(response, id=request_id)), file=output, flush=True)
        else:
            document = args.text if args.text is not None else args.file.read_bytes().decode("utf8")
            print(json.dumps(engine.decide(document, args.explain)), file=output)


if __name__ == "__main__":
    main()
