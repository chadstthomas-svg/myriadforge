"""Transport a Gather manifest and both native 2020.3 blobs losslessly."""
import argparse
import json
from pathlib import Path
import struct
import zlib

from blob_layout import inspect_blob


MAGIC = b"M8B1"
HEADER = struct.Struct("<4s3I")
TARGETS = {"ncs1": ("ncs1-native.blob", "VPU_MYRIAD_2450"),
           "ncs2": ("ncs2-2020.3.blob", "VPU_MYRIAD_2480")}


def _validate(manifest, blobs):
    if not isinstance(manifest, dict) or manifest.get("variant") != "gather":
        raise ValueError("Release requires a Gather manifest")
    inputs = manifest.get("input_shapes")
    shape = inputs.get("features") if isinstance(inputs, dict) else None
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
        raise ValueError("Release has invalid Gather geometry")
    for blob in blobs.values():
        layout = inspect_blob(blob)
        if (layout["input_bytes"] != shape[0] * 4 or layout["output_bytes"] != shape[0] * 2
                or layout["stage_count"] != 1 or layout["sole_stage"]["type"] != 100):
            raise ValueError("Native Gather blob does not match the release geometry")


def pack_bundle(directory):
    manifest = (directory / "model.json").read_bytes()
    blobs = {stick: (directory / filename).read_bytes()
             for stick, (filename, _) in TARGETS.items()}
    _validate(json.loads(manifest), blobs)
    fields = (manifest, blobs["ncs1"], blobs["ncs2"])
    return zlib.compress(HEADER.pack(MAGIC, *(len(field) for field in fields)) + b"".join(fields), 9)


def read_release(path):
    try:
        data = zlib.decompress(path.read_bytes())
    except zlib.error as exc:
        raise ValueError("Unreadable MagicEightBall release") from exc
    if len(data) < HEADER.size:
        raise ValueError("Missing MagicEightBall release header")
    magic, manifest_size, ncs1_size, ncs2_size = HEADER.unpack_from(data)
    boundary = HEADER.size + manifest_size
    split = boundary + ncs1_size
    if magic != MAGIC or split + ncs2_size != len(data):
        raise ValueError("Invalid MagicEightBall release layout")
    manifest = json.loads(data[HEADER.size:boundary])
    blobs = {"ncs1": data[boundary:split], "ncs2": data[split:]}
    _validate(manifest, blobs)
    return manifest, blobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive = pack_bundle(args.bundle)
    args.output.write_bytes(archive)
    print("%d bytes" % len(archive))
