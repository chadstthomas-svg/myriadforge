"""Read the documented section layout of an OpenVINO 2020.3 MYRIAD blob."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


# OpenVINO 2020.3.2 blob_format.hpp and backend/serialize.cpp.
ELF_HEADER_SIZE = 52
HEADER = struct.Struct("<20I")
SECTION_NAMES = ("headers", "input_descriptor", "output_descriptor", "stages", "constants")


def inspect_blob(data):
    if not data.startswith(b"\x7felf") or len(data) < ELF_HEADER_SIZE + HEADER.size:
        raise ValueError("Expected an OpenVINO 2020.3 MYRIAD blob")
    fields = HEADER.unpack_from(data, ELF_HEADER_SIZE)
    magic, size, major, minor = fields[:4]
    boundaries = (0, *fields[16:], size)
    if (magic != 9709 or major != 5 or size != len(data)
            or list(boundaries) != sorted(boundaries) or any(n % 64 for n in boundaries)):
        raise ValueError("Invalid legacy MYRIAD header or section layout")
    layout = {name: end - start for name, start, end in zip(SECTION_NAMES, boundaries, boundaries[1:])}
    stage_count = fields[6]
    stage = None
    if stage_count == 1:
        if layout["stages"] < 12:
            raise ValueError("Invalid MYRIAD stage")
        length, operation, shaves = struct.unpack_from("<3I", data, fields[18])
        if not 12 <= length <= layout["stages"]:
            raise ValueError("Invalid legacy MYRIAD stage length")
        stage = {"length": length, "type": operation, "shaves_field": shaves}
    return dict(bytes=size, sha256=hashlib.sha256(data).hexdigest(), version=[major, minor],
                inputs=fields[4], outputs=fields[5], stage_count=stage_count,
                input_bytes=fields[7], output_bytes=fields[8], sections=layout, sole_stage=stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blobs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({str(path): inspect_blob(path.read_bytes()) for path in args.blobs}, indent=2))
