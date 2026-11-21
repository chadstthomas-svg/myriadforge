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
GATHER_WORDS = struct.Struct("<32I")


def _gather_coefficient_bytes(data, fields):
    """Verify the stock rank-one Gather stage before trusting its table extent."""
    if fields[2:7] != (5, 0, 1, 1, 1) or fields[16:] != (192, 256, 320, 448):
        raise ValueError("Compact terminal storage requires a canonical rank-one Gather")
    words = GATHER_WORDS.unpack_from(data, fields[18])
    slots, remainder = divmod(fields[7], 4)
    # The three descriptors are FP16 table, FP16 output and signed I32 index.
    # Their rank, element stride, location, port and offset come from data.cpp.
    table = (0, 3, 1, words[8], 2, 3, 0)
    output = (0, 3, 1, slots, 2, 2, 0, slots * 2, 0)
    indices = (2, 3, 1, slots, 4, 1, 0, slots * 4, 0)
    if (not slots or remainder or fields[8] != slots * 2 or not words[8]
            or words[:2] != (128, 100) or words[3:5] != (8, 0)
            or words[5:12] != table or words[12:21] != output or words[21:30] != indices
            or words[30:] != (100, 2139356953)):
        raise ValueError("Unexpected rank-one Gather descriptors")
    return words[8] * 2


def inspect_blob(data, compact_coefficients=None):
    if not data.startswith(b"\x7felf") or len(data) < ELF_HEADER_SIZE + HEADER.size:
        raise ValueError("Expected an OpenVINO 2020.3 MYRIAD blob")
    fields = HEADER.unpack_from(data, ELF_HEADER_SIZE)
    magic, size, major, minor = fields[:4]
    boundaries = (0, *fields[16:], size)
    if (magic != 9709 or major != 5 or size != len(data)
            or list(boundaries) != sorted(boundaries) or any(n % 64 for n in boundaries[:-1])):
        raise ValueError("Invalid legacy MYRIAD header or section layout")
    layout = {name: end - start for name, start, end in zip(SECTION_NAMES, boundaries, boundaries[1:])}
    if size % 64:
        if type(compact_coefficients) is not int or compact_coefficients != layout["constants"]:
            raise ValueError("Invalid compact Gather constant extent")
        if _gather_coefficient_bytes(data, fields) != compact_coefficients:
            raise ValueError("Compact Gather stage and manifest disagree on the constant extent")
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


def compact_gather_blob(data, coefficient_bytes):
    """Drop only the terminal zero padding after a stock Gather's FP16 table."""
    inspect_blob(data)
    fields = HEADER.unpack_from(data, ELF_HEADER_SIZE)
    if type(coefficient_bytes) is not int or _gather_coefficient_bytes(data, fields) != coefficient_bytes:
        raise ValueError("Gather stage and manifest disagree on the constant extent")
    end = fields[19] + coefficient_bytes
    if not 0 <= len(data) - end < 64 or any(data[end:]):
        raise ValueError("Expected zero padding in the final Gather section")
    compact = bytearray(data[:end])
    struct.pack_into("<I", compact, ELF_HEADER_SIZE + 4, end)
    return bytes(compact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blobs", nargs="+", type=Path)
    parser.add_argument("--compact-coefficients", type=int)
    args = parser.parse_args()
    print(json.dumps({str(path): inspect_blob(path.read_bytes(), args.compact_coefficients)
                      for path in args.blobs}, indent=2))
