"""Lossless storage for the pinned Needle 3 SentencePiece model."""
import argparse
import hashlib
import lzma
from pathlib import Path
import struct


TOKENIZER_HASH = "97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25"
MAGIC = b"M8T1"
HEADER = struct.Struct("<4s3H")
SPECIAL_PIECES = 284
MAX_SHORT_PIECE = 120


def _varint(number):
    encoded = bytearray()
    while number >= 128:
        encoded.append((number & 127) | 128)
        number >>= 7
    encoded.append(number)
    return bytes(encoded)


def _field(tag, contents):
    return bytes([tag]) + _varint(len(contents)) + contents


def _restore(data):
    if len(data) < HEADER.size:
        raise ValueError("Missing compact tokenizer header")
    magic, prefix_size, suffix_size, count = HEADER.unpack_from(data)
    prefix_end = HEADER.size + prefix_size
    suffix_end = prefix_end + suffix_size
    lengths_end = suffix_end + count
    lengths = data[suffix_end:lengths_end]
    if (magic != MAGIC or count == 0 or len(lengths) != count
            or min(lengths) == 0 or max(lengths) > MAX_SHORT_PIECE
            or sum(lengths) != len(data) - lengths_end):
        raise ValueError("Invalid compact tokenizer layout")
    original = bytearray(data[HEADER.size:prefix_end])
    # The outer length is piece length plus the two string bytes and five score bytes.
    prefixes = [bytes([10, length + 7, 10, length]) for length in range(max(lengths) + 1)]
    scores = struct.pack("<%df" % count, *(-float(rank) for rank in range(count)))
    offset = lengths_end
    for rank_offset, length in enumerate(lengths):
        piece = data[offset:offset + length]
        offset += length
        # NORMAL is the protobuf default. Rank zero has the original -0.0 bit pattern.
        start = rank_offset * 4
        original.extend(prefixes[length] + piece + b"\x15" + scores[start:start + 4])
    original.extend(data[prefix_end:suffix_end])
    return bytes(original)


def read_tokenizer(path):
    data = path.read_bytes()
    if path.suffix == ".xz":
        data = lzma.decompress(data)
    if data.startswith(MAGIC):
        data = _restore(data)
    if hashlib.sha256(data).hexdigest() != TOKENIZER_HASH:
        raise ValueError("Unexpected Needle 3 tokenizer hash")
    return data


def pack_tokenizer(original):
    # Only the build tool uses protobuf. Runtime reconstruction is stdlib-only.
    from sentencepiece import sentencepiece_model_pb2

    if hashlib.sha256(original).hexdigest() != TOKENIZER_HASH:
        raise ValueError("Expected the pinned Needle 3 tokenizer")
    model = sentencepiece_model_pb2.ModelProto.FromString(original)
    prefix = b"".join(_field(10, piece.SerializeToString(deterministic=True))
                      for piece in model.pieces[:SPECIAL_PIECES])
    words = [piece.piece.encode("utf8") for piece in model.pieces[SPECIAL_PIECES:]]
    del model.pieces[:]
    suffix = model.SerializeToString(deterministic=True)
    header = HEADER.pack(MAGIC, len(prefix), len(suffix), len(words))
    packed = header + prefix + suffix + bytes(map(len, words)) + b"".join(words)
    if _restore(packed) != original:
        raise ValueError("Compact format failed to reproduce the exact SentencePiece bytes")
    return lzma.compress(packed, preset=9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = read_tokenizer(args.source)
    compressed = pack_tokenizer(original)
    args.output.write_bytes(compressed)
    print("%d original bytes, %d compact bytes" % (len(original), len(compressed)))
