"""Lossless storage for the pinned Needle 3 SentencePiece model."""
import argparse
from collections import Counter
import hashlib
import lzma
from pathlib import Path
import struct


TOKENIZER_HASH = "97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25"
LEGACY_MAGIC = b"M8T1"
MAGIC = b"M8T2"
HEADER = struct.Struct("<4s3H")
SPECIAL_PIECES = 284
MAX_SHORT_PIECE = 120
METASPACE = "\u2581".encode("utf8")
HAS_METASPACE = 128
LENGTH_TABLE = bytes(range(HAS_METASPACE)) * 2


def _varint(number):
    encoded = bytearray()
    while number >= 128:
        encoded.append((number & 127) | 128)
        number >>= 7
    encoded.append(number)
    return bytes(encoded)


def _field(tag, contents):
    return bytes([tag]) + _varint(len(contents)) + contents


def _byte_fallback():
    # SentencePiece BYTE messages are the same except for the hex spelling.
    return b"".join(b"\x0a\x0f\x0a\x06<0x%02X>\x15\0\0\0\0\x18\x06" % value
                    for value in range(256))


def _restore(data):
    if len(data) < HEADER.size:
        raise ValueError("Missing compact tokenizer header")
    magic, prefix_size, suffix_size, count = HEADER.unpack_from(data)
    prefix_end = HEADER.size + prefix_size
    suffix_end = prefix_end + suffix_size
    lengths_end = suffix_end + count
    descriptors = data[suffix_end:lengths_end]
    if magic not in (MAGIC, LEGACY_MAGIC) or count == 0 or len(descriptors) != count:
        raise ValueError("Invalid compact tokenizer layout")
    bucketed = magic == MAGIC
    lengths = descriptors.translate(LENGTH_TABLE) if bucketed else descriptors
    if sum(lengths) != len(data) - lengths_end:
        raise ValueError("Invalid compact tokenizer layout")
    original = bytearray(data[HEADER.size:prefix_end])
    # The outer length is piece length plus the two string bytes and five score bytes.
    prefixes = [None] * 256
    for desc in set(descriptors):
        marker = METASPACE if bucketed and desc & HAS_METASPACE else b""
        length = (LENGTH_TABLE[desc] if bucketed else desc) + len(marker)
        if not 1 <= length <= MAX_SHORT_PIECE:
            raise ValueError("Invalid compact tokenizer piece length")
        prefixes[desc] = bytes([10, length + 7, 10, length]) + marker
    scores = struct.pack("<%df" % count, *(-float(rank) for rank in range(count)))
    offset = lengths_end
    score_offset = 0
    # NORMAL is the protobuf default. Rank zero has the original -0.0 bit pattern.
    if bucketed:
        original.extend(_byte_fallback())
        positions = [0] * HAS_METASPACE
        for length, amount in sorted(Counter(lengths).items()):
            positions[length] = offset
            offset += length * amount
        for desc in descriptors:
            length = desc & (HAS_METASPACE - 1)
            start = positions[length]
            end = start + length
            positions[length] = end
            original.extend(prefixes[desc] + data[start:end] + b"\x15" + scores[score_offset:score_offset + 4])
            score_offset += 4
    else:
        for length in lengths:
            end = offset + length
            original.extend(prefixes[length] + data[offset:end] + b"\x15" + scores[score_offset:score_offset + 4])
            offset = end
            score_offset += 4
    original.extend(data[prefix_end:suffix_end])
    return bytes(original)


def read_tokenizer(path):
    data = path.read_bytes()
    if path.suffix == ".xz":
        data = lzma.decompress(data)
    if data.startswith((MAGIC, LEGACY_MAGIC)):
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
    fallback = _byte_fallback()
    if not prefix.endswith(fallback):
        raise ValueError("Unexpected Needle 3 byte fallback ordering")
    prefix = prefix[:-len(fallback)]
    words = []
    descriptors = bytearray()
    for piece in model.pieces[SPECIAL_PIECES:]:
        word = piece.piece.encode("utf8")
        has_metaspace = word.startswith(METASPACE)
        if has_metaspace:
            word = word[len(METASPACE):]
        words.append(word)
        descriptors.append(len(word) | (HAS_METASPACE if has_metaspace else 0))
    del model.pieces[:]
    suffix = model.SerializeToString(deterministic=True)
    header = HEADER.pack(MAGIC, len(prefix), len(suffix), len(words))
    # Stable grouping by length keeps each bucket in vocabulary rank order.
    packed = header + prefix + suffix + descriptors + b"".join(sorted(words, key=len))
    if _restore(packed) != original:
        raise ValueError("Compact format failed to reproduce the exact SentencePiece bytes")
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 65536, "pb": 0}]
    return lzma.compress(packed, filters=filters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = read_tokenizer(args.source)
    compressed = pack_tokenizer(original)
    args.output.write_bytes(compressed)
    print("%d original bytes, %d compact bytes" % (len(original), len(compressed)))
