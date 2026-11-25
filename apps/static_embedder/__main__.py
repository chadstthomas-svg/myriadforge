"""Encode one English input with the compact static model."""

from argparse import ArgumentParser
import json
from pathlib import Path

from apps.static_embedder import StaticEncoder


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("text")
    parser.add_argument("--backend", choices=("cpu", "myriad"), default="cpu")
    parser.add_argument("--shaves", type=int, default=16)
    parser.add_argument("--lexical-weights", type=Path)
    args = parser.parse_args()
    encoder = StaticEncoder(args.tokenizer, args.weights, backend=args.backend, shaves=args.shaves,
                            lexical_path=args.lexical_weights)
    vector = encoder.encode(args.text)
    print(json.dumps({"runtime": encoder.identity, "embedding": vector.tolist()}, allow_nan=False))


if __name__ == "__main__":
    main()
