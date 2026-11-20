"""Fetch and verify the training inputs for one model family."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request


SLURP = "https://raw.githubusercontent.com/pswietojanski/slurp/8eb16545762be97ace75334109d73824217311f1/dataset/slurp/"
MIND = "https://huggingface.co/datasets/osunlp/Mind2Web/resolve/17ece8eb89862368edc0cc806acee6fca5163474/data/train/"
SOURCES = {
    "voice": [("mini_speech_commands.zip", "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
               "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0")],
    "intent": [(name + ".jsonl", SLURP + name + ".jsonl", checksum) for name, checksum in [
        ("train", "4a05a3de6c7f028aff298c4a90196e6a3fecd1938d96fb1c125d6fd10d5e301e"),
        ("devel", "00d346b183ecc2d160853cd3dc20e8ccd207d9e1809cf6e2c05182076bd0ecc9"),
        ("test", "fe3449af69b42fda7163345482556066c35a80f7e552a6d67e212a0e2f0783cc")]],
    "grounding": [(f"mind2web-{i}.json", MIND + f"train_{i}.json", checksum) for i, checksum in [
        (0, "c8b622901057bca813a6d171733c41e4fc266c2902a23d63b9094c0add3f8f2c"),
        (1, "41084922c50174ec032185707ae43bb04d3e8d725788b2e95936db629ba59100"),
        (10, "182542d7947b3fa9e90fc57a3d82d4d8f2997ca5a06664217720d7a78a956e33")]],
}


def matches(path, checksum):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest() == checksum


def fetch(cache, family):
    cache.mkdir(exist_ok=True, parents=True)
    for name, url, checksum in SOURCES[family]:
        target = cache / name
        if target.exists() and matches(target, checksum):
            continue
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as dst:
                shutil.copyfileobj(response, dst)
            if not matches(partial, checksum):
                raise ValueError("Wrong SHA-256 for " + name)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=sorted(SOURCES))
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    fetch(args.cache, args.family)
