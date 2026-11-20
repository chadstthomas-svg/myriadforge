"""Assemble a local Hugging Face repository tree for MagicEightBall."""

import argparse
import hashlib
from pathlib import Path
from shutil import copyfile


CARD = Path(__file__).resolve().parent
LAB = CARD.parent
REPO = LAB.parents[2]
ARTIFACTS = (
    "runtime.py", "policy.py", "release.py", "radix.py", "blob_layout.py", "compact_tokenizer.py", "stream.py",
    "support.json", "COMPETITORS.md", "bundles/magiceightball.m8b", "needle3-tokenizer.m8t.xz",
    "measurements/release-vector-ncs1.json", "measurements/release-stream-ncs1.json",
    "measurements/ncs1-cpu-comparison.json", "measurements/two-banks-ncs1.json",
    "measurements/constant-free-pooling.json", "measurements/native-release-and-stock-controls.json",
    "measurements/streamed-parameters-compatibility.json",
)
CARD_FILES = ("README.md", "assets/workbench.webp", "assets/relay-decisions.webp", "assets/relay-decisions-still.webp")
NOTICE = """MagicEightBall
MyriadForge source and policy are distributed under the MIT License.

The Needle 3 tokenizer is distributed under the Apache License, Version 2.0.
Source: Cactus Compute, Cactus-Compute/needle3
Revision: d6c7d6a2e9655505e0f8e23339356dd1705900eb
URL: https://huggingface.co/Cactus-Compute/needle3/blob/d6c7d6a2e9655505e0f8e23339356dd1705900eb/tokenizer/tokenizer.model
Source SHA-256: 97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25
The tokenizer is repackaged losslessly. The loader restores and verifies the source bytes.
"""


def prepare(output, extra_reports):
    sources = {name: LAB / name for name in ARTIFACTS}
    sources.update({name: CARD / name for name in CARD_FILES})
    for report in extra_reports:
        key = f"measurements/{report.name}"
        if key in sources:
            raise ValueError(f"Duplicate report name: {report.name}")
        sources[key] = report
    mit = (REPO / "LICENSE").read_text(encoding="utf-8-sig")
    apache = (LAB.parent / "bundles/needle-ui/LICENSE").read_text(encoding="utf-8-sig")
    absent = [str(source) for source in sources.values() if not source.is_file()]
    if absent:
        raise FileNotFoundError("Missing upload files:\n" + "\n".join(absent))

    output.mkdir(parents=True, exist_ok=False)
    for name, source in sorted(sources.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)
    (output / "LICENSE").write_text(
        f"MyriadForge source and policy\n\n{mit.rstrip()}\n\nNeedle 3 tokenizer\n\n{apache.rstrip()}\n",
        encoding="utf-8",
    )
    (output / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}"
                 for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, action="append", default=[])
    options = parser.parse_args()
    prepare(options.output, options.report)
    print(f"Assembled model card and artifacts at {options.output.resolve()}")


if __name__ == "__main__":
    main()
