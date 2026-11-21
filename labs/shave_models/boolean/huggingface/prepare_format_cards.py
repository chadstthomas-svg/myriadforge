"""Add Hugging Face cards and shared original art to verified portable exports."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile

from prepare_upload import CARD, LAB, NOTICE, REPO


FAMILY = "https://huggingface.co/tablinum/MagicEightBall"
FORMATS = {
    "onnx": ("ONNX", "model.onnx", "onnxruntime",
             "Portable radix Gather export. The graph accepts 128 packed INT32 addresses and returns 128 FP32 "
             "values that exactly represent the packed lookup words. The included runner builds the addresses "
             "from the document evidence and decodes the raw decision bits."),
    "fp32": ("FP32 Base", "model.onnx", "onnxruntime",
             "Full-precision dense Boolean reference. The graph reads seven FP32 facts and returns seven FP32 "
             "Boolean outputs. The weights were compiled from the support rules and use FP32 throughout."),
    "gguf": ("GGUF", "model.gguf", "gguf",
             "GGUF container with a custom `magiceightball` architecture. Its I8 decision table stores one "
             "packed byte for each of the 128 fact assignments. The included CPU reader evaluates it. "
             "Use that reader for this architecture. It has no `llama.cpp` loader."),
}


def prepare(format_name, source, output):
    format_title, model, dependency, summary = FORMATS[format_name]
    model_data = (source / model).read_bytes()
    size = len(model_data)
    proof = json.loads((source / "verification.json").read_bytes())
    if (proof["artifact_sha256"] != hashlib.sha256(model_data).hexdigest()
            or proof["fact_assignments"] != 128 or proof["short_documents"] != 128
            or [item["tokens"] for item in proof["context_cases"]] != [131072, 131072]
            or proof["first_rejected_length"] != 131073):
        raise ValueError("The portable export does not match its verified workload")
    for name in ("portable_runtime.py", "needle3-tokenizer.m8t.xz"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in ("README.md", "LICENSE", "NOTICE.txt", "SHA256SUMS.txt"):
            copyfile(path, output / path.name)
    assets = output / "assets"
    assets.mkdir()
    poster = "workbench.webp"
    motion = "workbench-motion.webp"
    for name in (poster, motion, "size-portable-cpu.webp"):
        candidate = CARD / "assets" / name
        if candidate.is_file():
            copyfile(candidate, assets / name)
    hero = motion if (assets / motion).is_file() else poster
    still_link = " · [Still banner](./assets/workbench.webp)" if hero == motion else ""
    size_chart = ""
    if (assets / "size-portable-cpu.webp").is_file():
        size_chart = ("## File size comparison\n\n"
                      "![Portable CPU model files. ONNX 192 bytes, GGUF 1,024, FP32 dense 1,832.]"
                      "(./assets/size-portable-cpu.webp)\n\n"
                      f"[Native and published size comparisons]({FAMILY}#size-comparisons)\n\n")
    neighbors = " · ".join(f"[{FORMATS[key][0]}]({FAMILY}-{suffix})"
                           for key, suffix in (("onnx", "ONNX"), ("fp32", "FP32"), ("gguf", "GGUF"))
                           if key != format_name)
    card = f'''---
language:
  - en
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE
tags:
  - magiceightball
  - decision-model
  - rule-engine
  - {format_name}
---

# MagicEightBall {format_title}

![Eight, a black billiards 8-ball, and Relay, a floating chip, light routes across their workbench.](./assets/{hero})

MagicEightBall evaluates a fixed support policy and returns a Boolean answer, a category, and a proposed action. The included CPU runner accepts up to **131,072 Needle 3 tokens**. The [{format_title} model](./{model}) is **{size:,} bytes**. The shared tokenizer, runner, and artwork have separate sizes.

{summary}

CPU verification passed all **128 fact assignments**, **128 short text fixtures**, two exact-limit documents, and rejection at 131,073 tokens. [Verification report](./verification.json)

## Try the policy

Download the files together and run the provided CLI from that directory.

```bash
python -m pip install numpy sentencepiece {dependency}
python portable_runtime.py {model} --text "Duplicate charge. Review authorized."
```

Expect `may_route=true`, `category=billing`, and `action=review_billing`. Try appending `Do not route.` to the same input. The answer becomes false and the action becomes empty. The category remains `billing`. An unrelated request has no category or action.

The model finds the policy's declared phrases. The caller receives the proposed action as data for review. [Measured CPU and NCS1 runtimes]({FAMILY}#runtime-comparison) For physical Intel NCS1 execution and industry starter packs, use the native model page. [Native NCS1 and industry packs]({FAMILY}) · {neighbors}{still_link}

{size_chart}## License

The decision rules and runner are MIT-licensed. The losslessly repackaged Cactus Compute Needle 3 tokenizer is Apache-2.0. [License](./LICENSE) · [Tokenizer provenance](./NOTICE.txt) · [Checksums](./SHA256SUMS.txt)
'''
    (output / "README.md").write_text(card, encoding="utf8")
    mit = (REPO / "LICENSE").read_text(encoding="utf-8-sig")
    apache = (LAB.parent / "bundles/needle-ui/LICENSE").read_text(encoding="utf-8-sig")
    (output / "LICENSE").write_text(mit.rstrip() + "\n\nNeedle 3 tokenizer\n\n" + apache.rstrip() + "\n", encoding="utf8")
    (output / "NOTICE.txt").write_text(NOTICE, encoding="utf8")
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}"
                 for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=FORMATS)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.format, args.source, args.output)
