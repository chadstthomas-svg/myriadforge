"""Build the vision-gate v0 fixtures (vision-gate/1): trigger vs non-trigger.

Trigger class = 500 COCO val2017 natural photos (official zip, SHA-verifiable;
dataset images are CC-licensed Flickr photography - provenance recorded here
and in the manifest). Non-trigger class = 260 Big Buck Bunny cartoon frames
(Blender Foundation, CC BY) extracted at 24 fps - the exact content the
measured cascade failed on (the ImageNet gate fired 59-60/60 on it).

Splits: COCO random 400/100; BBB TIME-based (first 180 train, last 80
held-out) so held-out frames come from a different part of the video - the
honest leak check for temporally-correlated footage.

Usage (main env):
    python build_vision_gate.py --coco D:/models/vision-gate/coco \
        --bbb D:/NTVStudy/data/samples/frames24 --out D:/models/vision-gate
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", type=Path, default=Path("D:/models/vision-gate/coco"))
    ap.add_argument("--bbb", type=Path, default=Path("D:/NTVStudy/data/samples/frames24"))
    ap.add_argument("--out", type=Path, default=Path("D:/models/vision-gate"))
    ap.add_argument("--coco-train", type=int, default=400)
    args = ap.parse_args()

    rng = random.Random(132)
    rows = []

    coco = sorted((args.coco / "val2017").glob("*.jpg"))
    picked = rng.sample(coco, min(args.coco_train + 100, len(coco)))
    for i, path in enumerate(picked):
        rows.append({"id": f"coco_{path.stem}", "wav": str(path), "label": "trigger",
                     "split": "train" if i < args.coco_train else "test",
                     "kind": "coco_natural_photo", "source":
                     "COCO val2017 (images.cocodataset.org zips/val2017.zip)"})

    bbb = sorted(args.bbb.glob("frame_*.png"))
    n_train_bbb = int(len(bbb) * 0.7)
    for i, path in enumerate(bbb):
        rows.append({"id": f"bbb_{path.stem}", "wav": str(path), "label": "non-trigger",
                     "split": "train" if i < n_train_bbb else "test",
                     "kind": "bbb_cartoon_frame", "source": "Big Buck Bunny (CC BY, Blender Foundation)"})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    counts = {}
    for r in rows:
        counts[(r["label"], r["split"])] = counts.get((r["label"], r["split"]), 0) + 1
    print(f"vision-gate fixtures: {len(rows)} rows -> {args.out / 'manifest.jsonl'}")
    for (label, split), n in sorted(counts.items()):
        print(f"  {label:12s} {split:5s} {n}")


if __name__ == "__main__":
    main()
