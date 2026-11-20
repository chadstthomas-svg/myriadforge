"""Train and export fixed-shape command and intent models for the MA2450."""
import argparse
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from features import audio_features, text_features


def voice_data(cache):
    parts = {key: [] for key in ("train", "valid", "test")}
    with zipfile.ZipFile(cache / "mini_speech_commands.zip") as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("mini_speech_commands/") or not name.endswith(".wav"):
                continue
            speaker = Path(name).name.split("_nohash_")[0]
            bucket = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 10
            split = "test" if bucket == 0 else "valid" if bucket == 1 else "train"
            samples, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32")
            parts[split].append((audio_features(samples, rate), Path(name).parent.name, name))
    return parts


def intent_data(cache):
    parts = {}
    seen = set()
    for split, stem in [("train", "train"), ("valid", "devel"), ("test", "test")]:
        rows = []
        with (cache / (stem + ".jsonl")).open(encoding="utf8") as src:
            for line in src:
                record = json.loads(line)
                sentence = record["sentence"]
                key = " ".join(sentence.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                rows.append((text_features(sentence), record["intent"], str(record["slurp_id"])))
        parts[split] = rows
    return parts


def classifier(kind, classes):
    if kind == "intent":
        return nn.Sequential(nn.Linear(4096, 192), nn.ReLU(), nn.Dropout(0.25),
                             nn.Linear(192, classes))
    blocks = []
    for inputs, outputs in [(1, 16), (16, 32), (32, 48)]:
        blocks.extend([nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                       nn.BatchNorm2d(outputs), nn.ReLU(), nn.MaxPool2d(2)])
    return nn.Sequential(*blocks, nn.Flatten(), nn.Linear(48 * 5 * 12, 96),
                         nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, classes))


def train(args):
    torch.manual_seed(132)
    torch.set_num_threads(4)
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    parts = (voice_data if args.kind == "voice" else intent_data)(args.cache)
    labels = sorted({label for _, label, _ in parts["train"]})
    data = {}
    for split, rows in parts.items():
        x = torch.from_numpy(np.stack([r[0] for r in rows])).to(run_device)
        y = torch.tensor([labels.index(r[1]) if r[1] in labels else -1 for r in rows], device=run_device)
        data[split] = (x, y)
    model = classifier(args.kind, len(labels)).to(run_device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015)
    train_x, train_y = data["train"]
    best_accuracy = -1
    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(train_x), device=run_device).split(128):
            x = train_x[batch]
            if args.kind == "voice":
                # Move the whole spectrogram slightly to vary the onset time.
                shift = int(torch.randint(-4, 5, ()).item())
                x = torch.roll(x, shift, -1)
                if shift > 0:
                    x[..., :shift] = x.amin()
                elif shift < 0:
                    x[..., shift:] = x.amin()
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), train_y[batch])
            loss.backward()
            opt.step()
        model.eval()
        val_x, val_y = data["valid"]
        with torch.no_grad():
            pred = torch.cat([model(t).argmax(1) for t in val_x.split(128)])
            accuracy = (pred == val_y).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch + 1
            weights = copy.deepcopy(model.state_dict())
        print("epoch", epoch + 1, "validation", round(accuracy, 4), flush=True)
    model.load_state_dict(weights)
    model.cpu().eval()
    x, y = [t.cpu() for t in data["test"]]
    with torch.no_grad():
        scores = torch.cat([model(t) for t in x.split(128)]).numpy()
    out = args.output / args.kind
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, x[:1], out / "model.onnx", opset_version=9,
                      input_names=["features"], output_names=["scores"], dynamo=False)
    report = dict(model=args.kind, labels=labels, input_shape=list(x[:1].shape),
                  seed=132, validation_accuracy=best_accuracy, selected_epoch=best_epoch,
                  test_accuracy=float((scores.argmax(1) == y.numpy()).mean()),
                  split_counts={k: len(v) for k, v in parts.items()},
                  test_labels_missing_from_train=int((y < 0).sum()),
                  parameters=sum(t.numel() for t in model.parameters()),
                  onnx_sha256=hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest())
    (out / "model.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    args.cache.mkdir(exist_ok=True, parents=True)
    np.savez(args.cache / (args.kind + "-test.npz"), x=x.numpy(), y=y.numpy(),
             scores=scores, ids=np.array([r[2] for r in parts["test"]]))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["voice", "intent"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "bundles")
    parser.add_argument("--epochs", type=int, default=24)
    train(parser.parse_args())
