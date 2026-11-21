"""Train the vision-gate suite model (vision-gate/1): trigger vs non-trigger.

Compute-dense design per the spec and the paired-gap ceiling (3x3 convs hit
35.8 GFLOPS on the stick; depthwise/1x1 are the MobileNet trap): four dense
3x3 conv stages, global average pool, single binary head. Input [1,3,224,224]
with the footage pipeline's normalization. Runs in the external training venv.

  train-venv/Scripts/python.exe train_vision_gate.py --fixtures D:/models/vision-gate
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vision_gate_spec import ACCEPTANCE, INPUT_SHAPE, NORMALIZE, TASK_SPEC_VERSION

SEED = 132


def frame_to_nchw(path: Path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - 0.5) / 0.5).transpose(2, 0, 1)


class DenseGate(nn.Module):
    """Compute-dense: four 3x3 conv stages (~181M MACs), no depthwise, no 1x1."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.head = nn.Linear(96, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = F.relu(self.c4(h))
        return self.head(h.mean(dim=(2, 3)))


def load(fixtures: Path):
    rows = [json.loads(line) for line
            in (fixtures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    X, Y, split_of = [], [], []
    for row in rows:
        X.append(frame_to_nchw(Path(row["wav"])))
        Y.append(0 if row["label"] == "non-trigger" else 1)
        split_of.append(row["split"])
    return np.stack(X), np.asarray(Y), np.asarray(split_of)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "bundles" / "vision-gate")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    print("loading frames...", flush=True)
    X, Y, split_of = load(args.fixtures)
    train_idx = np.where(split_of == "train")[0]
    test_idx = np.where(split_of == "test")[0]
    rng.shuffle(train_idx)
    print(f"frames: {len(X)} (train {len(train_idx)}, held-out {len(test_idx)})", flush=True)

    xt = torch.from_numpy(X[train_idx])
    yt = torch.from_numpy(Y[train_idx])
    xv = torch.from_numpy(X[test_idx])
    yv = torch.from_numpy(Y[test_idx])
    pos_w = torch.tensor([(Y[train_idx] == 0).sum() / max(1, (Y[train_idx] == 1).sum()), 1.0])

    net = DenseGate()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for b in range(0, len(xt), args.batch):
            sel = perm[b:b + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(net(xt[sel]), yt[sel])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        net.eval()
        with torch.no_grad():
            pred = net(xv).argmax(1)
            acc = float((pred == yv).float().mean())
            trig = yv == 1
            recall = float((pred[trig] == 1).float().mean())
            fire_non = float((pred[~trig] == 1).float().mean())
        history.append({"epoch": epoch, "train_loss": round(total / len(xt), 4),
                        "heldout_acc": round(acc, 4), "trigger_recall": round(recall, 4),
                        "fire_rate_nontrigger": round(fire_non, 4)})
        print(history[-1], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.from_numpy(X[:1]), str(args.out / "model.onnx"),
                      opset_version=9, input_names=["frames"], output_names=["logits"],
                      dynamo=False)
    info = {
        "model": "vision-gate", "spec_version": TASK_SPEC_VERSION,
        "input_shape": INPUT_SHAPE, "normalize": NORMALIZE,
        "classes": ["non-trigger", "trigger"], "seed": SEED, "epochs": args.epochs,
        "history": history, "acceptance": ACCEPTANCE,
        "onnx_sha256": hashlib.sha256((args.out / "model.onnx").read_bytes()).hexdigest(),
    }
    (args.out / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"exported {args.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
