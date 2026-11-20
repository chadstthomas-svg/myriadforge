"""Train on original randomized Windows Forms fixtures and export a static FP32 ONNX."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from forms import GRID, ROLES, hit_test, pixels


def train(cache, output):
    torch.manual_seed(37)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    directory = cache / "forms"
    records = json.loads((directory / "forms.json").read_text(encoding="utf-8-sig"))
    inputs, boxes = [], []
    for item in records:
        with Image.open(directory / item["file"]) as image:
            inputs.append(pixels(image))
            boxes.append(np.array(item["boxes"]) / np.tile(image.size, 2))
    boxes = np.asarray(boxes, dtype="float32")
    x = torch.from_numpy(np.stack(inputs)).to(device)
    gy, gx = np.mgrid[:GRID[0], :GRID[1]].astype("float32")
    coord = np.stack(((gx + 0.5) / GRID[1], (gy + 0.5) / GRID[0]), axis=-1)
    lower = boxes[:, :, None, None, :2]
    upper = lower + boxes[:, :, None, None, 2:]
    masks = torch.from_numpy(((coord >= lower) & (coord < upper)).all(-1)).float().to(device)
    order = np.random.default_rng(37).permutation(len(records))
    test, valid, training = np.split(order, [len(order)//8, len(order)//4])
    net = nn.Sequential(nn.Conv2d(1, 12, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 4, 1)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002)
    positive_weight = torch.tensor([15, 15, 55, 55], device=device).view(1, 4, 1, 1)
    best = -1
    for epoch in range(28):
        for group in torch.randperm(len(training)).split(32):
            ix = training[group.numpy()]
            logits = net(x[ix])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, masks[ix], pos_weight=positive_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            outputs = torch.cat([net(x[indices]) for indices in np.array_split(valid, 3)]).cpu().numpy()
        score = np.mean([hit_test(s, b) for s, b in zip(outputs, boxes[valid])])
        if score > best:
            best, best_epoch, state = float(score), epoch + 1, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    net.cpu().eval()
    x = x[test].cpu()
    with torch.no_grad():
        scores = torch.cat([net(t) for t in x.split(16)]).numpy()
    hits = np.array([hit_test(s, b) for s, b in zip(scores, boxes[test])])
    out = output / "forms"
    out.mkdir(exist_ok=True, parents=True)
    model = out / "model.onnx"
    torch.onnx.export(net, x[:1], model, input_names=["features"], output_names=["scores"],
                      opset_version=9, dynamo=False)
    info = dict(model="forms", labels=ROLES, input_shape=list(x[:1].shape),
                output_shape=list(scores[:1].shape), parameters=sum(p.numel() for p in net.parameters()),
                seed=37, selected_epoch=best_epoch, validation_point_accuracy=best,
                test_point_accuracy=float(hits.mean()), per_role_point_accuracy=dict(zip(ROLES,hits.mean(0))),
                split_counts={"train":len(training),"valid":len(valid),"test":len(test)},
                onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    np.savez(cache / "forms-test.npz", x=x.numpy(), scores=scores, boxes=boxes[test],
             ids=np.array([records[i]["file"] for i in test]))
    print(json.dumps(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "bundles")
    args = p.parse_args()
    train(args.cache, args.output)
