"""Train a compact browser control ranker from recorded Mind2Web demonstrations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from lxml import html
import torch
from torch import nn

from grounding import COUNT, WIDTH, encode_screen


def controls_for(action):
    all_candidates = action["pos_candidates"] + action["neg_candidates"]
    by_id = {str(c["backend_node_id"]): c for c in all_candidates}
    controls = []
    if not action["cleaned_html"].strip():
        return controls
    root = html.fromstring(action["cleaned_html"])
    for node in root.iter():
        key = node.get("backend_node_id")
        c = by_id.pop(key, None)
        if c is None:
            continue
        att = json.loads(c["attributes"])
        tag = str(c["tag"]).lower()
        role = att.get("role") or {
            "input": "textbox", "textarea": "textbox", "a": "link", "select": "combobox",
            "li": "listitem"}.get(tag, tag)
        if tag == "input" and att.get("type") in ("radio", "checkbox", "button", "submit"):
            role = "button" if att["type"] == "submit" else att["type"]
        name = " ".join(str(att.get(k) or "") for k in
                        ("aria_label", "aria-label", "placeholder", "title", "alt"))
        name += " " + " ".join(" ".join(node.itertext()).split())[:140]
        controls.append(dict(id=key, role=role, name=name, editable=role == "textbox",
                             clickable=att.get("is_clickable") == "true"))
    return controls


def prepare(cache):
    rows = []
    for shard in (0, 1, 10):
        with (cache / ("mind2web-%s.json" % shard)).open(encoding="utf8") as file:
            tasks = json.load(file)
        for task in tasks:
            for index, action in enumerate(task["actions"]):
                query = task["confirmed_task"]
                history = " ".join(task["action_reprs"][max(0, index-2):index])
                x, ids = encode_screen(query, controls_for(action), history)
                target_ids = {str(c["backend_node_id"]) for c in action["pos_candidates"]}
                good = np.zeros(COUNT, dtype=bool)
                good[:len(ids)] = [key in target_ids for key in ids]
                rows.append((x, good, len(ids), action["action_uid"], task["website"]))
        del tasks
    return rows


def train(cache, output):
    torch.manual_seed(89)
    torch.set_num_threads(4)
    rows = prepare(cache)
    sites = sorted({r[-1] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(sites) < 6:
        raise ValueError("Too few sites to evaluate a site-held-out split")
    split_sites = {"test": sites[::6], "valid": sites[1::6]}
    split_sites["train"] = sorted(set(sites) - set(split_sites["test"]) - set(split_sites["valid"]))
    data, identifiers = {}, {}
    for key, domain_list in split_sites.items():
        subset = [r for r in rows if r[-1] in domain_list]
        data[key] = (torch.from_numpy(np.stack([r[0] for r in subset])),
                     torch.from_numpy(np.stack([r[1] for r in subset])),
                     torch.tensor([r[2] for r in subset]))
        identifiers[key] = [r[3] for r in subset]
    net = nn.Sequential(nn.Conv2d(WIDTH, 64, 1), nn.ReLU(), nn.Conv2d(64, 16, 1),
                        nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Flatten())
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    positions = torch.arange(COUNT)[None]

    def evaluate(split):
        x, good, counts = data[split]
        with torch.no_grad():
            logits = net(x)
            chosen = logits.masked_fill(positions >= counts[:, None], -1e6).argmax(1)
            return float(good.gather(1, chosen[:, None]).float().mean()), logits

    x, good, counts = data["train"]
    trainable = torch.where(good.any(1))[0]
    best = -1
    for epoch in range(35):
        for ix in trainable[torch.randperm(len(trainable))].split(64):
            scores = net(x[ix]).masked_fill(positions >= counts[ix, None], -1e6)
            loss = (scores.logsumexp(1) - scores.masked_fill(~good[ix], -1e6).logsumexp(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        val, _ = evaluate("valid")
        if val > best:
            best, best_epoch, weights = val, epoch + 1, copy.deepcopy(net.state_dict())
    net.load_state_dict(weights)
    net.eval()
    accuracy, reference = evaluate("test")
    x, good, counts = data["test"]
    out = output / "grounding"
    out.mkdir(exist_ok=True, parents=True)
    model = out / "model.onnx"
    torch.onnx.export(net, x[:1], model, input_names=["features"], output_names=["scores"],
                      opset_version=9, dynamo=False)
    info = dict(model="grounding", labels=["candidate_%d" % i for i in range(COUNT)],
                input_shape=list(x[:1].shape), parameters=sum(p.numel() for p in net.parameters()),
                seed=89, selected_epoch=best_epoch, validation_accuracy=best, test_accuracy=accuracy,
                lexical_top1=float(good[:, 0].float().mean()),
                shortlist_recall=float(good.any(1).float().mean()),
                trainable_steps=len(trainable), split_counts={s: len(d[0]) for s, d in data.items()},
                sites=split_sites, onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    np.savez(cache / "grounding-test.npz", x=x.numpy(), good=good.numpy(), counts=counts.numpy(),
             y=np.where(good.any(1), good.numpy().argmax(1), -1), ids=np.array(identifiers["test"]),
             scores=reference.numpy())
    print(json.dumps(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "bundles")
    args = p.parse_args()
    train(args.cache, args.output)
