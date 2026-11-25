"""Train the feature-only, single-pass, tied and independent contextual adapters."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from apps.static_embedder import StaticEncoder, fit_lexical_weights, lexical_counts
from apps.static_embedder.contextual import (CONTEXT_WEIGHT, EPSILON, SCHEMA, VARIANTS, WIDTH,
                                              adapter_shapes, grouped_features)
from tools.ncs2.prepare_contextual_embedder import SEED, pair_prompts


class Adapter(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        arrays = {}
        for name, shape in adapter_shapes(variant).items():
            if name == "bias":
                array = torch.tensor([-1.2, .4], dtype=torch.float32)[:shape[0]]
            elif name == "position":
                array = torch.randn(shape) * .15
            else:
                scale = {"token": 4, "gate": .25}.get(name.rstrip("01"), 1)
                array = torch.randn(shape) * (scale / np.sqrt(shape[0]))
            arrays[name] = nn.Parameter(array)
        self.weights = nn.ParameterDict(arrays)

    def forward(self, features, mask):
        w = self.weights
        state = features @ w["token"] + w["position"]
        for step, block in enumerate(VARIANTS[self.variant]):
            x = F.layer_norm(state, (WIDTH,), eps=EPSILON)
            scores = (x @ w[f"q{block}"]) @ (x @ w[f"k{block}"]).transpose(-1, -2)
            scores = scores / np.sqrt(WIDTH) + (mask.transpose(-1, -2) - 1) * 64
            attn = scores.softmax(dim=-1)
            gate = torch.sigmoid(x @ w[f"gate{block}"] + w["bias"][step])
            state = state + gate * (attn @ (x @ w[f"v{block}"]) @ w[f"o{block}"])
            x = F.layer_norm(state, (WIDTH,), eps=EPSILON)
            state = state + gate * ((x @ w[f"up{block}"]).relu() @ w[f"down{block}"])
        return F.normalize((state * mask).sum(dim=-2), dim=-1, eps=1e-4)

    def save(self, path):
        with path.open("wb") as stream:
            np.savez(stream, schema=SCHEMA, variant=self.variant,
                     **{key: param.detach().cpu().numpy() for key, param in self.weights.items()})


def pack(pairs, encoder):
    trio = [(row["query"], row["positive"], row["negative"]) for row in pairs]
    texts = list(dict.fromkeys(text for group in trio for text in group))
    indices = {text: i for i, text in enumerate(texts)}
    features, masks, baseline = [], [], []
    inputs = encoder.tokenizer.encode_batch(texts, add_special_tokens=False)
    for text, tokens in zip(texts, inputs):
        array, mask = grouped_features(tokens.ids)
        features.append(array.astype(np.float16))
        masks.append(mask.astype(np.float16))
        baseline.append(encoder.encode_cpu(tokens.ids, lexical_counts(text)))
    ids = np.asarray([[indices[text] for text in group] for group in trio], dtype=np.int64)
    base = np.stack(baseline)
    scores = np.einsum("ij,ikj->ik", base[ids[:, 0]], base[ids[:, 1:]])
    return {"features": np.stack(features), "masks": np.stack(masks), "ids": ids, "base": scores}


def to_device(data, device):
    return {key: torch.as_tensor(value, device=device) for key, value in data.items()}


def scores(model, data, indices):
    row = data["ids"][indices]
    result = model(data["features"][row].float(), data["masks"][row].float())
    context = torch.einsum("ij,ikj->ik", result[:, 0], result[:, 1:])
    return (1 - CONTEXT_WEIGHT) * data["base"][indices] + CONTEXT_WEIGHT * context


def pair_accuracy(model, data):
    correct = []
    with torch.no_grad():
        for indices in torch.arange(len(data["ids"]), device=data["ids"].device).split(128):
            values = scores(model, data, indices)
            correct.append((values[:, 0] > values[:, 1]).float())
    return torch.cat(correct).mean().item()


def train_variant(root, variant, train, dev, industry_dev, *, general, seed, epochs):
    torch.manual_seed(seed)
    device = train["ids"].device
    model = Adapter(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator(device=device).manual_seed(seed)
    chosen_score = -1.0
    weights = root / f"context-{variant}.npz"
    history = []
    for epoch in range(epochs):
        model.train()
        general_order = torch.randperm(general, generator=generator, device=device)
        losses = []
        for primary in general_order.split(24):
            extra = torch.randint(general, len(train["ids"]), (8,), generator=generator, device=device)
            ids = torch.cat([primary, extra])
            prediction = scores(model, train, ids)
            supervised = F.cross_entropy(prediction / .05, torch.zeros(len(ids), dtype=torch.long, device=device))
            target = (train["teacher"][ids] / .07).softmax(dim=1)
            distillation = F.kl_div((prediction / .07).log_softmax(dim=1), target, reduction="batchmean")
            loss = .7 * supervised + .3 * distillation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.detach())
        model.eval()
        general_dev, rack_dev = pair_accuracy(model, dev), pair_accuracy(model, industry_dev)
        history.append({"epoch": epoch + 1, "loss": float(torch.stack(losses).mean()),
                        "squad_dev": general_dev, "industry_dev": rack_dev})
        print(variant, history[-1], flush=True)
        if general_dev + rack_dev > chosen_score:
            chosen_score = general_dev + rack_dev
            model.save(weights.with_suffix(".tmp"))
            weights.with_suffix(".tmp").replace(weights)
    return history


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("static_assets", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("At least one epoch is required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    root = args.directory
    datasets = json.loads((root / "pairs.json").read_text(encoding="utf8"))
    training = datasets["train"] + datasets["industry_train"]
    lexical = root / "training-lexical.npy"
    documents = list(dict.fromkeys(text for pair in training for text in (pair["positive"], pair["negative"])))
    np.save(lexical, fit_lexical_weights(documents), allow_pickle=False)
    encoder = StaticEncoder(args.static_assets / "tokenizer.json",
                            args.static_assets / "base1024-fp16.npy", lexical_path=lexical)
    print(f"Preparing {len(training)} train pairs and both dev splits", flush=True)
    train = pack(training, encoder)
    dev = pack(datasets["dev"], encoder)
    industry_dev = pack(datasets["industry_dev"], encoder)
    prompts = json.loads((root / "teacher-prompts.json").read_text(encoding="utf8"))
    teacher = np.load(root / "fit-teacher.npy", allow_pickle=False)
    if teacher.shape != (len(prompts), 768) or not np.isfinite(teacher).all():
        raise ValueError("The teacher cache must match the prompt manifest")
    prompt_ids = {text: i for i, text in enumerate(prompts)}
    selected = np.asarray([[prompt_ids[text] for text in pair_prompts(row)] for row in training])
    train["teacher"] = np.einsum("ij,ikj->ik", teacher[selected[:, 0]], teacher[selected[:, 1:]])
    train, dev, industry_dev = [to_device(data, args.device) for data in (train, dev, industry_dev)]
    report = {"squad_train_pairs": len(datasets["train"]), "industry_train_pairs": len(datasets["industry_train"]),
              "seed": args.seed, "epochs": args.epochs, "variants": {}}
    for variant in VARIANTS:
        report["variants"][variant] = train_variant(root, variant, train, dev, industry_dev,
                                                   general=len(datasets["train"]), seed=args.seed, epochs=args.epochs)
        (root / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
