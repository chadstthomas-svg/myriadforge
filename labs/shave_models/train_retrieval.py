"""Train the retrieval suite encoder (retrieval/1): contrastive text -> 128-d.

Training signal is the lab's own paraphrase structure: utterances with
IDENTICAL ordered tool calls are positives (template invariance), everything
else in batch is a negative (InfoNCE). Input = the intent bundle's hashed-gram
4096-d host feature (deterministic preprocessing, shared with the intent
suite), so the on-stick encoder is a single 4096->128 Gemm (~525K params,
~1 MB FP16 - well under the measured weight-fit boundary) and the kNN engine
(as-MatMul table) does the retrieval.

Baselines and gates (retrieval_spec.ACCEPTANCE): Recall@10 on held-out
slot-filled queries vs a 10k-row table, within 2 pts of the hashed-gram
baseline. Runs in the external training venv.

  train-venv/Scripts/python.exe train_retrieval.py --fixtures D:/models/ncs-shave-research/voicetools
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retrieval_spec import ACCEPTANCE, EMBED_DIM, TASK_SPEC_VERSION

SEED = 132


def text_features(sentence):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import text_features as shared
    return shared(sentence)


class Encoder(nn.Module):
    def __init__(self, in_dim=4096, out_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        z = self.proj(x)
        return F.normalize(z, dim=-1)


def load_pairs(fixtures: Path):
    """train-split utterances -> (features, call-key) for InfoNCE grouping."""
    rows = [json.loads(line) for line
            in (fixtures / "plan.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    train = [r for r in rows if r["split"] == "train" and r["calls"]]
    X = np.stack([text_features(r["text"]) for r in train]).astype(np.float32)
    keys = [json.dumps(r["calls"], sort_keys=True) for r in train]
    heldout = [r for r in rows if r["split"] != "train" and r["calls"]]
    return X, keys, train, heldout


def recall_at_k(query_z, table_z, table_keys, qkeys, k=10):
    """Mean Recall@10: fraction of the query's paraphrase group present in top-k."""
    sims = query_z @ table_z.T
    hits = []
    for i, sims_row in enumerate(sims):
        top = np.argsort(-sims_row)[:k]
        same = {j for j in top if table_keys[j] == qkeys[i]}
        group_total = sum(1 for kk in table_keys if kk == qkeys[i]) - 1
        hits.append(len(same) / max(1, group_total))
    return float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "bundles" / "retrieval")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X, keys, train_rows, heldout_rows = load_pairs(args.fixtures)
    print(f"corpus: {len(X)} utterances ({len(set(keys))} distinct calls)", flush=True)

    xt = torch.from_numpy(X)
    kt = torch.tensor([hash(k) % (2 ** 31) for k in keys])  # group id per row
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    paired = [v for v in groups.values() if len(v) > 1]
    print(f"positives available: {sum(len(v) - 1 for v in paired)} pairs in "
          f"{len(paired)} groups", flush=True)

    net = Encoder()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # group-aware batching: half of every batch is same-call pairs, so every
    # anchor has an in-batch positive and InfoNCE always has real signal
    import random as _random
    _sampler = _random.Random(SEED + 1)
    group_ids = list(groups.values())
    history = []
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        n_batches = len(xt) // args.batch
        for b in range(n_batches):
            sel_list = []
            for _ in range(args.batch // 2):
                g = group_ids[_sampler.randrange(len(group_ids))]
                sel_list.extend(_sampler.sample(g, min(2, len(g))))
            while len(sel_list) < args.batch:
                sel_list.append(rng.integers(len(xt)))
            sel = torch.tensor(sel_list[:args.batch])
            z = net(xt[sel])                                   # [B, 128] normalized
            sims = z @ z.T                                     # in-batch similarity
            same = (kt[sel][:, None] == kt[sel][None, :])
            eye = torch.eye(len(sel), dtype=torch.bool)
            pos_mask = same & ~eye
            if not pos_mask.any():
                continue
            # InfoNCE: for each anchor with >=1 positive, -log(exp(pos_sim/tau) /
            # sum exp(all_sim/tau)) averaged over positives
            logits = sims / 0.07
            logits = logits.masked_fill(eye, -1e9)
            pos_logits = logits[pos_mask]
            idx_pos = pos_mask.float().argmax(dim=1)
            has_pos = pos_mask.any(dim=1)
            loss = F.cross_entropy(logits[has_pos], idx_pos[has_pos])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        net.eval()
        with torch.no_grad():
            zall = net(xt).numpy()
            sample = rng.permutation(len(zall))[:500]
            r10 = recall_at_k(zall[sample], zall, keys, [keys[i] for i in sample])
        history.append({"epoch": epoch, "train_loss": round(total / len(xt), 4),
                        "recall10_sample": round(r10, 4)})
        print(history[-1], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    probe = torch.from_numpy(X[:1])
    torch.onnx.export(net, probe, str(args.out / "model.onnx"), opset_version=9,
                      input_names=["features"], output_names=["embedding"], dynamo=False)
    info = {
        "model": "retrieval-encoder", "spec_version": TASK_SPEC_VERSION,
        "input_shape": [1, 4096], "embed_dim": EMBED_DIM, "seed": SEED,
        "epochs": args.epochs, "corpus": len(X),
        "history": history, "acceptance": ACCEPTANCE,
        "onnx_sha256": hashlib.sha256((args.out / "model.onnx").read_bytes()).hexdigest(),
    }
    (args.out / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"exported {args.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
