"""Train the voicetools shared multi-head model (voicetools/1).

Architecture = the handoff-approved graph (Conv stem + temporal blocks with
1-D convs as 2-D kernels [1,9]) plus per-slot classification heads:
tool[13 = 12 tools + reject], room[6], device[6], scene[5], level[12],
seconds[6]. "none" indexes cover absent slots; negatives train all-none.
Honest v1 slice: ordered calls come from the per-window decision rule
(score every 2 s window at 1 s stride, take ordered unique fired tools) -
sequence/CTC heads are deferred until this slice clears the gate.

Runs in the external training venv (torch lives there, not in the repos):
  D:/models/ncs-shave-research/train-venv/Scripts/python.exe \\
      train_voicetools.py --fixtures D:/models/ncs-shave-research/voicetools --epochs 10
Outputs: bundles/voicetools/{model.json, model.onnx} + history in the report.
"""
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools_spec import (ACCEPTANCE, DURATIONS, LEVELS, SCENES, DEVICES, ROOMS,
                        TASK_SPEC_VERSION, TOOLS, WINDOW_SECONDS, WINDOW_STRIDE)

RATE = 16000
WIN_FRAMES = 200
HOP = 160
NFFT = 512
TOOLS_SORTED = sorted(TOOLS)
TOOL_IDX = {t: i for i, t in enumerate(TOOLS_SORTED)}
SLOT_ENUMS = {
    "room": sorted(ROOMS),
    "device": sorted(DEVICES),
    "scene": sorted(SCENES),
    "level": sorted(LEVELS),
    "seconds": sorted(s for s, _ in DURATIONS),
}
HEADS = {"tool": len(TOOLS_SORTED) + 1, "room": len(SLOT_ENUMS["room"]) + 1,
         "device": len(SLOT_ENUMS["device"]) + 1, "scene": len(SLOT_ENUMS["scene"]) + 1,
         "level": len(SLOT_ENUMS["level"]) + 1, "seconds": len(SLOT_ENUMS["seconds"]) + 1}
NONE_IDX = {name: count - 1 for name, count in HEADS.items()}
SEED = 132


def mel_filters():
    """Identical filterbank to features.py - shared host preprocessing."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import mel_filters as shared
    return shared()


def logmel(samples: np.ndarray) -> np.ndarray:
    """16 kHz mono samples -> [40, 200] float32 log-mel, per-window normalized."""
    need = (WIN_FRAMES - 1) * HOP + 400
    if len(samples) < need:
        samples = np.pad(samples, (0, need - len(samples)))
    offsets = np.arange(WIN_FRAMES)[:, None] * HOP + np.arange(400)
    frames = samples[offsets] * np.hanning(400)
    power = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)) ** 2
    mel = np.log(np.maximum(power @ mel_filters(), 1e-8))
    return ((mel.T - mel.mean()) / max(mel.std(), 0.1)).astype(np.float32)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as src:
        assert src.getframerate() == RATE and src.getnchannels() == 1 and src.getsampwidth() == 2
        return np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def slot_index(head, value):
    if value is None:
        return NONE_IDX[head]
    return SLOT_ENUMS[head].index(int(value) if head in ("level", "seconds") else str(value).casefold())


def window_starts(seconds):
    """The eval-contract window set: stride-aligned starts plus the end-aligned
    window (end-of-speech decision), deduped, in time order."""
    if seconds <= WINDOW_SECONDS:
        return [0.0]
    aligned = [w * WINDOW_STRIDE for w in range(int((seconds - WINDOW_SECONDS) / WINDOW_STRIDE) + 1)]
    aligned.append(max(0.0, seconds - WINDOW_SECONDS))
    return sorted(set(round(s, 3) for s in aligned))


def window_labels(row):
    """Utterance -> [(window_start, target_call_or_None), ...] for every window."""
    starts = window_starts(row["seconds"])
    calls = row["calls"]
    if not calls:
        return [(start, None) for start in starts]
    midpoint = row["seconds"] / 2.0
    out = []
    for start in starts:
        call = calls[0] if (start < midpoint or len(calls) == 1) else calls[1]
        out.append((start, call))
    return out


class HeadedNet(nn.Module):
    """arch v1: validated handoff graph, global GAP -> 32-d (collapses time).
    arch v2: third block without time pooling, GAP over the MEL axis only,
    flatten [C*T] -> heads - keeps the temporal structure slot heads need."""

    def __init__(self, arch="v1"):
        super().__init__()
        self.arch = arch
        self.stem = nn.Conv2d(1, 16, (1, 9), padding=(0, 4))
        self.block1 = nn.Conv2d(16, 24, (1, 9), padding=(0, 4))
        if arch == "v1":
            self.block2 = nn.Conv2d(24, 32, (1, 9), padding=(0, 4))
            self.heads = nn.ModuleDict({n: nn.Linear(32, c) for n, c in HEADS.items()})
        else:
            self.block2 = nn.Conv2d(24, 48, (1, 9), padding=(0, 4))
            self.heads = nn.ModuleDict({n: nn.Linear(48 * 50, c) for n, c in HEADS.items()})
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.max_pool2d(F.relu(self.stem(x)), (1, 2))
        h = F.max_pool2d(F.relu(self.block1(h)), (1, 2))
        h = F.relu(self.block2(h))
        if self.arch == "v1":
            flat = h.mean(dim=(2, 3))                      # global GAP -> [B, 32]
        else:
            flat = h.mean(dim=2).flatten(1)                # GAP over mel only -> [B, 48*50]
        return {name: head(flat) for name, head in self.heads.items()}


def extra_negative_rows(vad_dir: Path):
    """vad-suite noise/silence clips as extra all-none rows (train split only).

    Cross-suite reuse inside the lab: the vad bootstrap synthesized 1,500+
    deterministic non-speech clips the voicetools reject head never saw.
    """
    rows = []
    manifest = vad_dir / "manifest.jsonl"
    for mf in (manifest, vad_dir / "manifest-sealed.jsonl"):
        if not mf.exists():
            continue
        for line in mf.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["id"].startswith("sealed"):
                continue  # the sealed set is never trained on
            if r.get("calls") != [] and r.get("label") not in ("noise", "silence"):
                continue  # only all-none rows
            if r.get("split") not in (None, "train"):
                continue
            wav = Path(r["wav"])
            if not wav.is_absolute():
                wav = vad_dir / wav
            rows.append({"id": f"x_{r['id']}", "wav": str(wav), "text": "", "calls": [],
                         "kind": r.get("kind", r.get("label", "neg")), "voice": "", "rate": 0,
                         "pitch": "0%", "split": "train", "seconds": r.get("seconds", 2.0)})
    return rows


def build_examples(fixtures: Path, extra_neg=()):
    rows = [json.loads(line) for line in (fixtures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = rows + list(extra_neg)
    X, Y = [], []
    for row in rows:
        samples = read_wav(fixtures / row["wav"])
        need = (WIN_FRAMES - 1) * HOP + 400
        for start, call in window_labels(row):
            seg = samples[int(start * RATE): int(start * RATE) + need]
            if len(seg) < 400:
                continue
            X.append(logmel(seg)[None])  # [1, 40, 200] -> stacked to [N, 1, 40, 200]
            Y.append({
                "tool": NONE_IDX["tool"] if call is None else TOOL_IDX[call["tool"]],
                **{h: (NONE_IDX[h] if call is None or h not in call else slot_index(h, call.get(h)))
                   for h in ("room", "device", "scene", "level", "seconds")},
            })
    return np.stack(X), Y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--neg-weight", type=float, default=0.25,
                    help="loss weight of all-none (negative/silence) windows")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2")
    ap.add_argument("--extra-negatives", type=Path, nargs="*",
                    default=[Path("D:/models/ncs-shave-research/vad")],
                    help="fixture dirs whose train noise/silence/OOD clips become extra "
                         "all-none windows (vad + the OOD corpus)")
    ap.add_argument("--reject-weight", type=float, default=3.0,
                    help="multiplier on the tool head's reject-class weight")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "bundles" / "voicetools")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    print("building windows...", flush=True)
    extra = []
    for d in args.extra_negatives:
        extra.extend(extra_negative_rows(d))
    print(f"extra non-speech/OOD negatives: {len(extra)}", flush=True)
    X, Y = build_examples(args.fixtures, extra)
    n = len(X)
    idx = rng.permutation(n)
    n_valid = int(0.08 * n)
    valid_idx, train_idx = idx[:n_valid], idx[n_valid:]
    print(f"windows: {n} (train {len(train_idx)}, valid {n_valid})", flush=True)

    # Per-head class weights (inverse-sqrt frequency): fights the none-dominated
    # collapse the first recipe fell into (flat 0.7255 valid acc = predict none).
    weights = {}
    for h in HEADS:
        counts = np.bincount([Y[i][h] for i in train_idx], minlength=HEADS[h]).astype(np.float32)
        w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        if h == "tool":
            w[NONE_IDX["tool"]] *= args.reject_weight  # push borderline windows to reject
        weights[h] = torch.from_numpy(w / w.mean())
    win_w = torch.tensor([1.0 if Y[i]["tool"] != NONE_IDX["tool"] else args.neg_weight
                          for i in train_idx])

    xt = torch.from_numpy(X[train_idx])
    yv = {h: torch.tensor([row[h] for row in [Y[i] for i in valid_idx]]) for h in HEADS}
    xva = torch.from_numpy(X[valid_idx])

    net = HeadedNet(arch=args.arch)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for b in range(0, len(xt), args.batch):
            sel = perm[b:b + args.batch]
            batch = xt[sel]
            idxs = [train_idx[i] for i in sel.tolist()]
            targets = {h: torch.tensor([Y[i][h] for i in idxs]) for h in HEADS}
            wsel = win_w[sel]
            opt.zero_grad()
            out = net(batch)
            loss = sum((F.cross_entropy(out[h], targets[h], weight=weights[h], reduction="none")
                        * wsel).mean() for h in HEADS)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        net.eval()
        with torch.no_grad():
            vout = net(xva)
            # honest metric: tool accuracy on positive windows; slot accuracy where present
            pos = yv["tool"] != NONE_IDX["tool"]
            tool_acc = float((vout["tool"].argmax(1)[pos] == yv["tool"][pos]).float().mean()) if pos.any() else 0.0
            slot_accs = {}
            for h in ("room", "device", "scene", "level", "seconds"):
                m = yv[h] != NONE_IDX[h]
                slot_accs[h] = round(float((vout[h].argmax(1)[m] == yv[h][m]).float().mean()), 3) if m.any() else None
        history.append({"epoch": epoch, "train_loss": round(total / len(xt), 4),
                        "valid_tool_acc_pos_windows": round(tool_acc, 4), "valid_slot_acc": slot_accs})
        print(history[-1], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    probe = torch.from_numpy(X[:1])
    torch.onnx.export(net, probe, str(args.out / "model.onnx"), opset_version=9,
                      input_names=["features"],
                      output_names=list(HEADS), dynamo=False)
    info = {
        "model": "voicetools", "spec_version": TASK_SPEC_VERSION,
        "labels": {"tools": TOOLS_SORTED, "enums": SLOT_ENUMS, "none_idx": NONE_IDX},
        "input_shape": [1, 1, 40, 200], "seed": SEED, "epochs": args.epochs, "arch": args.arch,
        "recipe": {"lr": args.lr, "neg_weight": args.neg_weight, "class_weights": "inverse-sqrt"},
        "history": history, "windows": n,
        "acceptance": ACCEPTANCE,
        "onnx_sha256": hashlib.sha256((args.out / "model.onnx").read_bytes()).hexdigest(),
    }
    (args.out / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"exported {args.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
