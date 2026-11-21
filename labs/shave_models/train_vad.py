"""Train the VAD suite model (vad/1): tiny 3-class window classifier.

Speech / noise / silence per [1,1,40,200] window. The graph is deliberately
~10x smaller than the voicetools model (~1.0M MACs vs ~4.7M) because the
acceptance gate is p95 <= 2.5 ms/window against the measured ~1.5 ms dispatch
floor. Same mel filterbank and window contract as voicetools - one framing
harness serves both suites. Runs in the external training venv.

  train-venv/Scripts/python.exe train_vad.py --fixtures D:/models/ncs-shave-research/vad
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

from vad_spec import ACCEPTANCE, CLASSES, CLASS_IDX, SAMPLE_RATE, TASK_SPEC_VERSION, \
    WINDOW_SECONDS, WINDOW_STRIDE

RATE = SAMPLE_RATE
WIN_FRAMES = 200
HOP = 160
NFFT = 512
SEED = 132


def mel_filters():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import mel_filters as shared
    return shared()


def logmel(samples):
    need = (WIN_FRAMES - 1) * HOP + 400
    if len(samples) < need:
        samples = np.pad(samples, (0, need - len(samples)))
    offsets = np.arange(WIN_FRAMES)[:, None] * HOP + np.arange(400)
    frames = samples[offsets] * np.hanning(400)
    power = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)) ** 2
    mel = np.log(np.maximum(power @ mel_filters(), 1e-8))
    return ((mel.T - mel.mean()) / max(mel.std(), 0.1)).astype(np.float32)[None]


def read_wav(path):
    with wave.open(str(path), "rb") as src:
        assert src.getframerate() == RATE and src.getnchannels() == 1 and src.getsampwidth() == 2, \
            f"unexpected WAV format: {path}"
        return np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def window_starts(seconds):
    if seconds <= WINDOW_SECONDS:
        return [0.0]
    aligned = [w * WINDOW_STRIDE for w in range(int((seconds - WINDOW_SECONDS) / WINDOW_STRIDE) + 1)]
    aligned.append(max(0.0, seconds - WINDOW_SECONDS))
    return sorted(set(round(s, 3) for s in aligned))


class TinyVad(nn.Module):
    """Head = the handoff-proven pattern: GlobalAveragePool over BOTH axes ->
    a tiny Gemm (16->3). The v0 800-wide flatten head measured p95 9.8 ms on
    the stick; this shape matches the 5.0 ms handoff graph."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, (1, 9), padding=(0, 4))
        self.block1 = nn.Conv2d(8, 16, (1, 9), padding=(0, 4))
        self.head = nn.Linear(16, len(CLASSES))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.max_pool2d(F.relu(self.stem(x)), (1, 2))
        h = F.max_pool2d(F.relu(self.block1(h)), (1, 2))
        flat = h.mean(dim=(2, 3))  # full GAP -> [B, 16]
        return self.head(flat)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "bundles" / "vad")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows = [json.loads(line) for line
            in (args.fixtures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    print(f"building windows from {len(rows)} rows...", flush=True)
    X, Y, split_of = [], [], []
    for row in rows:
        wav = Path(row["wav"])
        if not wav.is_absolute():
            wav = args.fixtures / wav
        samples = read_wav(wav)
        need = (WIN_FRAMES - 1) * HOP + 400
        for start in window_starts(len(samples) / RATE):
            seg = samples[int(start * RATE): int(start * RATE) + need]
            if len(seg) < 400:
                continue
            X.append(logmel(seg))
            Y.append(CLASS_IDX[row["label"]])
            split_of.append(row["split"])
    X = np.stack(X)
    Y = np.asarray(Y)
    split_of = np.asarray(split_of)
    train_idx = np.where(split_of == "train")[0]
    valid_idx = np.where(split_of == "test")[0]
    rng.shuffle(train_idx)
    print(f"windows: {len(X)} (train {len(train_idx)}, held-out {len(valid_idx)})", flush=True)

    xt = torch.from_numpy(X[train_idx])
    yt = torch.from_numpy(Y[train_idx])
    xv = torch.from_numpy(X[valid_idx])
    yv = torch.from_numpy(Y[valid_idx])
    counts = np.bincount(Y[train_idx], minlength=len(CLASSES)).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = torch.from_numpy(w / w.mean())

    net = TinyVad()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for b in range(0, len(xt), args.batch):
            sel = perm[b:b + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(net(xt[sel]), yt[sel], weight=w)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        net.eval()
        with torch.no_grad():
            logits = net(xv)
            pred = logits.argmax(1)
            acc = float((pred == yv).float().mean())
            speech_recall = float((pred[yv == CLASS_IDX["speech"]]
                                   == CLASS_IDX["speech"]).float().mean())
            nonspeech = yv != CLASS_IDX["speech"]
            false_wake = float((pred[nonspeech] == CLASS_IDX["speech"]).float().mean())
        history.append({"epoch": epoch, "train_loss": round(total / len(xt), 4),
                        "heldout_acc": round(acc, 4), "speech_recall": round(speech_recall, 4),
                        "false_wake_rate": round(false_wake, 4)})
        print(history[-1], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.from_numpy(X[:1]), str(args.out / "model.onnx"),
                      opset_version=9, input_names=["features"], output_names=["logits"],
                      dynamo=False)
    info = {
        "model": "vad", "spec_version": TASK_SPEC_VERSION,
        "classes": list(CLASSES), "input_shape": [1, 1, 40, 200],
        "seed": SEED, "epochs": args.epochs, "windows": len(X),
        "history": history, "acceptance": ACCEPTANCE,
        "onnx_sha256": hashlib.sha256((args.out / "model.onnx").read_bytes()).hexdigest(),
    }
    (args.out / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"exported {args.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
