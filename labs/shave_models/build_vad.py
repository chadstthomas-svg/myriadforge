"""Build the VAD v0 bootstrap set (vad/1): deterministic noise synthesis + manifest.

Speech comes from the voicetools fixtures (TTS conditions, held-out cells) and
mini_speech_commands held-out speakers (real audio, SHA-256 in fetch_data.py).
Non-speech is synthesized from scratch: seeded noise generators (white, pink,
50/60 Hz hum, clicks, tone bursts, SNR-scaled speech-band noise) written as WAVs;
the sealed set uses different generator seeds AND unseen generator kinds, so
"held-out noise" is honestly out-of-distribution. Manifest rows mirror the
voicetools manifest shape so the shared eval/windowing harness serves both.
"""
import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

from vad_spec import SAMPLE_RATE

SEED = 132
TRAIN_NOISE = 1200
SEALED_NOISE = 300          # separate seeds + kinds (see NOISE_KINDS_SEALED)
SILENCE_TRAIN = 300
SILENCE_SEALED = 60
RATE = SAMPLE_RATE


def white(rng, n):
    return rng.standard_normal(n)


def pink(rng, n):
    # Voss-McCartney-ish: sum of octave bands
    out = np.zeros(n)
    ncomp = 12
    for octave in range(ncomp):
        period = 2 ** octave
        impulses = rng.standard_normal(n // period + 2)
        out += np.repeat(impulses, period)[:n] / np.sqrt(period)
    return out


def hum(rng, n):
    t = np.arange(n) / RATE
    f = rng.choice([50.0, 60.0, 100.0])
    return np.sin(2 * math.pi * f * t) + 0.3 * np.sin(2 * math.pi * 2 * f * t + rng.uniform(0, 6))


def clicks(rng, n):
    out = np.zeros(n)
    for _ in range(rng.randint(4, 12)):
        pos = rng.randint(0, n - 64)
        amp = rng.uniform(0.4, 1.0)
        out[pos:pos + 64] += amp * np.exp(-np.arange(64) / 12.0) * rng.standard_normal(64)
    return out


def tone_bursts(rng, n):
    t = np.arange(n) / RATE
    out = np.zeros(n)
    for _ in range(rng.randint(3, 8)):
        f = rng.uniform(300, 3000)
        start = rng.randint(0, max(1, n - RATE // 4))
        length = rng.randint(RATE // 16, RATE // 4)
        env = np.hanning(min(length, n - start))
        out[start:start + len(env)] += env * np.sin(2 * math.pi * f * t[:len(env)])
    return out


def speech_band(rng, n):
    # crude speech-shaped noise: low-pass white via cumulative smoothing
    x = rng.standard_normal(n)
    kernel = np.hanning(160)
    return np.convolve(x, kernel / kernel.sum(), mode="same")


NOISE_KINDS = {"white": white, "pink": pink, "hum": hum, "clicks": clicks,
               "tone_bursts": tone_bursts, "speech_band": speech_band}
# Sealed set: same families but distinct seeds PLUS held-out mixes the trainer never sees.
NOISE_KINDS_SEALED = {"mix_white_pink": None, "mix_hum_clicks": None}  # kinds built by mixing


def write_wav(path, samples):
    import hashlib
    x = np.asarray(samples, dtype=np.float64)
    peak = np.max(np.abs(x)) or 1.0
    # stable per-file amplitude (0.15-0.9) from the name - no per-process hash()
    h = int(hashlib.blake2s(path.stem.encode(), digest_size=4).hexdigest(), 16) % 1000
    x = x / peak * (0.15 + 0.75 * h / 1000.0)
    data = (np.clip(x, -1, 1) * 32000).astype(np.int16)
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(RATE)
        dst.writeframes(data.tobytes())


def build(out, mini_cache):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.RandomState(SEED)
    n_short = int(0.55 * RATE)
    n_long = int(1.9 * RATE)
    kinds = sorted(NOISE_KINDS)
    for i in range(TRAIN_NOISE):
        kind = kinds[i % len(kinds)]
        gen = NOISE_KINDS[kind](rng, rng.choice([n_short, n_long]))
        name = f"noise_train_{i:05d}.wav"
        write_wav(out / name, gen)
        rows.append({"id": f"ntr{i:05d}", "wav": name, "label": "noise", "split": "train",
                     "kind": kind, "source": "synth"})
    for i in range(SILENCE_TRAIN):
        n = rng.choice([n_short, n_long])
        name = f"silence_train_{i:05d}.wav"
        write_wav(out / name, rng.standard_normal(n) * 1e-4)
        rows.append({"id": f"str{i:05d}", "wav": name, "label": "silence", "split": "train",
                     "kind": "silence", "source": "synth"})
    # Sealed: held-out noise - new seeds, new mixes, never seen in training
    sealed_rng = np.random.RandomState(SEED + 9000)
    for i in range(SEALED_NOISE):
        a, b = sealed_rng.choice(sorted(NOISE_KINDS), 2, replace=False)
        gen = (NOISE_KINDS[a](sealed_rng, n_long) + 0.5 * NOISE_KINDS[b](sealed_rng, n_long))
        name = f"noise_sealed_{i:05d}.wav"
        write_wav(out / name, gen)
        rows.append({"id": f"nse{i:05d}", "wav": name, "label": "noise", "split": "test",
                     "kind": f"mix_{a}_{b}", "source": "synth-sealed"})
    for i in range(SILENCE_SEALED):
        name = f"silence_sealed_{i:05d}.wav"
        write_wav(out / name, sealed_rng.standard_normal(n_long) * 1e-4)
        rows.append({"id": f"sse{i:05d}", "wav": name, "label": "silence", "split": "test",
                     "kind": "silence", "source": "synth-sealed"})
    # Speech rows reference EXISTING fixtures (voicetools + mini_speech_commands)
    vt = Path("D:/models/ncs-shave-research/voicetools")
    vt_rows = [json.loads(line) for line in (vt / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    speech = [r for r in vt_rows if r["kind"] in ("single", "pair")]
    for r in speech:
        rows.append({"id": f"vt_{r['id']}", "wav": str(vt / r["wav"]), "label": "speech",
                     "split": ("test" if r["split"] == "test" else "train"),
                     "kind": f"voicetools_{r['kind']}", "source": "voicetools"})
    if mini_cache:
        mini_root = mini_cache / "mini_speech_commands"
        nested = mini_root / "mini_speech_commands"
        if nested.exists():  # the zip ships a top-level dir of its own
            mini_root = nested
        if mini_root.exists():
            idx = 0
            for cmd_dir in sorted(p for p in mini_root.iterdir() if p.is_dir()):
                for clip in sorted(cmd_dir.glob("*.wav")):
                    rows.append({"id": f"msc_{cmd_dir.name}_{clip.stem}", "wav": str(clip),
                                 "label": "speech", "split": ("test" if idx % 5 == 0 else "train"),
                                 "kind": "mini_speech", "source": "real"})
                    idx += 1
        else:
            print(f"mini_speech_commands not under {mini_cache} - run "
                  f"fetch_data.py voice --cache <dir>; continuing without real-audio anchor")
    counts = {}
    for r in rows:
        counts[(r["label"], r["split"])] = counts.get((r["label"], r["split"]), 0) + 1
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"manifest: {len(rows)} rows -> {out / 'manifest.jsonl'}")
    for (label, split), n in sorted(counts.items()):
        print(f"  {label:8s} {split:5s} {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("D:/models/ncs-shave-research/vad"))
    ap.add_argument("--mini-cache", type=Path, default=Path("D:/models/ncs-shave-research/cache"))
    args = ap.parse_args()
    build(args.out, args.mini_cache)


if __name__ == "__main__":
    main()
