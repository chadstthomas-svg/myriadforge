"""Build the voice-to-tool fixture set (voicetools/1): plan -> SAPI WAVs -> manifest.

Steps:
  make_voice_tools.py plan   --out <data>/voicetools   writes plan.jsonl (utterance
                             text + TTS condition + ground-truth calls + split,
                             seeded RNG - deterministic across runs/machines).
  make_voice_tools.py render --out <data>/voicetools   invokes render_voice_tools.ps1
                             (one PowerShell process synthesizes every plan line).
  make_voice_tools.py verify --out <data>/voicetools   checks every WAV (16 kHz mono
                             16-bit, duration recorded), adds sha256, writes
                             manifest.jsonl + summary. Re-run any time; it only reads.

Splits are held-out TTS condition cells (the honest "unseen speaker" proxy);
slot values are drawn from the same enums in every split. Silence negatives
are appended at verify time, not synthesized.
"""
import argparse
import hashlib
import itertools
import json
import random
import subprocess
import sys
import wave
from pathlib import Path

from tools_spec import (ACCEPTANCE, DURATION_WORDS, LEVEL_WORDS,
                        NEG_IN_DOMAIN_UNSUPPORTED, NEG_OUT_OF_DOMAIN,
                        NEG_PER_KIND, NEG_UNKNOWN_SLOTS, PAIR_CONNECTIVES,
                        PAIRS, SILENCE_NEGATIVES, SINGLES_PER_TOOL,
                        TEMPLATES, TEST_CELLS, TOOLS, TTS_PITCHES, TTS_RATES,
                        TTS_VOICES, VALID_CELLS, WINDOW_SECONDS, WINDOW_STRIDE)

RATE = 16000
LAB = Path(__file__).resolve().parent
RENDER = LAB / "render_voice_tools.ps1"
SEED = 132


def cells():
    return [(v, r, p) for v in VOICE_ORDER for r in TTS_RATES for p in TTS_PITCHES]


VOICE_ORDER = list(TTS_VOICES)


def split_of(cell):
    if cell in TEST_CELLS:
        return "test"
    if cell in VALID_CELLS:
        return "valid"
    return "train"


def make_positive(tool, rng):
    """Pick the slots ONCE, then derive both the spoken text and the ground-truth
    call from the same values - the text must never disagree with the label."""
    slots = {}
    for key, values in TOOLS[tool].items():
        slots[key] = rng.choice(values)
    template = TEMPLATES[tool][rng.randrange(len(TEMPLATES[tool]))]
    kwargs = dict(slots)
    if "level" in slots:
        kwargs["level"] = LEVEL_WORDS[slots["level"]]
    if "seconds" in slots:
        kwargs["dur"] = DURATION_WORDS[slots["seconds"]]
    return template.format(**kwargs), [{"tool": tool, **slots}]


def build_plan(rng):
    rows = []

    def add(text, calls, kind):
        cell = cells()[rng.randrange(len(cells()))]
        rows.append({"text": text, "calls": calls, "kind": kind,
                     "voice": cell[0], "rate": cell[1], "pitch": cell[2],
                     "split": split_of(cell)})

    singles = 0
    for tool in sorted(TOOLS):
        for _ in range(SINGLES_PER_TOOL):
            text, calls = make_positive(tool, rng)
            singles += 1
            add(text, calls, "single")

    for _ in range(PAIRS):
        first, second = rng.sample(sorted(TOOLS), 2)
        t1, c1 = make_positive(first, rng)
        t2, c2 = make_positive(second, rng)
        add(t1 + rng.choice(PAIR_CONNECTIVES) + t2, c1 + c2, "pair")

    for kind, texts in (("neg_out_of_domain", NEG_OUT_OF_DOMAIN),
                        ("neg_unsupported", NEG_IN_DOMAIN_UNSUPPORTED),
                        ("neg_unknown_slot", NEG_UNKNOWN_SLOTS)):
        for _ in range(NEG_PER_KIND):
            add(texts[rng.randrange(len(texts))], [], kind)

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"vt{i:05d}"
        row["wav"] = f"{row['id']}.wav"
    return rows


def cmd_plan(out):
    rows = build_plan(random.Random(SEED))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plan.jsonl").open("w", encoding="utf-8") as dst:
        for row in rows:
            row["out"] = str(out / row["wav"])
            dst.write(json.dumps(row) + "\n")
    kinds = {}
    splits = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    print(f"plan: {len(rows)} utterances -> {out / 'plan.jsonl'}")
    print(f"  kinds  {kinds}")
    print(f"  splits {splits}")
    print(f"  negative kinds total {sum(v for k, v in kinds.items() if k.startswith('neg'))} "
          f"(acceptance needs >= {ACCEPTANCE['min_negatives']}, plus {SILENCE_NEGATIVES} silence)")


def cmd_render(out):
    plan = out / "plan.jsonl"
    if not plan.exists():
        sys.exit("no plan.jsonl - run `make_voice_tools.py plan --out ...` first")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(RENDER), "-Plan", str(plan)],
        check=False, text=True, errors="replace")
    if result.returncode != 0:
        sys.exit(result.stdout or "render failed")


def wav_info(path):
    with wave.open(str(path), "rb") as src:
        return src.getframerate(), src.getnchannels(), src.getsampwidth(), src.getnframes()


def sha256(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def cmd_verify(out):
    plan = out / "plan.jsonl"
    rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest, bad = [], []
    for row in rows:
        wav = Path(row["out"])
        if not wav.exists():
            bad.append((row["id"], "missing"))
            continue
        rate, ch, width, frames = wav_info(wav)
        if (rate, ch, width) != (RATE, 1, 2):
            bad.append((row["id"], f"format {rate}/{ch}/{width}"))
            continue
        seconds = frames / rate
        windows = max(1, int((max(0.0, seconds - WINDOW_SECONDS)) / WINDOW_STRIDE) + 1)
        manifest.append({**{k: row[k] for k in
                            ("id", "wav", "text", "calls", "kind", "voice", "rate", "pitch", "split")},
                         "seconds": round(seconds, 3), "windows": windows,
                         "sha256": sha256(wav)})
    # Silence negatives: low-level noise + trailing quiet, deterministic, never a call.
    import struct
    noise = random.Random(SEED)
    for i in range(SILENCE_NEGATIVES):
        frames = noise.randrange(int(0.8 * RATE), int(1.9 * RATE))
        payload = b"".join(struct.pack("<h", noise.randint(-80, 80)) for _ in range(frames))
        wav = out / f"sil{i:03d}.wav"
        with wave.open(str(wav), "wb") as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(RATE)
            dst.writeframes(payload)
        manifest.append({"id": f"sil{i:03d}", "wav": wav.name, "text": "", "calls": [],
                         "kind": "neg_silence", "voice": "", "rate": 0, "pitch": "0%",
                         "split": "test", "seconds": round(frames / RATE, 3),
                         "windows": 1, "sha256": sha256(wav)})

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    if bad:
        print(f"BAD: {len(bad)} (first 5: {bad[:5]})")
        return 1
    splits, kinds, windows = {}, {}, 0
    for row in manifest:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        windows += row["windows"]
    print(f"manifest: {len(manifest)} clips -> {out / 'manifest.jsonl'}")
    print(f"  splits {splits}  kinds {kinds}")
    print(f"  total {windows} x {WINDOW_SECONDS:.1f} s graph windows "
          f"(stride {WINDOW_STRIDE:.1f} s, 50% overlap)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["plan", "render", "verify"])
    ap.add_argument("--out", type=Path, required=True, help="fixture directory (data lives outside the repo)")
    args = ap.parse_args()
    {"plan": cmd_plan, "render": cmd_render, "verify": cmd_verify}[args.step](args.out)


if __name__ == "__main__":
    sys.exit(main())
