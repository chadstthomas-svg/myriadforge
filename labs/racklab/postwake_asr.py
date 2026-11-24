"""Score critical fields after a fixed raw BreakShot wake and common capture."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import unicodedata

import numpy as np

from isolation_bench import FIXTURES, RATE, baseline, detector_for, inworld_lane, load_clip, pcm, score, write_report


WHISPER_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
NUMBERS = {key: str(number) for number, key in enumerate(NUMBER_WORDS)}
NUMBERS.update(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(),
                   ("20", "30", "40", "50", "60", "70", "80", "90")))


def field_tokens(text):
    tokens = re.findall("[a-z]+|[0-9]+", unicodedata.normalize("NFKC", text).casefold())
    return [NUMBERS.get(token, token) for token in tokens if token not in {"dash", "hyphen"}]


def field_match(text, choices):
    """Match complete contiguous field spans with exact letter and digit order."""
    tokens = field_tokens(text)
    expected = {"".join(field_tokens(alias)) for alias in choices}
    limit = max(map(len, expected), default=0)
    for offset in range(len(tokens)):
        joined = ""
        for index in range(offset, len(tokens)):
            joined += tokens[index]
            if len(joined) > limit:
                break
            if joined in expected:
                return True
    return False


def sessions(args):
    paths = [args.whisper_kit]
    if args.cueball:
        paths.append(args.cueball)
    sys.path[:0] = list(map(str, paths))
    from score_asr import create_transcriber, edit_counts, normalize

    transcribe, whisper = create_transcriber(
        args.whisper_kit / "whisper-tiny.en", args.whisper_kit / "whisper-manifest.json",
    )
    if whisper["repository"] != "openai/whisper-tiny.en" or whisper["revision"] != WHISPER_REVISION:
        raise ValueError("Unexpected Whisper checkpoint")
    voice = None
    if args.cueball:
        from infer22 import Voice22

        voice = Voice22(args.cueball / "export22", espeak_dir=args.espeak)
    return transcribe, edit_counts, normalize, whisper, voice


def mixes(args, labels, manifest, voice):
    from scipy.signal import resample_poly

    if set(labels["existing"]) != manifest.keys():
        raise ValueError("The frozen field labels must match the eleven industry workflows")
    for key, entry in manifest.items():
        cases = entry["breakshot"]["conditions"]
        clean, mixed = [next(row for row in cases if row["condition"] == kind) for kind in ("speech", "mixed")]
        yield (key, "existing", labels["existing"][key], entry["cueball"]["input"],
               load_clip(args.fixtures / clean["audio"], clean["sha256"]),
               load_clip(args.fixtures / mixed["audio"], mixed["sha256"]), mixed["events"])
    if voice is None:
        return
    for item in labels["synthetic_identifiers"]:
        noise_row = next(row for row in manifest[item["noise"]]["breakshot"]["conditions"]
                         if row["condition"] == "ambient")
        noise = load_clip(args.fixtures / noise_row["audio"], noise_row["sha256"])
        speech, metadata = voice.synthesize(item["text"])
        if metadata["sample_rate"] != 22050:
            raise ValueError("Unexpected CueBall sample rate")
        clean = np.pad(resample_poly(speech, 320, 441), 2400).astype(np.float32)
        noise = np.resize(noise, len(clean))
        noise *= np.linalg.norm(clean) / np.linalg.norm(noise)
        raw = clean + noise
        scale = max(1, float(np.abs(raw).max()))
        yield (item["id"], "synthetic-identifiers", item, item["text"],
               pcm(clean / scale).astype(np.float32) / 32768,
               pcm(raw / scale).astype(np.float32) / 32768, None)


def measure(audio, crop, prompt, fields, transcribe, edit_counts, normalize):
    if audio.ndim != 1 or not crop < len(audio) <= RATE * 30 or not np.isfinite(audio).all():
        raise ValueError("Invalid matched ASR capture")
    # Score the same PCM16 range the file/live boundary would deliver.
    capture = pcm(audio[crop:]).astype(np.float32) / 32768
    started = perf_counter()
    text = transcribe(capture)
    elapsed = (perf_counter() - started) * 1000
    words = normalize(prompt)
    edits = edit_counts(words, normalize(text))
    result = {
        "transcript": text, "fields": {name: field_match(text, forms) for name, forms in fields.items()},
        "words": len(words), "errors": int(sum(edits)), "asr_ms": elapsed,
    }
    return result


def summarize(rows):
    output = {}
    slices = {
        "all-captures": rows,
        "raw-wake": [row for row in rows if row["raw_wake"]],
        "forced-capture": [row for row in rows if not row["raw_wake"]],
        "synthetic-identifiers": [row for row in rows if row["cohort"] == "synthetic-identifiers"],
    }
    lanes = list(rows[0]["systems"])
    for group, part in slices.items():
        if not part:
            continue
        counts = {lane: Counter() for lane in lanes}
        for row in part:
            reference = row["systems"]["raw"]["fields"]
            for lane, stats in row["systems"].items():
                fields = stats["fields"]
                counts[lane].update({
                    "clips": 1, "fields": len(fields), "exact_fields": sum(fields.values()),
                    "all_fields_exact": int(all(fields.values())), "word_errors": stats["errors"],
                    "words": stats["words"],
                    "gains_vs_raw": sum(matches and not reference[key] for key, matches in fields.items()),
                    "losses_vs_raw": sum(reference[key] and not matches for key, matches in fields.items()),
                })
        output[group] = {lane: {**value, "wer": value["word_errors"] / value["words"]}
                         for lane, value in counts.items()}
    return output


def run(args):
    if not 0 <= args.pre_roll_ms <= 5_000:
        raise ValueError("Pre-roll must be between 0 and 5,000 ms")
    pre_roll = args.pre_roll_ms * RATE // 1000
    fixtures = (args.fixtures / "audio-results.json").read_bytes()
    if sha256(fixtures).hexdigest() != FIXTURES:
        raise ValueError("Use the pinned industry-wave-02 CPU audio fixture manifest")
    probes = Path(__file__).with_name("postwake-probes.json").read_bytes()
    transcribe, edits, normalize, whisper, voice = sessions(args)
    stream, detector = detector_for(args)
    spectral, _ = baseline("spectral")
    isolators = {"compact": spectral}
    profiles = {}
    if args.inworld:
        isolate, profile = inworld_lane("inworld-realtime", args.inworld, 2)
        isolators["inworld-realtime"] = isolate
        profiles["inworld-realtime"] = profile
    args.out.mkdir(parents=True)
    rows = []
    for key, cohort, config, prompt, clean, raw, recorded in mixes(args, json.loads(probes), json.loads(fixtures)["workflows"], voice):
        events = score(stream, raw)
        if recorded is not None and recorded != events:
            raise ValueError(f"Raw wake trace changed for {key}")
        wakes = [event for event in events if event["type"] == "wake"]
        crop = max(0, wakes[0]["start"] - pre_roll) if wakes else 0
        fields = config["fields"]
        systems = {}
        variants = {"raw": raw, "clean-control": clean}
        for name, isolate in isolators.items():
            cleaned, _, _, _ = isolate(raw)
            if cleaned.shape != raw.shape:
                raise ValueError(f"Isolation changed the timeline for {key}")
            variants[name] = cleaned
        for name, samples in variants.items():
            systems[name] = measure(samples, crop, prompt, fields, transcribe, edits, normalize)
        rows.append({
            "id": key, "cohort": cohort, "sector": config["sector"], "prompt": prompt,
            "expected_fields": fields, "samples": len(raw), "crop_start_samples": crop,
            "raw_pcm16_sha256": sha256(pcm(raw).tobytes()).hexdigest(),
            "clean_pcm16_sha256": sha256(pcm(clean).tobytes()).hexdigest(),
            "raw_wake": bool(wakes), "raw_events": events, "systems": systems,
        })
        print(key, "wake", bool(wakes), {name: sum(s["fields"].values()) for name, s in systems.items()}, flush=True)
    summary = summarize(rows)
    gates = {name: stats["gains_vs_raw"] > 0 and stats["losses_vs_raw"] == 0
             for name, stats in summary["raw-wake"].items() if name in isolators}
    write_report(args.out, {
        "scope": "authored post-wake ASR replay with synthetic CueBall speech",
        "labels_sha256": sha256(probes).hexdigest(), "fixture_manifest_sha256": FIXTURES,
        "detector": detector, "whisper": {"repository": whisper["repository"], "revision": whisper["revision"],
                                         "weights_sha256": whisper["files"]["model.safetensors"]["sha256"]},
        "pre_roll_ms": args.pre_roll_ms, "isolators": profiles, "summary": summary, "passes_authored_gate": gates,
        "libraries": {package: version(package) for package in ("torch", "transformers", "numpy", "scipy")},
        "rows": rows,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("detector-checkout", "research", "fixtures", "whisper-kit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cueball", type=Path)
    parser.add_argument("--espeak", type=Path)
    parser.add_argument("--inworld", choices=("cpu", "cuda"))
    parser.add_argument("--pre-roll-ms", type=int, default=400)
    run(parser.parse_args())
