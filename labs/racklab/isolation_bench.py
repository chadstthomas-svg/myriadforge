"""Compare matched BreakShot wakes before and after voice isolation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
import wave

import numpy as np

from isolation import SpectralIsolation


RATE = 16_000
CHUNK = 320
REVISION = "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970"
MODELS = {
    "inworld-realtime": ("denoiser_realtime.safetensors", "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f"),
    "inworld-offline": ("denoiser.safetensors", "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53"),
}
FIXTURES = "8184d8dc881681d2119d1d719dac05a858286df0966186955b8e76b64bb467fa"
DESED = "a88aa2a4fdcc2f89ae2b54a0b47f9277878e24c06ad26d00d5a6a0f3786d522a"
DETECTOR_MANIFEST = "337795a1ab2a5ffbb121b053591996ac0e60e46b05379ea067f59566a5fb54a3"


def load_clip(path, checksum):
    if sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Changed audio fixture: {path.name}")
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"Invalid audio fixture: {path.name}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def pcm(samples):
    return np.rint(samples * 32768).clip(-32768, 32767).astype("<i2")


def save_clip(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm(samples).tobytes())


def chunks(stream, audio, block):
    output = []
    delivered = None
    started = perf_counter()
    costs = []
    for start in range(0, len(audio), block):
        before = perf_counter()
        block_out = stream.push(audio[start:start + block])
        costs.append((perf_counter() - before) * 1000)
        output.append(block_out)
        if delivered is None and len(block_out):
            delivered = min(start + block, len(audio))
    output.append(stream.flush())
    return output, delivered, costs, (perf_counter() - started) * 1000


def inworld_lane(lane, device, threads):
    import importlib.metadata

    from huggingface_hub import hf_hub_download
    from inworld import voice_isolation
    from scipy.signal import resample_poly
    import torch

    if importlib.metadata.version("inworld-voice-isolation") != "1.0.0":
        raise ValueError("Pin inworld-voice-isolation==1.0.0 for this benchmark")
    torch.set_num_threads(threads)
    name, expected = MODELS[lane]
    path = Path(hf_hub_download("inworld/voice-isolator", name, revision=REVISION, token=False))
    with path.open("rb") as file:
        if sha256(file.read()).hexdigest() != expected:
            raise ValueError("Unexpected Inworld checkpoint digest")
    realtime = lane == "inworld-realtime"
    started = perf_counter()
    model = voice_isolation.VoiceIsolator(
        str(path), mode="streaming" if realtime else "offline", device=device, precision="fp32",
        **({"lookahead_frames": 2} if realtime else {}),
    )
    init_ms = (perf_counter() - started) * 1000

    def enhance(audio):
        up = torch.from_numpy(np.asarray(resample_poly(audio, 3, 1), dtype=np.float32))
        with torch.inference_mode():
            if realtime:
                blocks, first, costs, model_ms = chunks(model.stream(), up, CHUNK * 3)
                voice = torch.cat(blocks)
            else:
                started = perf_counter()
                voice, first, costs = model.isolate(up, 48_000), None, []
                model_ms = (perf_counter() - started) * 1000
        cleaned = resample_poly(voice.detach().cpu().numpy().reshape(-1), 1, 3)
        return cleaned, None if first is None else first // 3, costs, model_ms

    return enhance, {"checkpoint_sha256": expected, "checkpoint_bytes": path.stat().st_size,
                     "model_revision": REVISION, "device": device,
                     "torch": torch.__version__, "precision": "fp32", "threads": threads, "init_ms": init_ms}


def baseline(lane):
    if lane == "raw":
        return lambda audio: (audio, 0, [], 0), {"device": "cpu", "weights_bytes": 0}

    def enhance(audio):
        blocks, first, costs, busy_ms = chunks(SpectralIsolation(), audio, CHUNK)
        return np.concatenate(blocks), first, costs, busy_ms

    return enhance, {"device": "cpu", "weights_bytes": 0, "hop_samples": SpectralIsolation.hop,
                     "lookahead_samples": SpectralIsolation.lookahead_samples}


def si_sdr(reference, candidate):
    target = reference.astype(np.float64) - np.mean(reference, dtype=np.float64)
    output = candidate.astype(np.float64) - np.mean(candidate, dtype=np.float64)
    target *= np.dot(output, target) / max(float(np.dot(target, target)), 1e-12)
    error = output - target
    return round(10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(error, error) + 1e-12)), 3)


def score(stream, samples):
    stream.reset()
    audio = pcm(samples)
    events = []
    for start in range(0, len(audio), stream.block):
        events.extend(stream.push(audio[start:start + stream.block]))
    events.extend(stream.flush())
    return events


def confirmed_score(stream, samples, isolated):
    """Allow filtered posteriors to help only in the raw detector's hysteresis band."""
    predictor = stream.posterior
    filtered = []

    def record(*args):
        confidence = predictor(*args)
        filtered.append(confidence.copy())
        return confidence

    def combine(*args):
        raw = predictor(*args)
        cleaned = next(observations)
        if cleaned.shape != raw.shape:
            raise ValueError("Detector timelines did not align")
        return np.maximum(raw, np.where(raw >= 0.60, cleaned, raw))

    try:
        stream.posterior = record
        score(stream, isolated)
        observations = iter(filtered)
        stream.posterior = combine
        events = score(stream, samples)
        if next(observations, None) is not None:
            raise ValueError("Detector timelines did not align")
        return events
    finally:
        stream.posterior = predictor


def detect(lane, stream, audio, voice):
    if lane == "spectral-confirm":
        return confirmed_score(stream, audio, voice)
    return score(stream, voice)


def detector_for(args):
    # The existing research runner verifies its two exported ONNX graph digests on open.
    sys.path.insert(0, str(args.detector_checkout / "labs" / "racklab"))
    from breakshot_cached import open_cached

    model_dir = args.research / "racklab-launch" / "cached-head"
    if sha256((model_dir / "manifest.json").read_bytes()).hexdigest() != DETECTOR_MANIFEST:
        raise ValueError("Use the pinned cached BreakShot detector")
    stream, detector = open_cached(args.research / "breakshot-holdout" / "silero-mini" / "acoustic-state", model_dir, "cpu")
    detector["manifest_sha256"] = DETECTOR_MANIFEST
    return stream, detector


def run(args):
    frozen = (args.fixtures / "audio-results.json").read_bytes()
    if sha256(frozen).hexdigest() != FIXTURES:
        raise ValueError("Use the pinned industry-wave-02 CPU audio fixture manifest")
    manifest = json.loads(frozen)["workflows"]
    if args.workflow and not set(args.workflow) <= manifest.keys():
        raise ValueError("Unknown workflow")
    stream, detector = detector_for(args)
    enhance, profile = inworld_lane(args.lane, args.device, args.threads) if args.lane in MODELS else baseline(args.lane)
    args.out.mkdir(parents=True)
    selected = args.workflow or list(manifest)
    results = []
    for name in selected:
        cases = manifest[name]["breakshot"]["conditions"]
        clean = next(case for case in cases if case["condition"] == "speech")
        reference = load_clip(args.fixtures / clean["audio"], clean["sha256"])
        for case in cases:
            audio = load_clip(args.fixtures / case["audio"], case["sha256"])
            started = perf_counter()
            voice, first, costs, model_ms = enhance(audio)
            wall_ms = (perf_counter() - started) * 1000
            if voice.shape != audio.shape or not np.isfinite(voice).all():
                raise ValueError(f"Lane changed sample alignment or returned invalid samples: {case['audio']}")
            events = detect(args.lane, stream, audio, voice)
            woke = any(event["type"] == "wake" for event in events)
            if args.lane == "raw" and events != case["events"]:
                raise ValueError(f"Original BreakShot event trace changed for {case['audio']}")
            if args.save_audio:
                save_clip(args.out / case["audio"], voice)
            result = {
                "workflow": name, "condition": case["condition"], "input_sha256": case["sha256"],
                "expected_wake": case["expected_wake"], "observed_wake": woke, "events": events,
                "input_seconds": len(audio) / RATE, "isolation_ms": wall_ms, "model_ms": model_ms,
                "rtf": wall_ms / (len(audio) * 1000 / RATE),
                "first_output_input_ms": None if first is None else first * 1000 / RATE,
                "push_p95_ms": float(np.percentile(costs, 95)) if costs else None,
                "si_sdr_db": si_sdr(reference, voice) if case["condition"] == "mixed" else None,
            }
            results.append(result)
            print(name, case["condition"], woke, f"{result['rtf']:.3f} RTF", flush=True)
    report = {
        "scope": "matched synthetic speech and licensed ambient replay",
        "lane": args.lane, "profile": profile, "host": platform.platform(),
        "fixture_manifest_sha256": FIXTURES, "detector": detector,
        "wakes_correct": sum(row["observed_wake"] == row["expected_wake"] for row in results),
        "cases": len(results), "median_rtf": statistics.median(row["rtf"] for row in results),
        "rows": results,
    }
    write_report(args.out, report)


def run_desed(args):
    if args.lane in MODELS or args.workflow or args.save_audio:
        raise ValueError("The DESED check compares compact lanes and stores event counts only")
    raw = (args.desed / "manifest.jsonl").read_bytes()
    if sha256(raw).hexdigest() != DESED:
        raise ValueError("Use the pinned DESED room regression manifest")
    stream, detector = detector_for(args)
    sys.path.insert(0, str(args.research / "breakshot-holdout" / "voxconverse-test-v1"))
    from score_vox import one

    enhance, profile = baseline(args.lane)
    rows = []
    args.out.mkdir(parents=True)
    for i, line in enumerate(raw.splitlines(), 1):
        record = json.loads(line)
        audio = load_clip(args.desed / "audio" / record["file"], record["wav_sha256"])
        if len(audio) != record["samples"]:
            raise ValueError("Audio length differs from annotations")
        voice, _, _, _ = enhance(audio)
        if voice.shape != audio.shape:
            raise ValueError("Isolation changed sample alignment")
        events = detect(args.lane, stream, audio, voice)
        segments = [(event["start"], event["end"]) for event in events if event["type"] == "interval"]
        rows.append({"id": record["id"], **one(record, segments)})
        if i % 100 == 0:
            print(i, "annotated recordings", flush=True)
    totals = {key: sum(row[key] for row in rows)
              for key in ("samples", "truth", "covered", "expected_events", "missed_events", "false_wakes")}
    totals["detected_events"] = totals["expected_events"] - totals["missed_events"]
    write_report(args.out, {"scope": "DESED annotated room-audio regression", "lane": args.lane,
                           "detector": detector, "profile": profile, "manifest_sha256": DESED,
                           "recordings": len(rows), "summary": totals, "rows": rows})
    print(json.dumps(totals))


def write_report(directory, report):
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkout", required=True, type=Path)
    parser.add_argument("--research", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixtures", type=Path)
    inputs.add_argument("--desed", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lane", required=True, choices=("raw", "spectral", "spectral-confirm", *MODELS))
    parser.add_argument("--workflow", action="append")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()
    (run_desed if args.desed else run)(args)


if __name__ == "__main__":
    main()
