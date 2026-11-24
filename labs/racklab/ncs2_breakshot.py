"""Compare the pinned BreakShot graphs on physical Myriad X with the CPU replay."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

import numpy as np

from isolation_bench import DESED, DETECTOR_MANIFEST, FIXTURES, load_clip, score


def usb_products():
    command = ("Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "
               "'USB*VID_03E7*' } | Select-Object -ExpandProperty InstanceId")
    rows = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        text=True, timeout=20,
    )
    return sorted(re.findall(r"VID_03E7&PID_([0-9A-F]{4})", rows.upper()))


def replay_fixtures(stream, fixtures):
    fixture_manifest = (fixtures / "audio-results.json").read_bytes()
    if sha256(fixture_manifest).hexdigest() != FIXTURES:
        raise ValueError("Use the pinned industry audio manifest")
    rows = []
    for key, workflow in json.loads(fixture_manifest)["workflows"].items():
        for condition in workflow["breakshot"]["conditions"]:
            audio = load_clip(fixtures / condition["audio"], condition["sha256"])
            events = score(stream, audio)
            woke = any(event["type"] == "wake" for event in events)
            rows.append({"workflow": key, "condition": condition["condition"], "woke_on_ncs2": woke,
                         "woke_on_cpu": condition["observed_wake"], "events": events,
                         "exact_cpu_trace": events == condition["events"],
                         "expected_wake": condition["expected_wake"]})
    return {"scope": "physical NCS2 raw BreakShot replay on authored clips",
            "fixture_manifest_sha256": FIXTURES, "checks": len(rows),
            "matches_labels": sum(row["woke_on_ncs2"] == row["expected_wake"] for row in rows),
            "matches_cpu_wake": sum(row["woke_on_ncs2"] == row["woke_on_cpu"] for row in rows),
            "matches_cpu_trace": sum(row["exact_cpu_trace"] for row in rows), "rows": rows}


def desed_rows(desed):
    raw = (desed / "manifest.jsonl").read_bytes()
    if sha256(raw).hexdigest() != DESED:
        raise ValueError("Use the pinned DESED room regression manifest")
    return list(map(json.loads, raw.splitlines()))


def replay_desed(stream, desed):
    rows = []
    for index, row in enumerate(desed_rows(desed), 1):
        audio = load_clip(desed / "audio" / row["file"], row["wav_sha256"])
        if len(audio) != row["samples"]:
            raise ValueError("Audio length differs from annotations")
        events = score(stream, audio)
        cuts = [(event["start"], event["end"]) for event in events if event["type"] == "interval"]
        rows.append({"id": row["id"], "cuts": cuts})
        if index % 100 == 0:
            print(index, "annotated recordings", flush=True)
    return {"scope": "physical NCS2 raw BreakShot replay on annotated DESED room audio",
            "manifest_sha256": DESED, "recordings": len(rows), "rows": rows}


def measure_desed(args):
    sys.path.insert(0, str(args.research / "breakshot-holdout" / "voxconverse-test-v1"))
    from score_vox import one

    expected = desed_rows(args.desed)
    recorded = json.loads(args.score_input.read_bytes())
    observed = recorded["rows"]
    if recorded["manifest_sha256"] != DESED or [x["id"] for x in expected] != [x["id"] for x in observed]:
        raise ValueError("Replay does not correspond to the pinned DESED recordings")
    rows = [{"id": gold["id"], **one(gold, replay["cuts"])} for gold, replay in zip(expected, observed)]
    counts = {key: sum(row[key] for row in rows)
              for key in ("samples", "truth", "covered", "expected_events", "missed_events", "false_wakes")}
    counts["detected_events"] = counts["expected_events"] - counts["missed_events"]
    save(args.out, {**recorded, "summary": counts, "rows": rows})


def run(args):
    from openvino.runtime import Core, get_version

    products = usb_products()
    if products.count("2485") != 1:
        raise RuntimeError(f"Expected one cold NCS2, USB 03E7:2485. Visible products: {products}")
    core = Core()
    candidates = [device for device in core.available_devices
                  if device.split(".")[0] == "MYRIAD"
                  and "Myriad X" in core.get_property(device, "FULL_DEVICE_NAME")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Myriad X target in OpenVINO: {candidates}")
    target = candidates[0]
    device_name = core.get_property(target, "FULL_DEVICE_NAME")
    artifact = args.research / "racklab-launch" / "cached-head"
    raw_manifest = (artifact / "manifest.json").read_bytes()
    if sha256(raw_manifest).hexdigest() != DETECTOR_MANIFEST:
        raise ValueError("Use the pinned cached BreakShot detector")
    manifest = json.loads(raw_manifest)
    digests = {}
    times = defaultdict(list)
    placement = {}

    def graph(stage, output):
        directory = artifact / stage
        model = directory / "model.onnx"
        if sha256(model.read_bytes()).hexdigest() != manifest["graph_sha256"][stage]:
            raise ValueError(f"Changed {stage} graph")
        xml = directory / "ir" / "model.xml"
        digests[stage] = ({"onnx": manifest["graph_sha256"][stage]} if args.graph_format == "onnx"
                          else {ext: sha256(xml.with_suffix("." + ext).read_bytes()).hexdigest()
                                for ext in ("xml", "bin")})
        chosen = xml if args.graph_format == "ir" else model
        selected = "CPU" if stage == "scan" and args.scan_device == "cpu" else target
        compiled = core.compile_model(str(chosen), selected)
        placement[stage] = selected
        request = compiled.create_infer_request()
        port = compiled.output(output)

        def predict(data):
            started = perf_counter()
            result = request.infer(data)[port].astype(np.float32, copy=True)
            times[stage].append((perf_counter() - started) * 1000)
            return result

        return predict

    root = args.research / "breakshot-holdout" / "silero-mini"
    sys.path[:0] = [str(args.detector_checkout / "labs" / "racklab"),
                    str(root / "acoustic-state"), str(root)]
    from breakshot_cached import CachedStream, FeatureCache
    from tiny_frontend import frame, magnitudes

    stream = CachedStream(FeatureCache(
        (frame, magnitudes), graph("encoder", "frame_features"),
        graph("scan", "cell_probability"), manifest["history_frames"],
    ))
    evidence = (replay_fixtures(stream, args.fixtures) if args.fixtures
                else replay_desed(stream, args.desed))
    result = {
        **evidence,
        "placement": {**placement, "audio_frontend": "host"},
        "graph_format": args.graph_format,
        "boot_usb_products": products, "physical_usb_id": "03E7:2485", "device_name": device_name,
        "openvino": get_version(), "detector_manifest_sha256": DETECTOR_MANIFEST,
        "compiled_input_sha256": digests,
        "stage_wall_p95_ms": {key: float(np.percentile(samples, 95)) for key, samples in times.items()},
    }
    save(args.out, result)


def save(path, result):
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkout", type=Path)
    parser.add_argument("--research", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixtures", type=Path)
    inputs.add_argument("--desed", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--graph-format", choices=("ir", "onnx"), default="onnx")
    parser.add_argument("--scan-device", choices=("ncs2", "cpu"), default="ncs2")
    parser.add_argument("--score-input", type=Path, help="Score recorded DESED cuts in the modern host environment")
    args = parser.parse_args()
    if args.score_input:
        if not args.desed:
            parser.error("--score-input requires --desed")
        measure_desed(args)
    elif args.detector_checkout:
        run(args)
    else:
        parser.error("A device replay requires --detector-checkout")
