"""Sweep, confirm, and soak a qualified full decoder on one physical NCS2."""

from argparse import ArgumentParser
import json
from pathlib import Path
import random
from statistics import median
from subprocess import TimeoutExpired
import sys

from cached_decoder import decoder_fingerprint
from decoder_validation import DEFAULT, PROFILE_SCHEMA, parse_allocation
from worker_process import run_worker

GRID = [(n, n, 1) for n in range(1, 17)] + [(4, 8, 1), (4, 16, 1), (8, 12, 1), (8, 16, 1), (12, 16, 1)]
BINDING = ("checkpoint_sha256", "tokenizer_sha256", "context", "runtime", "device")
RUNNER = Path(__file__).with_name("decoder_validation.py")


def write(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def tune(args):
    control = json.loads(args.qualification.read_text(encoding="utf8"))
    if control["status"] != "ok" or control["mode"] != "qualify" or control["context"] < 128:
        raise ValueError("Supply a passing 128-token or larger decoder qualification")
    if not control.get("context_limit_ok") or control["comparisons"] != 192:
        raise ValueError("Supply the full qualification report")
    binding = {name: control[name] for name in BINDING}
    graph_count = len(control["graphs"])
    configuration = {key: control[key] for key in ("attention", "layout", "norm", "shave_only")}
    config = args.out_dir
    config.mkdir(parents=True, exist_ok=True)
    tally = {"binding": binding, "graph_ranges": control["graphs"], "steps": [], "profile": None}
    rng = random.Random(24)

    def trial(allocation, phase, *, mode="bench", seconds=0):
        setting = ":".join(map(str, allocation))
        label = f"{phase}-{'-'.join(map(str, allocation))}"
        result_file = config / f"{label}.json"
        command = [sys.executable, "-u", str(RUNNER), str(args.checkpoint), str(args.tokenizer),
                   "--controls", str(args.qualification), "--out", str(result_file),
                   "--context", str(control["context"]), "--graphs", str(graph_count),
                   "--layout", configuration["layout"], "--norm", configuration["norm"],
                   "--attention", configuration["attention"], "--allocation", setting,
                   "--mode", mode, "--seconds", str(seconds)]
        if configuration["shave_only"]:
            command.append("--shave-only")
        if phase == "sweep":
            command.append("--screening")
        print(f"START {label}", flush=True)
        try:
            worker = run_worker(command, graph_count * 125 + 180 + seconds)
            log = worker.stdout + "\n" + worker.stderr
            if result_file.exists():
                result = json.loads(result_file.read_text(encoding="utf8"))
            else:
                result = {"status": "worker_error", "detail": log[-1500:]}
        except TimeoutExpired as error:
            log = (error.output or "") + "\n" + (error.stderr or "")
            result = {"status": "timeout", "detail": log[-1500:]}
        (config / f"{label}.log").write_text(log, encoding="utf8")
        entry = {"phase": phase, "allocation": allocation, "status": result["status"]}
        for key in ("decode_ms", "first_token_ms", "load_ms", "detail", "rounds", "parity_checks",
                    "parity_matches", "matched", "comparisons"):
            if key in result:
                entry[key] = result[key]
        tally["steps"].append(entry)
        write(config / "sweep.json", tally)
        print(f"END {label} {entry['status']} {entry.get('decode_ms', {}).get('p95', '')}", flush=True)
        return entry

    choices = list(dict.fromkeys([DEFAULT, *(args.alloc or GRID)]))
    if any(cfg[-1] != 1 for cfg in choices):
        raise ValueError("Single-session latency tuning requires one stream")
    rng.shuffle(choices)
    sweep = [trial(cfg, "sweep") for cfg in choices]
    winners = sorted((entry for entry in sweep if entry["status"] == "ok"),
                     key=lambda entry: entry["decode_ms"]["p95"])[:3]
    finalists = list(dict.fromkeys([DEFAULT, *(entry["allocation"] for entry in winners)]))
    confirmed = {cfg: [] for cfg in finalists}
    for repetition in range(1, 4):
        rng.shuffle(finalists)
        for cfg in finalists:
            confirmed[cfg].append(trial(cfg, f"confirm-{repetition}"))
    valid = {cfg: median(row["decode_ms"]["p95"] for row in entries)
             for cfg, entries in confirmed.items() if all(row["status"] == "ok" for row in entries)}
    for allocation in sorted(valid, key=valid.get):
        if trial(allocation, "final-qualification", mode="qualify")["status"] != "ok":
            continue
        if trial(allocation, "soak", seconds=args.soak_seconds)["status"] != "ok":
            continue
        profile = {
            "schema": PROFILE_SCHEMA, "binding": binding, "allocation": allocation,
            "graphs": graph_count, "graph_ranges": control["graphs"], **configuration,
            "decoder_sha256": decoder_fingerprint(),
            "confirmation_p95_ms": valid[allocation], "stock_p95_ms": valid.get(DEFAULT),
            "soak_seconds": args.soak_seconds,
        }
        tally["profile"] = profile
        write(config / "profile.json", profile)
        write(config / "sweep.json", tally)
        break
    return 0 if tally["profile"] else 1


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alloc", type=parse_allocation, action="append")
    parser.add_argument("--soak-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.soak_seconds < 600:
        parser.error("A publishable tuning profile requires at least 600 seconds of soak")
    sys.exit(tune(args))
