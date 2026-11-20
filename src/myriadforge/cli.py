"""MyriadForge CLI — benchmark and explore the Movidius Myriad lanes.

Subcommands:
  devices  - scan every Myriad lane (auto-seen, fail-soft)
  ingest   - HuggingFace repo -> ONNX / OpenVINO IR artifact (prebuilt-first)
  bench    - run a latency/throughput benchmark on a Myriad device -> result JSON
  report   - render results into a self-contained HTML dashboard

The application pilots (gate / footage / podcast / kNN / cascade) live in
``apps/`` and run inside the isolated OpenVINO venvs; see README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from myriadforge import __version__
from myriadforge.devices.registry import benchable, default_providers, discover
from myriadforge.ingest import hf_resolver


def cmd_devices(args: argparse.Namespace) -> int:
    handles, warnings = discover()
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not handles:
        print("no devices discovered")
        return 1
    width = max(len(h.id) for h in handles)
    for h in handles:
        state = "ready" if h.available else "unavailable"
        marker = "" if h.available and h.bench_mode != "none" else "  (not benchable)"
        print(f"  {h.id:<{width}}  {h.kind:<10} {state:<12} {h.bench_mode:<11} {h.label}{marker}")
    print(f"\n{len(benchable(handles))} benchable of {len(handles)} discovered")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    if args.list:
        repo_id = hf_resolver.parse_repo_id(args.spec)
        candidates = hf_resolver.list_model_files(repo_id)
        if candidates:
            print(f"model artifacts in {repo_id}:")
            for f in candidates:
                print(f"  - {f}")
        else:
            print(f"no prebuilt artifacts in {repo_id} (conversion needed)")
        return 0

    result = hf_resolver.ingest_prebuilt(args.spec, pick=args.file)
    print(f"repo:   {result.repo_id}")
    print(f"status: {result.status}")
    if result.local_path:
        print(f"file:   {result.file} ({result.format})")
        print(f"path:   {result.local_path}")
    if result.candidates:
        print(f"candidates: {', '.join(result.candidates)}")
    print(result.message)
    return 0 if result.status == "ok" else 2


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        model_path = hf_resolver.resolve_local_model(args.model)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    handles, _ = discover()
    handle = next((h for h in handles if h.id == args.device), None)
    if handle is None:
        print(f"error: unknown device {args.device!r}; run `myriadforge devices`", file=sys.stderr)
        return 2
    if not handle.available or handle.bench_mode == "none":
        print(f"error: device {handle.id} is not benchable "
              f"({handle.bench_mode}, available={handle.available})", file=sys.stderr)
        return 2

    provider = next(p for p in default_providers() if p.name == handle.lane)
    print(f"model:    {model_path}")
    print(f"device:   {handle.id} - {handle.label}")
    out = provider.bench(
        handle, model_path,
        warmup_runs=args.warmup, measured_runs=args.runs,
        out_dir=Path(args.out),
    )
    record = json.loads(out.read_text(encoding="utf-8"))
    if record.get("status") == "error":
        print(f"error:   {record.get('error', 'bench failed')}", file=sys.stderr)
        print(f"result:  {out}")
        return 2
    _print_bench_record(out, record=record)
    return 0


def _print_bench_record(out: Path, record: dict | None = None) -> None:
    record = record or json.loads(out.read_text(encoding="utf-8"))
    m = record["metrics"]
    print(f"init:     {m['init_ms']:.1f} ms")
    print(f"1st inf:  {m['first_inference_ms']:.1f} ms")
    ss = m["steady_state_ms"]
    print(f"steady:   p50 {ss['p50']:.2f} ms | p90 {ss['p90']:.2f} ms | "
          f"p99 {ss['p99']:.2f} ms | sd {ss['stddev']:.2f} ms  (n={int(ss['runs'])})")
    print(f"throughput: {m['throughput_fps']:.1f} inf/s")
    backend = record["backend"]
    if backend.get("simulated"):
        print(f"simulated: yes ({backend.get('runtime')})")
    if backend.get("pipelined"):
        print(f"pipelined: {backend.get('pipeline_requests')} in-flight requests")
    print(f"result:   {out}")


def cmd_report(args: argparse.Namespace) -> int:
    from myriadforge.report import collect, render

    records = collect.load_results(Path(args.results_dir))
    if not records:
        print(f"no result records in {args.results_dir}", file=sys.stderr)
        return 2
    html = render.render_html(records)
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"report: {out} ({len(records)} record(s))")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myriadforge",
        description="Reviving the Intel Neural Compute Stick (Myriad 2) — benchmarks "
                    "and edge-AI apps on discontinued silicon.",
    )
    parser.add_argument("--version", action="version", version=f"myriadforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("devices", help="scan every Myriad lane (auto-seen)")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("ingest", help="HuggingFace URL/repo -> ONNX/IR artifact")
    p.add_argument("spec", help="HF URL or 'org/name' repo id")
    p.add_argument("--list", action="store_true", help="only list candidate model files")
    p.add_argument("--file", help="specific artifact filename to download")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("bench", help="benchmark a model on a Myriad device")
    p.add_argument("model", help="local path, HF repo id/URL, or registry key")
    p.add_argument("--device", required=True,
                   help="device id from `myriadforge devices` (ncs1-myriad, ncs2-myriad, "
                        "myriad-sim, myriad2-sim, ncs1-ovcpu, ncs2-ovcpu)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--out", default="results", help="results directory")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("report", help="render results/*.json into an HTML dashboard")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="report.html")
    p.set_defaults(fn=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
