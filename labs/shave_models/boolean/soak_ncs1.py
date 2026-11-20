"""Exercise one resident NCS1 release with batch, full-context and append traffic."""

import argparse
import json
from pathlib import Path
import time

from benchmark import CORPUS_SHA, exact_document, expected, quantiles
from benchmark_batch_heads import windows_usb_identity
from benchmark_compare import RELEASE, TOKENIZER, InferenceTimer, dispatch, file_details, fixtures
from policy import LIMIT
from runtime import Runtime
from stream import DecisionStream


def soak(engine, corpus, duration, interval, report, save):
    short, allow, veto, overflow = fixtures(engine.tokenizer, engine.policy, corpus)
    tokenizer = engine.tokenizer
    source = tokenizer.encode(corpus.read_bytes().decode("utf-8-sig"), out_type=int)
    tail = " do not route"
    tail_count = len(tokenizer.encode(tail, out_type=int))
    prefix = exact_document(tokenizer, source, LIMIT - tail_count,
                            start="duplicate charge. review authorized.")
    assert len(tokenizer.encode(prefix + tail, out_type=int)) == LIMIT
    request = InferenceTimer(engine.request)
    engine.request = request
    stream = DecisionStream(engine)
    workloads = ("short_batch", "full_text", "stream_prefix", "stream_append")

    def reject(action):
        calls = request.calls
        try:
            action()
        except ValueError as error:
            if str(LIMIT + 1) in str(error) and request.calls == calls:
                return
            raise
        raise AssertionError("Accepted over-limit input")

    def cycle(index):
        calls = request.calls
        batch = dispatch(engine, short, len(short))
        full = dispatch(engine, [allow if index % 2 else veto], 1)
        initial = stream.append(prefix, reset=True)
        assert initial["result"] == expected([1, 4])
        appended = stream.append(tail)
        assert appended["needle_tokens"] == LIMIT and appended["result"] == expected([1, 4, 5])
        state = stream.state
        reject(lambda: stream.append(" b"))
        assert stream.state is state and request.calls == calls + 4
        return dict(zip(workloads, (batch["wall_ms"], full["wall_ms"],
                                    initial["total_ms"], appended["total_ms"])))

    reject(lambda: engine.decide(overflow))
    for index in range(5):
        cycle(index)

    samples = {workload: [] for workload in workloads}
    start = time.perf_counter()
    next_report = start + interval
    last_report_index = 0
    cycles = 0
    baseline_calls = request.calls

    def snapshot(status, failure=None):
        report.update(status=status, elapsed_seconds=time.perf_counter() - start, cycles=cycles,
                      openvino_requests=request.calls - baseline_calls, successful_decisions=(len(short) + 3) * cycles,
                      rejected_stream_overflows=cycles, timing_ms={name: quantiles(rows)
                                                                   for name, rows in samples.items() if rows})
        if failure:
            report["failure"] = failure
        save(report)

    snapshot("running")
    try:
        while time.perf_counter() - start < duration:
            for name, time_ms in cycle(cycles).items():
                samples[name].append(time_ms)
            cycles += 1
            now = time.perf_counter()
            if now >= next_report:
                report["intervals"].append({"until_seconds": now - start, "cycles": cycles - last_report_index,
                                            "wall_ms": {name: quantiles(rows[last_report_index:])
                                                        for name, rows in samples.items()}})
                snapshot("running")
                last_report_index = cycles
                next_report = now + interval
    except (Exception, KeyboardInterrupt) as error:
        snapshot("failed", "%s: %s" % (type(error).__name__, error))
        raise
    snapshot("passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=RELEASE)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.interval <= args.duration:
        parser.error("Duration and progress interval must be positive, with interval at most duration")

    identity = windows_usb_identity("ncs1")
    engine = Runtime(args.archive, args.toolchain)
    report = {"model": engine.model_name, "backend": engine.target, "openvino_version": engine.openvino_version,
              "physical_usb_preflight": identity, "onnx_sha256": engine.info["onnx_sha256"],
              "archive": file_details(args.archive), "tokenizer": file_details(TOKENIZER),
              "corpus_sha256": CORPUS_SHA, "duration_seconds": args.duration, "intervals": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(state):
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf8")
        temp.replace(args.output)
        print(json.dumps({key: state[key] for key in ("status", "elapsed_seconds", "cycles", "openvino_requests")}),
              flush=True)

    soak(engine, args.corpus, args.duration, args.interval, report, save)


if __name__ == "__main__":
    main()
