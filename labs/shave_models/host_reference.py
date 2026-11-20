"""Host reference for the voice-to-tool slice: Whistle ASR -> Needle 3 -> exact match.

The number to beat. Runs the Cactus host stack (Whistle transcribe -> Needle 3
tool extraction) over the frozen voicetools fixtures and reports full-call
exact match, false-action rate on negatives, and latency percentiles - the
same fixtures, hence condition-matched, that the on-device model must beat
(see tools_spec.ACCEPTANCE). A --no-asr oracle mode runs Needle on the
manifest text directly to isolate ASR errors from extraction errors.

Needs the NTVStudy main env (cactus-compute 4.2.6 lives there):
  cd D:/NTVStudy && uv run python D:/MyriadForge/labs/shave_models/host_reference.py \
      --fixtures D:/models/ncs-shave-research/voicetools [--split test] [--limit 20]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import cactus

from tools_spec import TOOLS


def tool_schema():
    """The 12 benign tools in the Cactus tool-schema format."""
    spec = []
    for name, slots in sorted(TOOLS.items()):
        props, required = {}, []
        for slot, values in slots.items():
            if slot == "seconds":
                props["seconds"] = {"type": "integer", "minimum": 0}
            elif slot == "level":
                props["level"] = {"type": "integer", "minimum": 0, "maximum": 100}
            else:
                props[slot] = {"type": "string", "enum": sorted(values)}
            required.append(slot)
        spec.append({"name": name, "parameters": {"type": "object", "properties": props,
                                                  "required": required}})
    return spec


def normalize(call):
    """One call -> {"name": str, "args": {slot: canonical value}} for exact comparison."""
    name = str(call.get("name", "")).strip()
    args = {}
    for key, value in (call.get("arguments") or {}).items():
        if key in ("level", "seconds"):
            try:
                args[key] = int(round(float(value)))
            except (TypeError, ValueError):
                args[key] = value  # leave the malformed value; exact match will fail honestly
        else:
            args[key] = str(value).strip().casefold()
    return {"name": name, "args": args}


def normalize_manifest(call):
    args = {}
    for key, value in call.items():
        if key == "tool":
            continue
        if key in ("level", "seconds"):
            args[key] = int(value)
        else:
            args[key] = str(value).casefold()
    return {"name": call["tool"], "args": args}


def p95(values):
    if not values:
        return 0.0
    return sorted(values)[max(0, int(round(0.95 * len(values))) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--limit", type=int, default=0, help="cap fixtures (smoke runs)")
    ap.add_argument("--no-asr", action="store_true", help="oracle: Needle on manifest text")
    ap.add_argument("--out", default="",
                    help="report path (default fixtures/host-reference-resident.json)")
    args = ap.parse_args()

    manifest_path = args.fixtures / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} fixtures, split={args.split}, asr={'off (oracle)' if args.no_asr else 'whistle'}",
          flush=True)

    tools = tool_schema()
    report = {"fixtures": str(args.fixtures), "split": args.split, "asr": not args.no_asr,
              "n": len(rows), "rows": [],
              "timing_boundary": ("resident Whistle: asr_ms covers transcribe() only, "
                                  "extract_ms covers complete() only; model loads are "
                                  "outside both (prior per-fixture Whistle() construction "
                                  "inflated ASR latency and is superseded by this run)")}
    asr_ms, extract_ms = [], []
    false_on_neg = 0
    negatives = 0

    asr = None if args.no_asr else cactus.Whistle()  # resident - load once, time transcribe only
    with cactus.Cactus(tools=tools) as agent:
        for i, row in enumerate(rows):
            agent.reset()  # independent query per fixture (evaluate_needle pattern)
            if args.no_asr:
                text, t_asr = row["text"], 0.0
            else:
                t0 = time.perf_counter()
                heard = asr.transcribe(str(args.fixtures / row["wav"]))
                t_asr = (time.perf_counter() - t0) * 1000
                text = heard.get("text", "") if isinstance(heard, dict) else str(heard)
                asr_ms.append(t_asr)
            t0 = time.perf_counter()
            response = agent.complete(text)
            t_extract = (time.perf_counter() - t0) * 1000
            extract_ms.append(t_extract)
            got = [normalize(c) for c in response.get("function_calls", [])]
            want = [normalize_manifest(c) for c in row["calls"]]
            exact = got == want
            is_neg = row["kind"].startswith("neg")
            if is_neg:
                negatives += 1
                if got:
                    false_on_neg += 1
            report["rows"].append({"id": row["id"], "kind": row["kind"],
                                   "heard": text if not args.no_asr else "",
                                   "got": got, "want": want, "exact": exact,
                                   "asr_ms": round(t_asr, 1), "extract_ms": round(t_extract, 1)})
            if (i + 1) % 25 == 0:
                hits = sum(1 for r in report["rows"] if r["exact"])
                print(f"  {i + 1}/{len(rows)} exact {hits}/{i + 1}", flush=True)

    pos_rows = [r for r in report["rows"] if not r["kind"].startswith("neg")]
    hits = sum(1 for r in pos_rows if r["exact"])
    report["metrics"] = {
        "exact_match": round(hits / max(1, len(pos_rows)), 4),
        "exact_hits": hits, "positives": len(pos_rows),
        "false_action_rate": round(false_on_neg / max(1, negatives), 4),
        "negatives": negatives, "false_actions": false_on_neg,
        "p95_asr_ms": round(p95(asr_ms), 1), "p95_extract_ms": round(p95(extract_ms), 1),
        "p95_total_ms": round(p95([r["asr_ms"] + r["extract_ms"] for r in report["rows"]]), 1),
        "median_total_ms": round(statistics.median([r["asr_ms"] + r["extract_ms"]
                                                    for r in report["rows"]]), 1),
    }
    out = Path(args.out) if args.out else args.fixtures / "host-reference-resident.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
