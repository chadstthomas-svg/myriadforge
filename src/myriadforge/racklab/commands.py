"""Command line tools for running and scoring Rack Lab automotive packs."""

from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory

from .journal import RailShot, encode
from .workflow import Automotive, DATA, DEFAULT_PACK, RESULT_SCHEMA, digest, process


def configure(parser):
    commands = parser.add_subparsers(dest="racklab_command", required=True)
    demo = commands.add_parser("demo", help="Replay the bundled authored CPU preview")
    demo.set_defaults(fn=demo_command)
    run = commands.add_parser("run", help="Read local append, approval, hold, and reset JSONL")
    run.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    run.add_argument("--journal", type=Path, required=True)
    run.set_defaults(fn=run_command)
    score = commands.add_parser("evaluate", help="Replay labeled cases and assess the pilot gates")
    score.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--fit", type=Path, help="Previously used fit notes for split validation")
    score.add_argument("--selection", type=Path, help="Previously used selection notes for split validation")
    score.add_argument("--shadow-adapter", type=Path, help="Compare a fitted router without changing decisions")
    score.add_argument("--out", type=Path)
    score.set_defaults(fn=evaluate_command)


def open_journal(path, adapter):
    return closing(RailShot(path, adapter))


def failure(error):
    print(f"Rack Lab: {error}", file=sys.stderr)
    return 2


def demo_command(args):
    try:
        adapter = Automotive()
        fixture = json.loads((DATA / "demo.json").read_bytes())
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "demo.sqlite3"
            with open_journal(filename, adapter) as journal:
                replies = [process(journal, event) for event in fixture["opening"]]
            with open_journal(filename, adapter) as journal:
                resumed = [process(journal, event) for event in fixture["resumed"]]
        if any(not row["ok"] for row in replies + resumed) or replies[-1] != resumed[0]:
            raise ValueError("The preview replay or exact retry failed")
        observed = [row["decision"]["status"] for row in replies + resumed]
        references = [[ref["id"] for ref in row["decision"]["references"]] for row in replies + resumed]
        if observed != fixture["expected_statuses"] or references != [[key] for key in fixture["expected_references"]]:
            raise ValueError("The preview replay disagrees with the authored labels")
        print(json.dumps({
            "scope": "authored CPU preview", "pack_version": adapter.version,
            "opening": replies, "after_restart": resumed, "exact_retry": True,
        }, indent=2, ensure_ascii=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        return failure(error)


def run_command(args):
    try:
        with open_journal(args.journal, Automotive(args.pack)) as journal:
            for line in sys.stdin.buffer:
                try:
                    event = json.loads(line)
                except (ValueError, RecursionError):
                    reply = {"schema": RESULT_SCHEMA, "ok": False, "error": {
                        "code": "invalid_json", "message": "Expected one UTF-8 JSON object per line",
                    }}
                else:
                    reply = process(journal, event)
                print(encode(reply), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        return failure(error)


def case_note(case):
    if "note" in case:
        return case["note"]
    note = ""
    for event in case["events"]:
        if event["event"] == "reset":
            note = ""
        elif event["event"] == "append":
            note += event["text"]
    return note


def split_check(fit, selection, holdout):
    if len(fit) < 30 or len(selection) < 20:
        return False
    from myriadforge.racklab_tuning import check_disjoint, fingerprints

    seen = []
    for cases in (fit, selection, holdout):
        keys = [case["id"] for case in cases]
        groups = [case["scenario_group"] for case in cases]
        if len(set(keys)) != len(keys) or not all(isinstance(v, str) and v.strip() for v in keys + groups):
            raise ValueError("Split IDs and groups must be nonempty, with unique IDs within each split")
        group_keys = fingerprints([{"id": key, "scenario_group": group, "note": case_note(case)}
                                   for key, group, case in zip(keys, groups, cases)])
        group_keys["note"] = [key for key in group_keys["note"] if key != digest(b"")]
        seen.append(group_keys)
    for i, keys in enumerate(seen):
        for other in seen[i + 1:]:
            check_disjoint(keys, other)
    return True


def assess(adapter, cases, fit=(), selection=(), shadow=None):
    if not cases:
        raise ValueError("Provide at least one labeled JSONL case")
    ids = set()
    counts = Counter()
    grouped = {}
    field_groups = set()
    details = []
    for case in cases:
        key, group = case["id"], case["scenario_group"]
        if not all(isinstance(v, str) and v.strip() for v in (key, group)) or key in ids:
            raise ValueError("Cases require unique IDs and nonempty scenario groups")
        ids.add(key)
        expected, trail = case["expected"], case["events"]
        target_ids = expected["references"]
        if not isinstance(trail, list) or not trail:
            raise ValueError(f"Provide at least one event for {key}")
        if (not isinstance(target_ids, list) or not set(target_ids) <= {s["id"] for s in adapter.sources}
                or expected["proposal"] not in (None, "review_service_reference")
                or expected["route"] not in (None, "recall_lookup", "bulletin_lookup")):
            raise ValueError(f"Invalid evaluation labels for {key}")
        if expected["proposal"] and not target_ids:
            raise ValueError(f"Eligible cases must name their required references for {key}")
        provenance = case["provenance"]
        if provenance not in ("authored", "reviewed_operator"):
            raise ValueError(f"Unknown case provenance for {key}")
        if provenance == "reviewed_operator":
            field_groups.add(group)
        with open_journal(":memory:", adapter) as journal:
            for sequence, event in enumerate(trail, 1):
                result = process(journal, {**event, "stream_id": "evaluation", "seq": sequence})
                if not result["ok"]:
                    raise ValueError(f"Invalid event {sequence} in {key}: {result['error']['code']}")
        observed = result["decision"]
        sources = [ref["id"] for ref in observed["references"]]
        covered = set(target_ids) <= set(sources[:3])
        route_match = observed["route"] == expected["route"]
        subject_match = observed["subject"] == expected["subject"]
        matches = observed["proposal"] == expected["proposal"] and route_match and subject_match
        row = Counter({
            "cases": 1, "eligible": int(bool(expected["proposal"])),
            "eligible_exact": int(bool(expected["proposal"]) and matches and covered),
            "blocked": int(expected["proposal"] is None),
            "blocked_proposals": int(expected["proposal"] is None and observed["proposal"] is not None),
            "answerable": int(bool(target_ids)), "all_targets_top3": int(bool(target_ids) and covered),
            "route_matches": int(route_match), "subject_matches": int(subject_match),
        })
        if shadow is not None:
            from myriadforge.racklab_tuning import fitted_route

            predicted = fitted_route(shadow["router"], shadow["catalog"], "automotive", case_note(case), "learned")
            if predicted == "other":
                predicted = None
            row["shadow_route_matches"] = int(predicted == expected["route"])
        counts.update(row)
        grouped.setdefault(expected["route"] or "abstention", Counter()).update(row)
        details.append({"id": key, "status": observed["status"], "route_match": route_match,
                        "subject_match": subject_match, "covered": covered,
                        "proposal_match": observed["proposal"] == expected["proposal"]})
    gates = {
        "reviewed_operators": all(case["provenance"] == "reviewed_operator" for case in cases),
        "sample_floors": counts["eligible"] >= 60 and counts["blocked"] >= 60 and len(field_groups) >= 2,
        "document_types": all(grouped.get(route, {}).get("eligible", 0) > 0
                              for route in ("recall_lookup", "bulletin_lookup")),
        "eligible_accuracy": counts["eligible"] > 0 and counts["eligible_exact"] * 10 >= counts["eligible"] * 9,
        "reference_coverage": counts["answerable"] > 0 and counts["all_targets_top3"] * 20 >= counts["answerable"] * 19,
        "no_blocked_proposals": counts["blocked"] > 0 and not counts["blocked_proposals"],
        "independent_splits": split_check(fit, selection, cases),
    }
    return {"pack_version": adapter.version, "counts": dict(counts), "by_route": grouped,
            "gates": gates, "field_data_gates_passed": all(gates.values()), "cases": details}


def evaluate_command(args):
    try:
        raw = args.cases.read_bytes()
        fit, selection = (read_rows(path.read_bytes()) if path else [] for path in (args.fit, args.selection))
        shadow = json.loads(args.shadow_adapter.read_bytes()) if args.shadow_adapter else None
        if shadow is not None:
            routes = shadow["catalog"]["workflows"]
            if (shadow["schema"] != "racklab.tuning/1" or
                    set(routes) != {"recall_lookup", "bulletin_lookup", "other"} or
                    any(item["market"] != "automotive" for item in routes.values())):
                raise ValueError("Shadow adapter must use the three automotive training routes")
        report = assess(Automotive(args.pack), read_rows(raw), fit, selection, shadow)
        report["cases_sha256"] = digest(raw)
        payload = json.dumps(report, indent=2) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8") as out:
                out.write(payload)
        sys.stdout.write(payload)
        return 0 if report["field_data_gates_passed"] else 1
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        return failure(error)


def read_rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
