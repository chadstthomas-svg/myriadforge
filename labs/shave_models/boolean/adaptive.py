"""Learn exact preparation lanes by vertical and document structure behind each batch."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

from policy import SegmentMemo, needle_count, tokenizer_prior
from runtime import Runtime, json_output_stream
from supervisor import supervise


REFERENCE = "anchored"
LANES = (REFERENCE, "segmented", "cached")
EXPLORATION_ORDER = ("cached", "segmented", REFERENCE)
PROBE_INTERVAL = 8
COOLDOWN = 16


def profile_key(text):
    prefix = text[:512].lstrip()
    if prefix.startswith(("{", "[")):
        structure = "json"
    elif prefix.startswith("<"):
        structure = "markup"
    elif "\t" in prefix and "\n" in prefix:
        structure = "rows"
    else:
        structure = "text"
    length = len(text)
    size = "short" if length < 4096 else "medium" if length < 100000 else "long"
    return structure, size


def score_decision(result, expected):
    correct = {key: int(result[key] == expected[key]) for key in ("answers", "category", "action")}
    return dict(correct, labeled=1, joint=int(all(correct.values())),
                proposed=int(result["action"] is not None),
                correct_proposal=int(result["action"] is not None and correct["action"]))


def probe_excerpt(text):
    if len(text) <= 16384:
        return text
    return text[:8192] + " " + text[-8192:]


class Profiles:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS lanes (profile TEXT PRIMARY KEY, state TEXT NOT NULL)"
        )

    def get(self, key):
        record = self.db.execute("SELECT state FROM lanes WHERE profile = ?", (key,)).fetchone()
        return json.loads(record[0]) if record else {"batches": 0, "ms": {}, "retry_at": {}}

    def save(self, profiles):
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO lanes (profile, state) VALUES (?, ?)",
                [(key, json.dumps(state)) for key, state in profiles.items()],
            )

    def close(self):
        self.db.close()


class BatchEvidence:
    """A constrained latency bandit with an independent evidence reference."""

    def __init__(self, engine, state, memo):
        self.engine = engine
        self.state = state
        self.memo = memo
        self.used = {}
        self.failures = {}
        self.probe = state["batches"] % PROBE_INTERVAL == 0
        self.audited = False

    def measure(self, lane, text, evidence_text):
        start = time.perf_counter()
        memo = self.memo if lane == "cached" else None
        tokens = needle_count(self.engine.tokenizer, text, memo=memo)
        counted = time.perf_counter()
        policy = self.engine.policy
        atoms = (policy.anchored_atoms(evidence_text) if lane == REFERENCE
                 else policy.text_atoms(evidence_text))
        return (tokens, atoms, counted), (time.perf_counter() - start) * 1000

    def observe(self, lane, elapsed):
        score = self.state["ms"]
        score[lane] = 0.75 * score.get(lane, elapsed) + 0.25 * elapsed

    def quarantine(self, lane, reason):
        self.failures[lane] = reason
        self.state["ms"].pop(lane, None)
        self.state["retry_at"][lane] = self.state["batches"] + COOLDOWN
        if lane == "cached":
            self.memo.counts.clear()

    def prepare(self, text, evidence_text=None):
        evidence_text = text if evidence_text is None else evidence_text
        scores = self.state["ms"]
        checking = self.probe
        if checking:
            self.probe = False
            probe, sample = probe_excerpt(text), probe_excerpt(evidence_text)
            reference, _ = self.measure(REFERENCE, probe, sample)
            eligible = {REFERENCE}
            for lane in LANES[1:]:
                if self.state["retry_at"].get(lane, 0) > self.state["batches"]:
                    continue
                try:
                    prepared, _ = self.measure(lane, probe, sample)
                    if prepared[0] != reference[0] or not np.array_equal(prepared[1], reference[1]):
                        self.quarantine(lane, "evidence or token count disagreed with the reference")
                    else:
                        eligible.add(lane)
                        self.state["retry_at"].pop(lane, None)
                except (RuntimeError, ValueError):
                    self.quarantine(lane, "preparation failed")
            options = [lane for lane in EXPLORATION_ORDER if lane in eligible]
            next_lane = next((lane for lane in options if lane not in scores), None)
            rotation = (self.state["batches"] // PROBE_INTERVAL) % len(options)
            selected = next_lane if next_lane is not None else options[rotation]
        else:
            eligible = [lane for lane in LANES if lane in scores]
            selected = min(eligible, key=scores.__getitem__) if eligible else REFERENCE
        try:
            result, elapsed = self.measure(selected, text, evidence_text)
        except (RuntimeError, ValueError):
            if selected == REFERENCE:
                raise
            self.quarantine(selected, "preparation failed")
            selected = REFERENCE
            result, elapsed = self.measure(selected, text, evidence_text)
        if checking and selected != REFERENCE:
            policy = self.engine.policy
            reference_atoms = policy.anchored_atoms(evidence_text)
            if not np.array_equal(result[1], reference_atoms):
                self.quarantine(selected, "full-document evidence disagreed with the reference")
                selected = REFERENCE
                result, elapsed = self.measure(selected, text, evidence_text)
        self.observe(selected, elapsed)
        self.used[selected] = self.used.get(selected, 0) + 1
        return result

class AdaptiveRuntime:
    """Use the requested policy on the physical stick and retry a failed device session once."""

    def __init__(self, factories, profiles, scopes=None, banks=None):
        self.factories = factories
        self.profiles = profiles
        self.scopes = scopes or {}
        self.banks = banks or {}
        self.engines = {}
        self.memos = {}
        self.prior = None

    def engine(self, vertical):
        factory = self.factories[vertical]
        if factory not in self.engines:
            engine = factory()
            if engine.info["variant"] == "scan":
                raise ValueError("Adaptive mode needs a bundle that accepts fact bits")
            self.engines[factory] = engine
        engine = self.engines[factory]
        bank = self.banks.get(vertical)
        return engine if bank is None else engine.policy_view(bank)

    def batch(self, request, explain=False):
        if not isinstance(request, dict):
            raise ValueError("Expected a batch object")
        vertical = request.get("vertical")
        if not isinstance(vertical, str) or vertical not in self.factories:
            raise ValueError("Choose a configured vertical")
        rows = request.get("documents")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Expected a nonempty documents array")
        if any(not isinstance(row, dict) or not isinstance(row.get("text"), str) for row in rows):
            raise ValueError("Each document needs a text string")
        for row in rows:
            if "expected" not in row:
                continue
            label = row["expected"]
            if (not isinstance(label, dict) or set(label) != {"answers", "category", "action"}
                    or not isinstance(label["answers"], dict)
                    or any(type(value) is not bool for value in label["answers"].values())
                    or any(label[key] is not None and not isinstance(label[key], str)
                           for key in ("category", "action"))):
                raise ValueError("Expected feedback with Boolean answers, a category and an action")
        engine = self.engine(vertical)
        signature = (engine.info["policy"], engine.info.get("onnx_sha256"))
        policy_hash = hashlib.sha256(json.dumps(signature[0]).encode("utf8")).hexdigest()
        if self.prior is None:
            self.prior = tokenizer_prior(engine.tokenizer)
        memo = self.memos.setdefault(vertical, SegmentMemo(prior=self.prior))
        scope = self.scopes.get(vertical)
        contexts, outputs = {}, [None] * len(rows)
        pending = []
        device_requests = 0

        def flush():
            nonlocal engine, device_requests
            if not pending:
                return
            prepared = [job[3] for job in pending]

            def infer():
                nonlocal device_requests
                device_requests += 1
                results = engine.decide_prepared(prepared)
                if len(results) != len(pending):
                    raise RuntimeError("Device returned an unexpected number of decisions")
                for (_, evidence, _, _), result in zip(pending, results):
                    if evidence.audited or evidence.state["batches"] % PROBE_INTERVAL != 0:
                        continue
                    reference = engine.policy.reference(result["atoms"])
                    if not np.array_equal(reference, result["bits"]):
                        raise RuntimeError("Device disagreed with the Boolean reference")
                    evidence.audited = True
                return results

            recovered = False
            try:
                results = infer()
            except RuntimeError:
                recovered = True
            if recovered:
                self.engines.pop(self.factories[vertical], None)
                for _, context in contexts.values():
                    context.engine = None
                    context.audited = False
                del engine
                engine = self.engine(vertical)
                if (engine.info["policy"], engine.info.get("onnx_sha256")) != signature:
                    raise RuntimeError("Model changed during recovery")
                for _, context in contexts.values():
                    context.engine = engine
                results = infer()
            for (index, evidence, evidence_text, _), result in zip(pending, results):
                row = rows[index]
                if explain:
                    result["explanation"] = engine.policy.explain(evidence_text, result["atoms"])
                if "expected" in row:
                    scores = score_decision(result["result"], row["expected"])
                    totals = evidence.state.setdefault("feedback", {})
                    for name, score in scores.items():
                        totals[name] = totals.get(name, 0) + score
                    result["feedback"] = scores
                outputs[index] = dict(result, id=row.get("id"), device_reloaded=recovered)
            pending.clear()

        for index, row in enumerate(rows):
            text = row["text"]
            shape = profile_key(text)
            if shape not in contexts:
                key = json.dumps(["v2", engine.lane, vertical, policy_hash, scope, *shape])
                contexts[shape] = key, BatchEvidence(engine, self.profiles.get(key), memo)
            _, evidence = contexts[shape]
            try:
                evidence_text = text
                if scope is not None:
                    document = json.loads(text)
                    if not isinstance(document, dict) or not isinstance(document.get(scope), str):
                        raise ValueError("Expected a string in JSON evidence field %r" % scope)
                    evidence_text = document[scope]
                prepare = partial(evidence.prepare, evidence_text=evidence_text)
                prepared = engine.prepare_decision(text, prepare=prepare)
            except ValueError as exc:
                outputs[index] = {"id": row.get("id"), "error": str(exc)}
                continue
            pending.append((index, evidence, evidence_text, prepared))
            if len(pending) == engine.batch_capacity:
                flush()
        flush()
        updates, routes = {}, []
        for (structure, size), (key, evidence) in contexts.items():
            if evidence.used:
                evidence.state["batches"] += 1
                updates[key] = evidence.state
            routes.append(dict(structure=structure, size=size, used=evidence.used,
                               quarantined=evidence.failures, feedback=evidence.state.get("feedback", {})))
        self.profiles.save(updates)
        return {"id": request.get("id"), "vertical": vertical, "routes": routes,
                "documents": outputs, "device_requests": device_requests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON object mapping vertical names to bundle and optional blob paths")
    parser.add_argument("--state", type=Path, help="Retain only per-profile timing and health across runs")
    parser.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs1")
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--release-stick", choices=["ncs1", "ncs2"], help="Physical target for a .m8b archive")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--batch-timeout", type=float, help="Worker deadline in seconds, including cold model load")
    args = parser.parse_args()
    if args.batch_timeout is not None:
        if not math.isfinite(args.batch_timeout) or args.batch_timeout <= 0:
            parser.error("--batch-timeout must be finite and positive")
        command = [sys.executable, str(Path(__file__).resolve()), str(args.config), "--lane", args.lane]
        for key in ("state", "toolchain", "target"):
            value = getattr(args, key)
            if value is not None:
                command.extend(("--" + key, str(value)))
        if args.release_stick:
            command.extend(("--release-stick", args.release_stick))
        if args.explain:
            command.append("--explain")
        with json_output_stream() as output:
            supervise(command, sys.stdin.buffer, output, args.batch_timeout)
        return
    config = json.loads(args.config.read_bytes())
    if not isinstance(config, dict) or not config:
        parser.error("Config must map vertical names to bundle objects")
    for name, entry in config.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("bundle"), str):
            parser.error("Vertical %r needs a bundle path" % name)
        for field in ("blob", "evidence_field", "bank"):
            if field in entry and not isinstance(entry[field], str):
                parser.error("Vertical %r requires a string for %s" % (name, field))

    def load(bundle, blob):
        return Runtime(bundle, args.toolchain, lane=args.lane, target=args.target, blob=blob,
                       release_stick=args.release_stick or "ncs1")

    root = args.config.resolve().parent
    by_bundle, factories = {}, {}
    for name, entry in config.items():
        bundle = (root / entry["bundle"]).resolve()
        blob = (root / entry["blob"]).resolve() if "blob" in entry else None
        factories[name] = by_bundle.setdefault((bundle, blob), partial(load, bundle, blob))
    profiles = Profiles(args.state or ":memory:")
    try:
        with json_output_stream() as output:
            runtime = AdaptiveRuntime(factories, profiles,
                                      {name: entry.get("evidence_field") for name, entry in config.items()},
                                      {name: entry.get("bank") for name, entry in config.items()})
            for line in sys.stdin.buffer:
                request = None
                try:
                    request = json.loads(line)
                    response = runtime.batch(request, explain=args.explain)
                except (ValueError, RuntimeError) as exc:
                    response = {"error": str(exc), "id": request.get("id") if isinstance(request, dict) else None}
                print(json.dumps(response), file=output, flush=True)
    finally:
        profiles.close()


if __name__ == "__main__":
    main()
