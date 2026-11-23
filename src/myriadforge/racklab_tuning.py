"""Fit and measure compact Rack Lab routing and source-ranking adapters."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import log, sqrt
from pathlib import Path
import re

import numpy as np

SCHEMA = "racklab.tuning/1"
WORD = re.compile(r"[^\W_]+")
MIN_MARGIN = 0.12


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cases(path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return read(path)


def write(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def words(text):
    return WORD.findall(text.casefold())


def terms(text):
    items = words(text)
    return items + [" ".join(pair) for pair in zip(items, items[1:])]


def catalog(pack):
    workflows = {}
    markets = {}
    sources = pack["sources"]
    for workflow in pack["workflows"]:
        route, market = workflow["id"], workflow["market"]
        if route in workflows:
            raise ValueError(f"Repeated route in pack: {route}")
        workflows[route] = {"market": market, "keywords": workflow["keywords"]}
        accessible = markets.setdefault(market, set())
        accessible.update(workflow["target"])
        for targets in workflow.get("target_by_group", {}).values():
            accessible.update(targets)
    for market, refs in markets.items():
        if not refs or not refs <= sources.keys():
            raise ValueError(f"Invalid source catalog for {market}")
    return {
        "workflows": workflows, "markets": {key: sorted(value) for key, value in markets.items()},
        "sources": {key: sources[key] for key in sorted(set().union(*markets.values()))},
    }


def normalized_cases(items, index):
    if not isinstance(items, list) or not items:
        raise ValueError("Provide a nonempty list of case records")
    rows = []
    ids = set()
    for item in items:
        if not isinstance(item, dict) or ("route" not in item and "expected_route" not in item):
            raise ValueError("Each case needs an explicit route label or null")
        route = item.get("route", item.get("expected_route"))
        market = item["market"]
        note, group, key = item["note"], item["scenario_group"], item["id"]
        targets = item["targets"]
        if not all(isinstance(value, str) and value.strip() for value in (note, group, key)):
            raise ValueError("Cases need a note, id, and scenario_group")
        if key in ids:
            raise ValueError(f"Repeated case id: {key}")
        if market not in index["markets"]:
            raise ValueError(f"Unknown market: {market}")
        if route is not None and index["workflows"].get(route, {}).get("market") != market:
            raise ValueError(f"Invalid route for {key}")
        if not isinstance(targets, list) or not set(targets) <= set(index["markets"][market]):
            raise ValueError(f"Case targets are outside the market catalog: {key}")
        ids.add(key)
        rows.append({"id": key, "scenario_group": group, "market": market, "note": note,
                     "route": route, "targets": targets})
    return rows


def fingerprints(rows):
    keys = {"id": set(), "group": set(), "note": set()}
    for row in rows:
        for kind, value in (("id", row["id"]), ("group", row["scenario_group"]),
                            ("note", " ".join(words(row["note"])))):
            keys[kind].add(sha256(value.encode()).hexdigest())
    return {kind: sorted(values) for kind, values in keys.items()}


def check_disjoint(left, right):
    for kind in left:
        if set(left[kind]) & set(right[kind]):
            raise ValueError(f"Split overlap detected for {kind}")


def keyword_route(index, market, text):
    query = set(words(text))
    scored = sorted((sum(set(words(keyword)) <= query for keyword in spec["keywords"]), route)
                    for route, spec in index["workflows"].items() if spec["market"] == market)
    if scored and scored[-1][0] and (len(scored) == 1 or scored[-1][0] > scored[-2][0]):
        return scored[-1][1]
    return None


def fitted_route(router, index, market, text, mode):
    counts = Counter(terms(text))
    weight = np.zeros(len(router["terms"]), dtype=np.float32)
    for term, number in counts.items():
        position = router["terms"].get(term)
        if position is not None:
            weight[position] = (1 + log(number)) * router["idf"][position]
    size = np.linalg.norm(weight)
    if size == 0:
        return None
    weight /= size
    if mode == "int8":
        matrix = np.asarray(router["int8"], dtype=np.int8).astype(np.float32) * router["scale"]
    else:
        matrix = np.asarray(router["weights"], dtype=np.float32)
    bias = np.asarray(router["bias"], dtype=np.float32)
    scores = matrix @ weight + bias
    allowed = sorted((i for i, label in enumerate(router["labels"])
                      if index["workflows"][label]["market"] == market),
                     key=lambda i: (-scores[i], i))
    if not allowed or (len(allowed) > 1 and scores[allowed[0]] - scores[allowed[1]] < MIN_MARGIN):
        return None
    return router["labels"][allowed[0]]


def chargrams(text):
    padded = [" " + token + " " for token in words(text)]
    return {piece[at:at + 3] for piece in padded for at in range(len(piece) - 2)}


def retrieval_features(index, market, text):
    query, grams = set(words(text)), chargrams(text)
    identifiers, results = index["markets"][market], []
    for key in identifiers:
        title, _, description = index["sources"][key]
        headings, body = set(words(title)), set(words(description))
        docgrams = chargrams(title + " " + description)
        cosine = len(grams & docgrams) / sqrt(max(1, len(grams) * len(docgrams)))
        results.append((len(query & headings), len(query & body),
                        len(query & (headings | body)) / max(1, len(query)), cosine))
    return identifiers, np.asarray(results, dtype=np.float32)


def rank(model, market, text, mode):
    identifiers, values = retrieval_features(model["catalog"], market, text)
    fitted = model["ranker"]
    if mode == "learned" and fitted is not None:
        scores = values @ np.asarray(fitted["weights"], dtype=np.float32) + fitted["bias"]
        present = values[:, 3] > 0
    else:
        scores = values[:, 0] * 2 + values[:, 1]
        present = scores > 0
    order = sorted((i for i, keep in enumerate(present) if keep), key=lambda i: (-scores[i], identifiers[i]))
    return [identifiers[position] for position in order[:3]]


def metrics(model, rows):
    index, router = model["catalog"], model["router"]
    totals = Counter()
    for row in rows:
        market, text, label = row["market"], row["note"], row["route"]
        baseline = keyword_route(index, market, text)
        learned = fitted_route(router, index, market, text, "fp32")
        quantized = fitted_route(router, index, market, text, "int8")
        totals["keyword_routes"] += baseline == label
        totals["learned_routes"] += learned == label
        totals["int8_routes"] += quantized == label
        totals["int8_agreement"] += quantized == learned
        target = set(row["targets"])
        for choice in ("lexical", "learned"):
            ranked = rank(model, market, text, choice)
            totals[choice + "_top3"] += target <= set(ranked) if target else not ranked
    return {"cases": len(rows), **dict(totals)}


def fit(pack, training, validation):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise ValueError('Install the racklab-tuning extra before fitting: pip install -e ".[racklab-tuning]"') from error

    index = catalog(pack)
    training, validation = (normalized_cases(rows, index) for rows in (training, validation))
    fit_keys, selection_keys = fingerprints(training), fingerprints(validation)
    check_disjoint(fit_keys, selection_keys)
    labeled = {row["route"] for row in training if row["route"] is not None}
    if len(labeled) < 2 or labeled != index["workflows"].keys():
        raise ValueError("Training must cover all declared routes, with at least two routes")
    # Route labels can repeat for several policy states of one authored note.
    notes = {(row["market"], row["note"], row["route"]): row for row in training if row["route"] is not None}
    vector = TfidfVectorizer(analyzer=terms, sublinear_tf=True, min_df=1, dtype=np.float32)
    classifier = LogisticRegression(C=5, random_state=17).fit(
        vector.fit_transform(row["note"] for row in notes.values()), [row["route"] for row in notes.values()]
    )
    weights = classifier.coef_.astype(np.float32)
    bias = classifier.intercept_.astype(np.float32)
    labels = classifier.classes_.tolist()
    if len(labels) == 2:
        weights, bias = np.concatenate((-weights, weights)), np.concatenate((-bias, bias))
    scale = max(float(np.abs(weights).max()) / 127, 1e-12)
    integer = np.rint(weights / scale).astype(np.int8)
    router = {
        "terms": {key: int(value) for key, value in vector.vocabulary_.items()},
        "idf": vector.idf_.tolist(), "labels": labels, "weights": weights.tolist(),
        "bias": bias.tolist(), "int8": integer.tolist(), "scale": scale,
    }
    values, labels = [], []
    distinct = {(row["market"], row["note"], tuple(row["targets"])): row for row in training}
    for row in distinct.values():
        identifiers, features = retrieval_features(index, row["market"], row["note"])
        values.extend(features)
        labels.extend(key in row["targets"] for key in identifiers)
    ranker = None
    if len(set(labels)) > 1:
        scorer = LogisticRegression(C=1, random_state=17).fit(values, labels)
        ranker = {"weights": scorer.coef_[0].astype(np.float32).tolist(), "bias": float(scorer.intercept_[0])}
    model = {
        "schema": SCHEMA, "catalog": index, "router": router, "ranker": ranker,
        "seen": {key: sorted(set(fit_keys[key]) | set(selection_keys[key])) for key in fit_keys},
        "weights": {"router_fp32_bytes": weights.nbytes, "router_int8_bytes": integer.nbytes},
    }
    results = metrics(model, validation)
    model["selection"] = {
        **results, "route": "learned" if results["learned_routes"] > results["keyword_routes"] else "keywords",
        "rank": "learned" if results["learned_top3"] > results["lexical_top3"] else "lexical",
    }
    return model


def evaluate(model, inputs):
    rows = normalized_cases(inputs, model["catalog"])
    check_disjoint(model["seen"], fingerprints(rows))
    output = metrics(model, rows)
    return {**output, "by_market": {market: metrics(model, [row for row in rows if row["market"] == market])
                                    for market in sorted({row["market"] for row in rows})}}


def infer(model, market, text, route_mode, rank_mode):
    index = model["catalog"]
    if market not in index["markets"]:
        raise ValueError(f"Unknown market: {market}")
    if route_mode == "selected":
        route_mode = model["selection"]["route"]
    if rank_mode == "selected":
        rank_mode = model["selection"]["rank"]
    route = keyword_route(index, market, text) if route_mode == "keywords" else fitted_route(
        model["router"], index, market, text, route_mode,
    )
    references = [{"id": key, "title": index["sources"][key][0], "url": index["sources"][key][1]}
                  for key in rank(model, market, text, rank_mode)]
    return {"suggested_route": route, "references": references, "requires_policy_review": True,
            "route_profile": route_mode, "rank_profile": rank_mode}


def demo(pack, output):
    split = [[], [], []]
    for entry in pack["workflows"]:
        if len(entry["notes"]) != 8:
            raise ValueError("The demo pack requires eight authored notes per workflow")
        for number, note in enumerate(entry["notes"], 1):
            key = f"{entry['id']}-{number}"
            split[0 if number <= 4 else 1 if number == 5 else 2].append({
                "id": key, "scenario_group": key, "market": entry["market"], "note": note, "route": entry["id"],
                "targets": entry.get("target_by_group", {}).get(str(number), entry["target"]),
            })
    output.mkdir(parents=True)
    for name, rows in zip(("fit", "selection", "evaluation"), split):
        write(output / f"{name}.json", rows)
    model = fit(pack, split[0], split[1])
    write(output / "adapter.json", model)
    report = evaluate(model, split[2])
    write(output / "report.json", report)
    return {"selection": model["selection"], "evaluation": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    for action in ("demo", "fit"):
        sub = command.add_parser(action)
        sub.add_argument("--pack", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        if action == "fit":
            sub.add_argument("--train", type=Path, required=True)
            sub.add_argument("--select", type=Path, required=True)
    for action in ("evaluate", "predict"):
        sub = command.add_parser(action)
        sub.add_argument("--model", type=Path, required=True)
        if action == "evaluate":
            sub.add_argument("--cases", type=Path, required=True)
        else:
            sub.add_argument("--market", required=True)
            sub.add_argument("--text", required=True)
            sub.add_argument("--route", choices=("selected", "keywords", "learned", "int8"), default="selected")
            sub.add_argument("--rank", choices=("selected", "lexical", "learned"), default="selected")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            result = demo(read(args.pack), args.out)
        elif args.command == "fit":
            model = fit(read(args.pack), cases(args.train), cases(args.select))
            write(args.out, model)
            result = model["selection"]
        else:
            model = read(args.model)
            if model.get("schema") != SCHEMA:
                raise ValueError("Unsupported adapter schema")
            if args.command == "evaluate":
                result = evaluate(model, cases(args.cases))
            else:
                result = infer(model, args.market, args.text, args.route, args.rank)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
