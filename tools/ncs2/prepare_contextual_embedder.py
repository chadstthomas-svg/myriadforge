"""Build article-disjoint retrieval pairs and cache the full teacher embeddings."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from apps.embeddinggemma2.__main__ import Encoder, prompt
from apps.static_embedder import TOKEN_LIMIT, WORDS

SEED = 20261124
WORKFLOWS = {
    "automotive": (
        ("wheel inspection", "repair authorization"), ("paint inspection", "bodywork clearance"),
        ("diagnostic review", "service sign-off"),
        ("recall verification", "vehicle dispatch"), ("parts reconciliation", "closeout approval")),
    "manufacturing": (
        ("lot inspection", "shipment authorization"), ("tool audit", "line clearance"),
        ("sample review", "packaging sign-off"),
        ("calibration verification", "batch release"), ("material reconciliation", "order closure")),
    "retail": (
        ("stock inspection", "shelf authorization"), ("till audit", "cashout clearance"),
        ("label review", "price-change sign-off"),
        ("return verification", "refund processing"), ("delivery reconciliation", "receiving closure")),
}

# Each split has separate workflows and paraphrase templates. Pairs are authored
# operational examples, with no external documents or customer information.
FRAMES = {
    "train": {
        "precedence": ("What comes first, {a} or {b}, for {item}?",
                       "For {item}, {a} must come before {b}.", "For {item}, {b} must come before {a}."),
        "negation": ("Find the instruction that forbids {b} for {item} without {a}.",
                     "Do not proceed with {b} for {item} if {a} is missing.",
                     "Do proceed with {b} for {item} if {a} is missing."),
        "state": ("Find the current record where {a} is pending for {item}.",
                  "{item}: earlier {a} was completed. Now {a} is pending.",
                  "{item}: earlier {a} was pending. Now {a} is completed."),
    },
    "dev": {
        "precedence": ("Look up the sequence where {a} happens ahead of {b} for {item}.",
                       "{item}: finish {a} then begin {b}.", "{item}: finish {b} then begin {a}."),
        "negation": ("Which rule for {item} blocks {b} until there is {a}?",
                     "{item}: with no {a}, {b} is prohibited.", "{item}: with no {a}, {b} is allowed."),
        "state": ("Which {item} version leaves {a} unresolved?",
                  "{item} changed from resolved to unresolved for {a}.",
                  "{item} changed from unresolved to resolved for {a}."),
    },
    "test": {
        "precedence": ("Locate the procedure that puts {a} ahead of {b} for {item}.",
                       "The procedure for {item} schedules {b} after {a}.",
                       "The procedure for {item} schedules {a} after {b}."),
        "negation": ("Find the policy for {item} where absent {a} rules out {b}.",
                     "{item}: when {a} is absent, {b} is never permitted.",
                     "{item}: when {a} is absent, {b} is still permitted."),
        "state": ("Which entry describes {a} as back open for {item}?",
                  "Latest record for {item}: {a} reopened after it had been closed.",
                  "Latest record for {item}: {a} closed after it had been reopened."),
    },
}


def key(text):
    return sha256(text.encode("utf8")).digest()


def terms(text):
    return set(WORDS.findall(text.lower()))


def industry_pairs(split):
    cases = []
    for vertical, workflows in WORKFLOWS.items():
        relevant = {"train": workflows[:2], "dev": workflows[2:3], "test": workflows[3:]}[split]
        count = {"train": 24, "dev": 12, "test": 20}[split]
        for category, frames in FRAMES[split].items():
            for i in range(count):
                a, b = relevant[i % len(relevant)]
                identifier = f"{split}-{vertical}-{category}-{i:02}"
                item = f"{vertical} record {split[0].upper()}{i + 101}"
                q, p, n = [part.format(a=a, b=b, item=item) for part in frames]
                cases.append({"id": identifier, "query": q, "positive": p, "negative": n,
                              "vertical": vertical, "phenomenon": category,
                              "family": f"{vertical}-{category}-{a}"})
    return cases


def squad_pairs(path, tokenizer):
    articles = json.loads(path.read_text(encoding="utf8"))["data"]
    candidates = {"train": [], "dev": []}
    for article in articles:
        title = article["title"]
        split = "dev" if int.from_bytes(key(title)[:4], "big") % 10 == 0 else "train"
        for paragraph in article["paragraphs"]:
            passage = paragraph["context"]
            tokens = tokenizer.encode(passage, add_special_tokens=False)
            end = tokens.offsets[-1][1] if tokens.ids else 0
            for entry in paragraph["qas"]:
                answer = entry["answers"][0]
                if answer["answer_start"] + len(answer["text"]) <= end:
                    candidates[split].append((entry["id"], entry["question"], passage, title, answer["text"]))

    pairs = {}
    for split, limit in (("train", 8192), ("dev", 512)):
        ordered = sorted(candidates[split], key=lambda row: key(row[0]))
        pool = dict.fromkeys((r[2], r[3]) for r in ordered[:limit * 2])
        documents = list(pool)
        inverted = defaultdict(list)
        for i, (text, _) in enumerate(documents):
            for token in terms(text):
                inverted[token].append(i)
        result = []
        for identifier, question, positive, title, answer in ordered:
            overlap = Counter()
            for word in sorted(terms(question) & inverted.keys(), key=lambda w: (len(inverted[w]), w))[:8]:
                overlap.update(inverted[word][:256])
            ranked = sorted(overlap, key=lambda i: (-overlap[i], i))
            answer = answer.casefold()
            for index in ranked:
                negative, other_title = documents[index]
                if other_title != title and negative != positive and answer not in negative.casefold():
                    result.append({"id": identifier, "query": question, "positive": positive,
                                   "negative": negative, "positive_title": title,
                                   "negative_title": other_title})
                    break
            if len(result) == limit:
                break
        if len(result) != limit:
            raise ValueError(f"Only {len(result)} eligible {split} SQuAD pairs")
        pairs[split] = result
    return pairs


def public_fixture(root):
    corpus = pq.read_table(root / "nfc-corpus.parquet").to_pylist()
    query_text = {item["_id"]: item["text"]
                  for item in pq.read_table(root / "nfc-queries.parquet").to_pylist()}
    with (root / "nfc-test.tsv").open(encoding="utf8", newline="") as stream:
        labels = list(csv.DictReader(stream, delimiter="\t"))
    ids = sorted({row["query-id"] for row in labels})
    documents = [{"id": row["_id"], "kind": "document", "text": row["text"], "title": row["title"]}
                 for row in corpus]
    queries = [{"id": key, "kind": "query", "text": query_text[key]} for key in ids]
    if len(documents) != 3633 or len(queries) != 323:
        raise ValueError("Expected the full BEIR NFCorpus corpus and test queries")
    return {"rows": documents + queries, "qrels": labels}


def pair_prompts(pair):
    return (prompt({"kind": "query", "text": pair["query"]}),
            prompt({"text": pair["positive"], "title": pair.get("positive_title")}),
            prompt({"text": pair["negative"], "title": pair.get("negative_title")}))


def encode_teacher(encoder, texts, device, batch, path, teacher_revision):
    torch = encoder.torch
    encoded = encoder.tokenizer(texts, padding=False, truncation=True, max_length=TOKEN_LIMIT)
    signature = sha256(json.dumps([teacher_revision, device, TOKEN_LIMIT, texts]).encode()).hexdigest()
    metadata = path.with_suffix(".sha256")
    partial = path.with_suffix(".tmp.npy")
    if metadata.exists() and metadata.read_text(encoding="ascii") == signature:
        if path.exists():
            return
        cache = np.load(partial, mmap_mode="r+", allow_pickle=False) if partial.exists() else None
    else:
        cache = None
    if cache is None:
        cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(texts), 768))
        metadata.write_text(signature, encoding="ascii")
    complete = np.abs(np.linalg.norm(cache, axis=1) - 1) < .001
    order = sorted(np.flatnonzero(~complete), key=lambda i: len(encoded["input_ids"][i]))
    print(f"Teacher {path.name}: {len(order)}/{len(texts)} texts left on {device}", flush=True)
    with torch.inference_mode():
        for offset in range(0, len(order), batch):
            indices = order[offset:offset + batch]
            inputs = encoder.tokenizer.pad({"input_ids": [encoded["input_ids"][i] for i in indices],
                                            "attention_mask": [encoded["attention_mask"][i] for i in indices]},
                                           padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                values = encoder.model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            cache[indices] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
            if offset % (40 * batch) == 0:
                print(f"  {offset + len(indices)}/{len(order)}", flush=True)
    cache.flush()
    del cache
    partial.replace(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("static_assets", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--only", choices=("fit", "public", "both"), default="both")
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("The teacher batch size must be positive")
    root = args.directory
    if args.only in ("fit", "both"):
        pairs_path = root / "pairs.json"
        if not pairs_path.exists():
            tokenizer = Tokenizer.from_file(str(args.static_assets / "tokenizer.json"))
            tokenizer.enable_truncation(TOKEN_LIMIT)
            pairs = squad_pairs(root / "squad.json", tokenizer)
            for split in ("train", "dev", "test"):
                pairs[f"industry_{split}"] = industry_pairs(split)
            pairs_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf8")
        else:
            pairs = json.loads(pairs_path.read_text(encoding="utf8"))
        prompts = list(dict.fromkeys(text for split in ("train", "industry_train")
                                     for pair in pairs[split] for text in pair_prompts(pair)))
        (root / "teacher-prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf8")
    if args.only in ("public", "both"):
        public = public_fixture(root)
        (root / "nfcorpus-evaluation.json").write_text(json.dumps(public, ensure_ascii=False), encoding="utf8")
    encoder = Encoder(args.teacher, 4)
    encoder.model.to(args.device)
    revision = sha256((args.teacher / "config.json").read_bytes()).hexdigest()
    if args.only in ("fit", "both"):
        encode_teacher(encoder, prompts, args.device, args.batch, root / "fit-teacher.npy", revision)
    if args.only in ("public", "both"):
        encode_teacher(encoder, [prompt(row) for row in public["rows"]],
                       args.device, args.batch, root / "nfcorpus-teacher.npy", revision)


if __name__ == "__main__":
    main()
