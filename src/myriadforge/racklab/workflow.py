"""Versioned CPU automotive intake with explicit approval and hold events."""

from dataclasses import dataclass
from hashlib import sha256
import json
import lzma
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

from .journal import Request, StreamError, encode


DATA = Path(__file__).with_name("packs")
DEFAULT_PACK = DATA / "automotive.json"
RESULT_SCHEMA = "racklab.result/1"
TOKEN_LIMIT = 131_072
TOKENIZER_HASH = "97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25"
FACTS = {"vehicle confirmed", "route confirmed", "reference found", "review approved", "review hold"}
ARTIFACTS = {"policy.json", "policy-table.json", "needle3.model.xz"}
ROUTES = {"recall_lookup", "bulletin_lookup"}
REASONS = {
    "held": "Operator hold",
    "needs_route": "Route is missing, unsupported, or ambiguous",
    "needs_vehicle": "One confirmed year, make, and model is required",
    "needs_reference": "No uniquely supported reference for this route and vehicle",
    "needs_approval": "Explicit approval of this case is required",
    "proposed": "Reference and approval verified for staff review",
}


def digest(data):
    return sha256(data).hexdigest()


def words(value):
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))


def contains(text, phrase):
    return f" {words(phrase)} " in text


def vehicle(value):
    if not isinstance(value, dict) or set(value) != {"year", "make", "model"}:
        raise ValueError("vehicle requires year, make, and model")
    year, make, model = (value[field] for field in ("year", "make", "model"))
    if type(year) is not int or not 1900 <= year <= 2100:
        raise ValueError("vehicle year must be an integer from 1900 through 2100")
    if not all(isinstance(s, str) and len(s) <= 80 and words(s) for s in (make, model)):
        raise ValueError("vehicle make and model must be nonempty strings of at most 80 characters")
    return {"year": year, "make": make.strip(), "model": model.strip()}


def event_request(value):
    if not isinstance(value, dict) or value.keys() - {"stream_id", "seq", "event", "text", "vehicle"}:
        raise StreamError("invalid_request", "Use stream_id, seq, event, and optional text and vehicle")
    event = value.get("event")
    if event not in ("append", "approve", "hold", "reset"):
        raise StreamError("invalid_request", "event must be append, approve, hold, or reset")
    fields = {"event": event}
    if event == "append":
        if not isinstance(value.get("text"), str):
            raise StreamError("invalid_request", "append requires a text string")
        try:
            value["text"].encode("utf-8")
            fields["text"] = value["text"]
            if "vehicle" in value:
                fields["vehicle"] = vehicle(value["vehicle"])
        except ValueError as error:
            raise StreamError("invalid_request", str(error)) from error
    elif value.keys() & {"text", "vehicle"}:
        raise StreamError("invalid_request", "Only an append event accepts text and vehicle")
    return Request.parse({
        "stream_id": value.get("stream_id"), "seq": value.get("seq"),
        "append": encode(fields), "reset": event == "reset",
    })


def source_record(source):
    return {key: source[key] for key in ("id", "title", "url", "kind", "review_note") if key in source}


class Automotive:
    def __init__(self, path=DEFAULT_PACK):
        path = Path(path)
        raw = path.read_bytes()
        pack = json.loads(raw)
        if pack["schema"] != "racklab.pack/1" or set(pack["artifacts"]) != ARTIFACTS:
            raise ValueError("Invalid automotive pack or artifact inventory")
        if not isinstance(pack["version"], str) or not pack["version"].strip():
            raise ValueError("Pack version must be a nonempty string")
        files = {}
        for name, expected in pack["artifacts"].items():
            data = (path.parent / name).read_bytes()
            if digest(data) != expected:
                raise ValueError(f"Digest mismatch for {name}")
            files[name] = data
        table = json.loads(files["policy-table.json"])
        if (set(table["facts"]) != FACTS or len(table["facts"]) != len(FACTS)
                or len(table["actions"]) != 2 ** len(FACTS)
                or table["policy_sha256"] != digest(files["policy.json"])):
            raise ValueError("Invalid policy truth table")
        self.facts, self.actions = table["facts"], table["actions"]
        for mask, action in enumerate(self.actions):
            atoms = {fact: mask >> index & 1 for index, fact in enumerate(reversed(self.facts))}
            expected = all(bool(atoms[fact]) == (fact != "review hold") for fact in FACTS)
            if action != ("review_service_reference" if expected else None):
                raise ValueError("Pack policy must preserve the automotive approval and hold contract")
        tokenizer = lzma.decompress(files["needle3.model.xz"])
        if digest(tokenizer) != TOKENIZER_HASH:
            raise ValueError("Unexpected Needle 3 tokenizer")
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError as error:
            raise ValueError('Install the workflow extra with pip install "myriadforge[racklab]"') from error
        self.tokenizer = SentencePieceProcessor(model_proto=tokenizer)
        self.version = pack["version"]
        self.identity = encode({"adapter": "racklab.automotive/1", "pack_sha256": digest(raw)})
        self.sources = pack["sources"]
        self.routes = pack["router"]
        if set(self.routes) != ROUTES | {"other"} or any(
            not isinstance(items, list) or not items or
            any(not isinstance(item, str) or not words(item) for item in items)
            for items in self.routes.values()
        ):
            raise ValueError("Invalid routing phrases")
        ids = set()
        for source in self.sources:
            url = urlsplit(source["url"])
            if (not isinstance(source["id"], str) or not words(source["id"])
                    or words(source["id"]) in ids or source["kind"] not in ROUTES
                    or url.scheme != "https" or url.hostname not in {"api.nhtsa.gov", "static.nhtsa.gov"}):
                raise ValueError("Invalid source ID, type, or URL")
            if not source["years"] or not source["terms"]:
                raise ValueError("Catalog sources require model years and search terms")
            for year in source["years"]:
                vehicle({"year": year, "make": source["make"], "model": source["model"]})
            ids.add(words(source["id"]))

    def open_stream(self):
        return AutomotiveStream(self)

    def analyze(self, note, supplied):
        text = " " + words(note) + " "
        make_models = {(s["make"], s["model"]) for s in self.sources}
        identified = set()
        year = r"(19\d\d|20\d\d|2100)"
        for make, model in make_models:
            car = re.escape(words(make)) + " (?:" + "|".join(
                re.escape(alias) for alias in {words(model), model_key(model)}
            ) + ")"
            pattern = rf"(?<= ){year} {car}(?= )|(?<= ){car} {year}(?= )"
            identified.update((int(a or b), make, model) for a, b in re.findall(pattern, text))
        subject = None
        if supplied:
            expected = (supplied["year"], words(supplied["make"]), model_key(supplied["model"]))
            if not identified or all((y, words(m), model_key(car)) == expected for y, m, car in identified):
                subject = supplied
        elif len(identified) == 1:
            matched_year, make, model = next(iter(identified))
            subject = {"year": matched_year, "make": make, "model": model}

        explicit = [s for s in self.sources if contains(text, s["id"])]
        labels = {route for route, phrases in self.routes.items() if any(contains(text, phrase) for phrase in phrases)}
        labels.update(source["kind"] for source in explicit)
        route = next(iter(labels)) if len(labels) == 1 else None
        if route not in ROUTES:
            route = None
        if route is None or subject is None:
            return route, subject, []
        scope = [
            source for source in self.sources
            if source["kind"] == route and subject["year"] in source["years"]
            and words(subject["make"]) == words(source["make"])
            and model_key(subject["model"]) == model_key(source["model"])
        ]
        if explicit:
            return route, subject, explicit if len(explicit) == 1 and explicit[0] in scope else []
        ranked = sorted(
            ((sum(contains(text, term) for term in source["terms"]), source["id"], source) for source in scope),
            key=lambda row: (-row[0], row[1]),
        )
        if ranked and ranked[0][0] > (ranked[1][0] if len(ranked) > 1 else 0):
            return route, subject, [ranked[0][2]]
        return route, subject, []


def model_key(model):
    return words(model).replace(" ", "")


@dataclass
class AutomotiveStream:
    adapter: Automotive
    note: str = ""
    subject: dict | None = None
    approval: str | None = None
    hold: bool = False
    count: int = 0

    def append(self, payload, reset=False):
        event = json.loads(payload)
        kind = event["event"]
        if kind == "append":
            note = self.note + event["text"]
            count = len(self.adapter.tokenizer.encode(note, out_type=int))
            if count > TOKEN_LIMIT:
                raise ValueError(f"Input has {count} Needle tokens; maximum is {TOKEN_LIMIT}")
            self.note, self.count = note, count
            if "vehicle" in event:
                self.subject = event["vehicle"]
        elif kind == "hold":
            self.hold = True
        route, subject, sources = self.adapter.analyze(self.note, self.subject)
        eligible = bool(route and subject and sources)
        case = encode((route, subject, [source["id"] for source in sources])) if eligible else None
        if self.approval != case:
            self.approval = None
        if kind == "approve" and not self.hold:
            self.approval = case
        approved = case is not None and self.approval == case
        facts = {
            "vehicle confirmed": subject is not None,
            "route confirmed": route is not None,
            "reference found": bool(sources),
            "review approved": approved,
            "review hold": self.hold,
        }
        index = int("".join(str(int(facts[fact])) for fact in self.adapter.facts), 2)
        action = self.adapter.actions[index]
        if self.hold:
            state = "held"
        elif route is None:
            state = "needs_route"
        elif subject is None:
            state = "needs_vehicle"
        elif not sources:
            state = "needs_reference"
        else:
            state = "proposed" if action else "needs_approval"
        return {
            "pack_version": self.adapter.version, "backend": "cpu", "route": route,
            "subject": subject, "references": [source_record(source) for source in sources],
            "status": state, "reason": REASONS[state], "proposal": action, "needle_tokens": self.count,
        }


def process(journal, value):
    try:
        response = journal.process(event_request(value))
        return {**response, "schema": RESULT_SCHEMA}
    except StreamError as error:
        response = {"schema": RESULT_SCHEMA, "ok": False, "error": {"code": error.code, "message": str(error)}}
        if isinstance(value, dict):
            response.update({key: value[key] for key in ("stream_id", "seq") if key in value})
        if error.expected_seq is not None:
            response["error"]["expected_seq"] = error.expected_seq
        return response
    except sqlite3.Error:
        return {"schema": RESULT_SCHEMA, "ok": False, "error": {
            "code": "journal_error", "message": "Journal operation failed",
        }}
