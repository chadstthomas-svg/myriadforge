"""CPU differential tests for the exact tokenizer and decision compiler."""
import itertools
import io
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from policy import LIMIT, Policy, SegmentMemo, count_tokens, load_tokenizer, needle_count, tokenizer_prior


HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("sentencepiece")
    return load_tokenizer()


def test_tokenizer_split_preconditions(tokenizer):
    from sentencepiece import sentencepiece_model_pb2 as sp

    model = sp.ModelProto.FromString(tokenizer.serialized_model_proto())
    assert model.trainer_spec.model_type == sp.TrainerSpec.BPE
    assert model.trainer_spec.byte_fallback
    assert model.normalizer_spec.name == "identity"
    assert model.normalizer_spec.escape_whitespaces
    assert not model.normalizer_spec.precompiled_charsmap
    assert not model.normalizer_spec.add_dummy_prefix
    assert not model.normalizer_spec.remove_extra_whitespaces
    for piece in model.pieces:
        if piece.type == sp.ModelProto.SentencePiece.NORMAL:
            assert "\u2581" not in piece.piece[1:]
        elif piece.type == sp.ModelProto.SentencePiece.USER_DEFINED:
            assert " " not in piece.piece


def test_compact_tokenizer_restores_the_pinned_model(tokenizer, tmp_path):
    from compact_tokenizer import TOKENIZER_HASH, read_tokenizer
    import hashlib

    compact = HERE / "needle3-tokenizer.m8t.xz"
    original = read_tokenizer(compact)
    assert compact.stat().st_size == 26136
    assert len(original) == 126520
    assert hashlib.sha256(original).hexdigest() == TOKENIZER_HASH
    assert tokenizer.serialized_model_proto() == original
    corrupted = tmp_path / "modified.model"
    corrupted.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="hash"):
        read_tokenizer(corrupted)


def test_all_preseeded_token_counts_match_sentencepiece(tokenizer):
    prior = tokenizer_prior(tokenizer)
    assert len(prior) == 4200
    encoded = tokenizer.encode([" " + segment for segment in prior], out_type=int, num_threads=4)
    assert [len(ids) for ids in encoded] == list(prior.values())


def test_count_matches_original_encoder(tokenizer):
    rng = random.Random(1102)
    alphabet = ["", " ", "  ", "\t", "\r\n", "\x00", "\u200d", "\u2028", "\u00a0", "\u2581",
                "é", "e\u0301", "ß", "📃", "東京", "שלום", "☀️", "<think>", "<context>", "don't", "123"]
    unbroken = "東京☀️" * 2500
    fixtures = ["", " " * 9000, unbroken, unbroken + " X", " a" * LIMIT, " a" * (LIMIT + 1)]
    fixtures.extend("".join(rng.choices(alphabet, k=3500)) for _ in range(110))
    fixtures.extend(" ".join(str(rng.getrandbits(64)) for _ in range(2200)) for _ in range(4))
    for doc in fixtures:
        assert needle_count(tokenizer, doc) == len(tokenizer.encode(doc, out_type=int))
    assert count_tokens(tokenizer, " a" * LIMIT, fast=True) == LIMIT
    with pytest.raises(ValueError, match="131073"):
        count_tokens(tokenizer, " a" * (LIMIT + 1), fast=True)


def test_regex_projection_and_boolean_semantics():
    spec = {"rules": {"text": '"aa aa" OR "Straße"', "veto": '"a-a" OR "☀️"'},
            "answers": {"matched": "@text AND NOT @veto OR \"a, a\""},
            "categories": {"two": '"aa" AND "東京"'},
            "actions": {"review": '"é!" AND NOT @_bad'} }
    with pytest.raises(ValueError, match="earlier rule"):
        Policy(spec)
    spec["actions"]["review"] = '"é!" AND (NOT "other" OR "(a)" OR "💡b")'
    policy = Policy(spec)
    rng = random.Random(3403)
    alphabet = ["AA", "a", "aa", "x", ".", "-", ",", "_", " ", "\r\n", "\t", "\u00a0",
                "Straße", "東京", "é", "💡", "b", "☀️", "other", "(", ")", "!"]
    for _ in range(1100):
        doc = "".join(rng.choices(alphabet, k=rng.randrange(1, 500)))
        oracle = policy.atoms(policy.project(doc))
        assert np.array_equal(policy.text_atoms(doc), oracle)
        assert np.array_equal(policy.anchored_atoms(doc), oracle)
    assert policy.result(policy.reference(policy.text_atoms("STRASSE")))["answers"] == {"matched": True}
    assert policy.result(policy.reference(policy.text_atoms("STRASSE + a-a")))["answers"] == {"matched": False}


def test_unused_explanation_rule_keeps_device_input_small():
    policy = Policy({"rules": {"annotation": '"only for explanation"'},
                     "answers": {"yes": '"evidence"'}})
    document = "Evidence only for explanation"
    atoms = policy.text_atoms(document)
    assert atoms.tolist() == [1] and len(policy.literals) == 1
    assert policy.explain(document, atoms) == {
        "matched": ["evidence", "only for explanation"], "named_rules": {"annotation": True}
    }


def test_anchored_match_falls_back_without_losing_late_evidence():
    policy = Policy({"answers": {"found": '"package missing" OR "not a veto"'}})
    document = "subpackage delivered ; package reported " * 400 + "Package\tmissing"
    oracle = policy.atoms(policy.project(document))
    assert oracle.tolist() == [1, 0]
    assert np.array_equal(policy.anchored_atoms(document), oracle)


def test_published_batch_blobs_still_use_one_stage_and_512_bytes():
    from blob_layout import inspect_blob

    folder = HERE / "bundles" / "magiceightball-batch"
    for filename in ("ncs1-native.blob", "ncs2-2020.3.blob"):
        data = (folder / filename).read_bytes()
        layout = inspect_blob(data)
        assert layout["sections"] == {
            "headers": 192, "input_descriptor": 64, "output_descriptor": 64,
            "stages": 128, "constants": 64,
        }
        assert layout["bytes"] == 512 and layout["stage_count"] == 1
        assert layout["input_bytes"] == 128 * 4 and layout["output_bytes"] == 128 * 2
        with pytest.raises(ValueError, match="layout"):
            inspect_blob(data[:-1])


def test_native_release_round_trips_both_devices_and_banked_manifest(tmp_path):
    from release import TARGETS, pack_bundle, read_release

    for name in ("magiceightball-batch", "magiceightball-banks"):
        bundle = HERE / "bundles" / name
        archive = tmp_path / "candidate.m8b"
        archive.write_bytes(pack_bundle(bundle))
        manifest, blobs = read_release(archive)
        assert manifest == json.loads((bundle / "model.json").read_bytes())
        assert all(blobs[stick] == (bundle / filename).read_bytes()
                   for stick, (filename, _) in TARGETS.items())
        if name == "magiceightball-batch":
            published = HERE / "bundles" / "magiceightball.m8b"
            assert len(published.read_bytes()) == 735
            assert archive.read_bytes() == published.read_bytes()
        archive.write_bytes(archive.read_bytes()[:-3])
        with pytest.raises(ValueError, match="Unreadable"):
            read_release(archive)


def test_stream_matches_full_tokenizer_and_evidence_for_each_append(tokenizer):
    from runtime import Runtime
    from stream import DecisionStream

    engine = Runtime.__new__(Runtime)
    engine.info = {"variant": "gates"}
    engine.model_name = "test"
    engine.policy = Policy({
        "rules": {"unused": '"zero crossing"'},
        "answers": {"accept": '"review authorized" AND "package missing" AND NOT "do not route"'},
        "categories": {"seen": '"Ångström" OR "東京" OR "zero crossing here"'},
        "actions": {"review": '"review authorized"'},
    })
    engine.tokenizer = tokenizer
    engine.head = engine.policy.reference
    stream = DecisionStream(engine, explain=True)
    rng = random.Random(128)
    phrases = ["review authorized", " package missing", " do not route", "ÅNGSTRÖM", "東京",
               "zero crossing here", "가", "\r\n", " ", "▁", "<think>", "ß"]
    for number in range(9):
        whole = ""
        parts = ["package", " missing", " review authorized", "_later", ". "]
        text = " ".join(rng.choices(phrases, k=30))
        while text:
            width = rng.randrange(1, 12)
            parts.append(text[:width])
            text = text[width:]
        for i, fragment in enumerate(parts):
            row = stream.append(fragment, reset=i == 0)
            whole += fragment
            expected_facts = engine.policy.atoms(engine.policy.project(whole))
            assert row["needle_tokens"] == len(tokenizer.encode(whole, out_type=int))
            assert row["atoms"] == expected_facts.tolist()
            assert row["explanation"] == engine.policy.explain(whole, expected_facts)
            if i in (2, 3):
                assert row["result"]["answers"]["accept"] == (i == 2)

    # The cold fast path projects only a trailing window. The final literal
    # spans the boundary between the committed text and the mutable suffix.
    whole = "review authorized. " + "neutral " * 1000 + "package missing"
    row = stream.append(whole, reset=True)
    assert row["result"]["answers"]["accept"]
    assert row["needle_tokens"] == len(tokenizer.encode(whole, out_type=int))
    row = stream.append("_pending")
    whole += "_pending"
    assert not row["result"]["answers"]["accept"]
    assert row["atoms"] == engine.policy.atoms(engine.policy.project(whole)).tolist()
    assert row["needle_tokens"] == len(tokenizer.encode(whole, out_type=int))


def test_stream_rolls_back_overflow_and_failed_inference(tokenizer, monkeypatch):
    from runtime import Runtime
    from stream import DecisionStream

    engine = Runtime.__new__(Runtime)
    engine.info = {"variant": "gates"}
    engine.model_name = "test"
    engine.tokenizer = tokenizer
    engine.policy = Policy({"answers": {"ok": '"a" AND NOT "b"'}})
    engine.head = engine.policy.reference
    stream = DecisionStream(engine)
    assert stream.append(" a" * (LIMIT - 1))["needle_tokens"] == LIMIT - 1
    saved = stream.state
    with pytest.raises(ValueError, match="131073"):
        stream.append(" b b")
    assert stream.state is saved
    with monkeypatch.context() as patch:
        def fail(atoms):
            raise RuntimeError("test device failure")

        patch.setattr(engine, "head", fail)
        with pytest.raises(RuntimeError, match="device failure"):
            stream.append(" b")
    assert stream.state is saved
    row = stream.append(" b")
    assert row["needle_tokens"] == LIMIT and not row["result"]["answers"]["ok"]
    assert stream.append(" a", reset=True)["result"]["answers"]["ok"]
    assert stream.state.explanation_text == ""


@pytest.mark.parametrize("stick", ["ncs1", "ncs2"])
def test_runtime_imports_release_with_exact_target_tag(monkeypatch, tmp_path, tokenizer, stick):
    from release import TARGETS
    from runtime import Runtime

    legacy = SimpleNamespace(setup=lambda: None)
    monkeypatch.setitem(sys.modules, "ncs1_paths", legacy)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    imports = []

    class FakeCore:
        available_devices = ["MYRIAD"]

        def import_network(self, filename, target, config, num_requests):
            path = Path(filename)
            imports.append((path, path.read_bytes(), target, config, num_requests))
            return object()

    monkeypatch.setitem(sys.modules, "openvino.inference_engine",
                        SimpleNamespace(IECore=FakeCore, get_version=lambda: "test"))
    engine = Runtime(HERE / "bundles" / "magiceightball.m8b", toolchain=tmp_path,
                     release_stick=stick)
    path, blob, target, config, requests = imports[0]
    filename, platform = TARGETS[stick]
    assert blob == (HERE / "bundles" / "magiceightball-batch" / filename).read_bytes()
    assert target == "MYRIAD" and requests == 1 and config["VPU_MYRIAD_PLATFORM"] == platform
    assert not path.exists() and engine.batch_capacity == 128 and engine.ir is None


def test_usb_identity_distinguishes_both_cold_sticks():
    from benchmark_batch_heads import check_usb_identity

    old = r"USB\VID_03E7&PID_2150\ORIGINAL"
    newer = r"USB\VID_03E7&PID_2485\SECOND"
    assert check_usb_identity("ncs1", [old])["boot_usb_id"] == "03E7:2150"
    assert check_usb_identity("ncs2", [newer])["boot_usb_id"] == "03E7:2485"
    for devices in ([old], [old, newer], [r"USB\VID_03E7&PID_F63B\FIRMWARE"]):
        with pytest.raises(RuntimeError, match="Expected one cold NCS2"):
            check_usb_identity("ncs2", devices)


def test_policy_banks_share_device_calls_and_keep_results_separate(tmp_path, tokenizer):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from adaptive import AdaptiveRuntime, Profiles
    from benchmark_policy_banks import literal_fixtures
    from build import compile_banks
    from runtime import Runtime

    alias = tmp_path / "alias"
    compile_banks({"support": HERE / "support.json", "alias": HERE / "support.json"}, alias, 128)
    reused = json.loads((alias / "model.json").read_text(encoding="utf8"))
    assert reused["fp16_coefficient_bytes"] == 32
    assert reused["banks"]["support"]["bank_offset"] == reused["banks"]["alias"]["bank_offset"] == 0

    compile_banks({"support": HERE / "support.json", "oncall": HERE / "oncall.json"}, tmp_path, 128)
    engine = Runtime.__new__(Runtime)
    engine.folder = tmp_path
    engine.bundle_info = json.loads((tmp_path / "model.json").read_text(encoding="utf8"))
    engine._use_bank(None)
    engine.tokenizer = tokenizer
    engine.fast_count = True
    engine.lane = "ncs1"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model = ort.InferenceSession(str(tmp_path / "model.onnx"), options, providers=["CPUExecutionProvider"])
    calls = []

    class Request:
        def infer(self, inputs):
            calls.append(inputs["features"].copy())
            out = model.run([engine.info["decisions"]], inputs)[0]
            return {"gather": out.astype(np.float16).view(np.int16)}

    engine.request = Request()
    cases = list(literal_fixtures())
    rows = [dict(entry, id=i) for i, (entry, _) in enumerate(cases)]
    rows.insert(3, {"id": "wrong-bank", "bank": "unknown", "text": "irrelevant"})
    rows.insert(7, {"id": "overflow", "bank": "oncall", "text": " a" * 131073})
    reply = engine.decide_mixed(rows)
    assert reply["device_requests"] == len(calls) == 2
    assert calls[0][::2].max() < 16 and calls[0][1::2].min() >= 16
    assert "Unknown policy bank" in reply["documents"][3]["error"]
    assert "131073" in reply["documents"][7]["error"]
    accepted = [row for row in reply["documents"] if "result" in row]
    assert [row["id"] for row in accepted] == list(range(256))
    assert [row["result"] for row in accepted] == [expected for _, expected in cases]

    profiles = Profiles()
    loads = []

    def factory():
        loads.append(True)
        return engine

    try:
        live = AdaptiveRuntime({"support_notes": factory, "oncall_notes": factory}, profiles,
                               banks={"support_notes": "support", "oncall_notes": "oncall"})
        a = live.batch({"vertical": "support_notes", "documents": [{"text": "package missing; review authorized"}]})
        b = live.batch({"vertical": "oncall_notes", "documents": [{"text": "queue backed up; paging approved"}]})
        assert a["documents"][0]["result"]["action"] == "review_delivery"
        assert b["documents"][0]["result"] == {
            "answers": {"may_page": False}, "category": "capacity", "action": None,
        }
        assert a["device_requests"] == b["device_requests"] == len(loads) == 1
    finally:
        profiles.close()


def test_json_protocol_keeps_native_stdout_off_the_wire():
    script = (
        "from runtime import json_output_stream; import os; "
        "output = json_output_stream(); os.write(1, b'firmware diagnostic'); "
        "print('{}', file=output); output.close()"
    )
    child = subprocess.run([sys.executable, "-c", script], cwd=HERE,
                           capture_output=True, text=True, check=True, timeout=15)
    assert child.stdout == "{}\n"
    assert child.stderr == "firmware diagnostic"


def test_worker_deadline_recovers_for_the_next_batch():
    from supervisor import supervise

    script = """
import sys, time
for line in sys.stdin:
    if 'stall' in line:
        time.sleep(30)
    print(line, end='', flush=True)
"""
    inputs = [b'{"id":"normal"}\n', b'{"id":"stall"}\n', b'{"id":"next"}\n']
    output = io.StringIO()
    supervise([sys.executable, "-u", "-c", script], inputs, output, 1.0)
    assert "\r" not in output.getvalue()
    first, second, third = (json.loads(line) for line in output.getvalue().splitlines())
    assert first == {"id": "normal"}
    assert second["id"] == "stall" and "timed out" in second["error"]
    assert third == {"id": "next"}


def test_segment_memo_is_exact_and_bounded(tokenizer):
    memo = SegmentMemo(capacity=7, prior=tokenizer_prior(tokenizer))
    fixtures = [
        (" a space Straße\r\n 東京 " * 230) + "x" * 1000,
        (" a very different space 東京 markup " * 230),
        (" a space Straße\r\n 東京 " * 240),
        " a" * (LIMIT + 1),
    ]
    for document in fixtures * 2:
        assert needle_count(tokenizer, document, memo=memo) == len(tokenizer.encode(document))
        assert len(memo.counts) <= 7
        assert all(len(key) <= 96 for key in memo.counts)


def test_bandit_quarantines_wrong_lane_and_retests_after_cooldown(monkeypatch):
    from adaptive import BatchEvidence, COOLDOWN, Profiles, REFERENCE

    profiles = Profiles()
    try:
        state = profiles.get("support")
        wrong = {"cached"}

        def measure(_self, lane, text, _evidence):
            cost = {REFERENCE: 3, "segmented": 2, "cached": 1}[lane]
            return (len(text), np.array([int(lane not in wrong)]), 0), cost

        monkeypatch.setattr(BatchEvidence, "measure", measure)
        memo = SegmentMemo()
        engine = SimpleNamespace(policy=Policy({"answers": {"seen": '"data"'}}))
        probe = BatchEvidence(engine, state, memo)
        assert probe.prepare("data")[1].tolist() == [1]
        assert probe.used == {"segmented": 1}
        assert "cached" in probe.failures
        assert state["retry_at"]["cached"] == COOLDOWN

        wrong.clear()
        state["batches"] = COOLDOWN
        retry = BatchEvidence(engine, state, memo)
        assert retry.prepare("data")[1].tolist() == [1]
        assert retry.used == {"cached": 1}
        assert not state["retry_at"]
    finally:
        profiles.close()


def test_batch_routes_verticals_and_recovers_a_failed_device(tokenizer, tmp_path):
    from adaptive import AdaptiveRuntime, Profiles

    profiles = Profiles(tmp_path / "routing.db")
    engines = []

    class Device:
        lane = "ncs1"
        batch_capacity = 2

        def __init__(self, label, fail):
            self.tokenizer = tokenizer
            self.fail = fail
            self.info = {"variant": "gather", "policy": {"answers": {label: '"match"'}}}
            self.policy = Policy(self.info["policy"])
            engines.append(self)

        def prepare_decision(self, text, prepare):
            tokens, facts, _ = prepare(text)
            return tokens, facts

        def decide_prepared(self, documents):
            if self.fail:
                raise RuntimeError("device failed")
            results = []
            for tokens, facts in documents:
                bits = self.policy.reference(facts)
                result = self.policy.result(bits)
                results.append({"needle_tokens": tokens, "result": result, "atoms": facts.tolist(), "bits": bits.tolist()})
            return results

    batcher = AdaptiveRuntime({
        "support": lambda: Device("support_yes", not engines),
        "ops": lambda: Device("ops_yes", False),
    }, profiles, scopes={"ops": "query"})
    support = batcher.batch({"vertical": "support", "documents": [
        {"id": 1, "text": "match"}, {"id": 2, "text": '{ "event": "match" }'}
    ]})
    assert support["documents"][0]["device_reloaded"]
    assert support["documents"][1]["device_reloaded"]
    assert support["device_requests"] == 2  # Failed submission, then one shared retry.
    assert support["documents"][0]["result"]["answers"] == {"support_yes": True}
    ops = batcher.batch({"vertical": "ops", "documents": [
        {"id": 3, "text": '{"archive":"match","query":"nothing"}',
         "expected": {"answers": {"ops_yes": False}, "category": None, "action": None}},
        {"id": 4, "text": '{"archive":"nothing"}'},
    ]})
    assert ops["documents"][0]["result"]["answers"] == {"ops_yes": False}
    assert ops["routes"][0]["feedback"]["joint"] == 1
    assert "JSON evidence field" in ops["documents"][1]["error"]
    assert ops["device_requests"] == 1
    profiles.close()
    reopened = Profiles(tmp_path / "routing.db")
    try:
        states = reopened.db.execute("SELECT profile, state FROM lanes ORDER BY profile").fetchall()
        assert len(states) == 3
        assert all(json.loads(state)["batches"] == 1 for _, state in states)
    finally:
        reopened.close()


def test_radix_codec_round_trips_every_normal_fp16_bit_pattern():
    from radix import CAPACITY, encode_ranks, decode_ranks

    ranks = np.arange(CAPACITY)
    words = encode_ranks(ranks)
    assert np.unique(words).size == CAPACITY
    assert np.array_equal(decode_ranks(words), ranks)
    assert np.array_equal(decode_ranks(words.astype(np.float32)), ranks)
    with pytest.raises(RuntimeError, match="non-normal"):
        decode_ranks(np.array([0], dtype=np.float16))
    with pytest.raises(RuntimeError, match="inexact"):
        decode_ranks(np.array([0.2], dtype=np.float32))


def test_radix_packs_more_facts_into_the_same_constant_section():
    pytest.importorskip("onnx")
    from build import packed_table
    from radix import decode_ranks

    terms = ['"fact%d"' % index for index in range(8)]
    policy = Policy({
        "rules": {"some": " OR ".join(terms), "every": " AND ".join(terms)},
        "answers": {"approved": "@every"},
        "categories": {"has_evidence": "@some"},
        "actions": {"approve": "@every"},
    })
    binary, _ = packed_table(policy)
    compact, info = packed_table(policy, "radix")
    assert binary.size * 2 == 128
    assert info["radix_digits"] == 10 and compact.size * 2 == 52
    words = decode_ranks(compact)
    for mask in range(256):
        digit = (int(words[mask // 10]) // 3 ** (mask % 10)) % 3
        assert info["codebook"][digit] == (int(mask == 255), int(mask != 0), int(mask == 255))


@pytest.mark.parametrize("variant,codec", [
    ("gates", "binary"), ("lookup", "binary"), ("packed", "binary"),
    ("gather", "binary"), ("gather", "radix"), ("dense", "binary"), ("scan", "binary"),
])
def test_compile_and_evaluate_all_assignments(tmp_path, variant, codec):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from benchmark import PHRASES, expected
    from build import compile_policy
    from runtime import Runtime

    compile_policy(HERE / "support.json", tmp_path, variant, side=8,
                   slots=8 if variant == "gather" else 1, codec=codec)
    engine = Runtime.__new__(Runtime)
    engine.info = json.loads((tmp_path / "model.json").read_text(encoding="utf8"))
    if codec == "radix":
        assert engine.info["fp16_coefficient_bytes"] == 32
        assert engine.info["radix_digits"] == 8
    engine.lane = "ncs1"
    engine.policy = Policy(engine.info["policy"])
    engine.output = engine.info["decisions"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), options, providers=["CPUExecutionProvider"])

    class Request:
        def infer(self, inputs):
            output = session.run([engine.output], inputs)[0]
            if variant == "gather":
                output = output.astype(np.float16).view(np.int16)
            return {engine.output: output}

    engine.request = Request()
    fixtures = []
    for flags in itertools.product((False, True), repeat=len(PHRASES)):
        selected = [i for i, yes in enumerate(flags) if yes]
        phrase_text = "neutral " * 63 + " ; ".join(PHRASES[i] for i in selected)
        atoms = engine.policy.atoms(engine.policy.project(phrase_text))
        if variant == "scan":
            bits, found, _ = engine.scan(engine.policy.project(phrase_text))
            assert np.array_equal(found, atoms)
        else:
            bits = engine.head(atoms)
            fixtures.append((atoms, bits))
        assert np.array_equal(bits, engine.policy.reference(atoms))
        assert engine.policy.result(bits) == expected(selected)
    if variant == "gather":
        for start in range(0, len(fixtures), 7):
            batch = fixtures[start:start + 7]
            outputs = engine.gather_heads([atoms for atoms, _ in batch])
            assert np.array_equal(outputs, [bits for _, bits in batch])
