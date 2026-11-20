"""CPU differential tests for the exact tokenizer and decision compiler."""
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

from policy import LIMIT, Policy, count_tokens, load_tokenizer, needle_count


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
        assert np.array_equal(policy.text_atoms(doc), policy.atoms(policy.project(doc)))
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


@pytest.mark.parametrize("variant", ["gates", "lookup", "packed", "dense", "scan"])
def test_compile_and_evaluate_all_assignments(tmp_path, variant):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from benchmark import PHRASES, expected
    from build import compile_policy
    from runtime import Runtime

    compile_policy(HERE / "support.json", tmp_path, variant, side=8)
    engine = Runtime.__new__(Runtime)
    engine.info = json.loads((tmp_path / "model.json").read_text(encoding="utf8"))
    engine.policy = Policy(engine.info["policy"])
    engine.output = engine.info["decisions"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), options, providers=["CPUExecutionProvider"])

    class Request:
        def infer(self, inputs):
            return {engine.output: session.run([engine.output], inputs)[0]}

    engine.request = Request()
    for flags in itertools.product((False, True), repeat=len(PHRASES)):
        selected = [i for i, yes in enumerate(flags) if yes]
        phrase_text = "neutral " * 63 + " ; ".join(PHRASES[i] for i in selected)
        atoms = engine.policy.atoms(engine.policy.project(phrase_text))
        if variant == "scan":
            bits, found, _ = engine.scan(engine.policy.project(phrase_text))
            assert np.array_equal(found, atoms)
        else:
            bits = engine.head(atoms)
        assert np.array_equal(bits, engine.policy.reference(atoms))
        assert engine.policy.result(bits) == expected(selected)
