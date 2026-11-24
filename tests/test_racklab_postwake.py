import importlib
from pathlib import Path

import pytest


@pytest.fixture
def scoring(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "labs" / "racklab"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("postwake_asr")


def test_identifier_matching_accepts_speech_forms_without_changing_digits(scoring):
    assert scoring.field_match("Recall one seven V six five two.", ["17V652"])
    assert scoring.field_match("A one six four seven is the model.", ["A1647"])
    assert scoring.field_match("Ford F one fifty.", ["Ford F150"])
    assert not scoring.field_match("Recall 17V625.", ["17V652"])
    assert not scoring.field_match("Return A16470.", ["A1647"])


def test_reports_field_losses_separately_from_word_error(scoring):
    report = scoring.summarize([{
        "cohort": "existing",
        "raw_wake": True,
        "systems": {
            "raw": {"fields": {"vehicle": True, "campaign": False}, "words": 8, "errors": 4},
            "compact": {"fields": {"vehicle": False, "campaign": True}, "words": 8, "errors": 1},
        },
    }])
    stats = report["raw-wake"]["compact"]
    assert stats["gains_vs_raw"] == stats["losses_vs_raw"] == 1
    assert stats["wer"] == 0.125
