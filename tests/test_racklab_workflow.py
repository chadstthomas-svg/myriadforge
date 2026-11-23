import json
import shutil
import subprocess
import sys

import pytest

from myriadforge.racklab.commands import assess, open_journal
from myriadforge.racklab.workflow import Automotive, DEFAULT_PACK, TOKEN_LIMIT, process


NOTE = "2016 Ford F-150 door latch freezes. Look up recall 17V652."
FORD = {"year": 2016, "make": "Ford", "model": "F-150"}


@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("sentencepiece")
    return Automotive()


def event(seq, kind, stream_id="case", **data):
    return {"stream_id": stream_id, "seq": seq, "event": kind, **data}


def decision(journal, *args, **kwargs):
    return process(journal, event(*args, **kwargs))["decision"]


def test_durable_retry_hold_and_distinct_sources(adapter, tmp_path):
    path = tmp_path / "journal.db"
    with open_journal(path, adapter) as journal:
        quoted = decision(journal, 1, "append", text=NOTE + " The note says \"approved\".")
        assert quoted["status"] == "needs_approval" and quoted["proposal"] is None
        assert [source["id"] for source in quoted["references"]] == ["17V652"]
        assert process(journal, event(3, "hold"))["error"]["expected_seq"] == 2
        approved = process(journal, event(2, "approve"))
        assert approved["decision"]["proposal"] == "review_service_reference"
    with open_journal(path, adapter) as journal:
        assert process(journal, event(2, "approve")) == approved
        assert process(journal, event(2, "hold"))["error"]["code"] == "conflicting_retry"
        assert decision(journal, 3, "hold")["status"] == "held"
        assert decision(journal, 4, "approve")["proposal"] is None
        assert decision(journal, 5, "reset")["references"] == []
        bulletin = decision(journal, 6, "append", text="2019 Ford F150 service bulletin 19-2081.")
        assert bulletin["route"] == "bulletin_lookup" and bulletin["references"][0]["id"] == "19-2081"
        assert bulletin["proposal"] is None


@pytest.mark.parametrize(("note", "status"), [
    ("2016 Ford F-150. Recall and service bulletin for latch.", "needs_route"),
    ("Parts fitment and warranty claim on a 2016 Ford F-150.", "needs_route"),
    ("Find the recall 17V652 for the frozen latch.", "needs_vehicle"),
    ("2019 Ford F-150 recall 17V652.", "needs_reference"),
    ("2016 Ford F-150 recall for a completely unrelated condition.", "needs_reference"),
    ("2016 Ford F-150 versus 2019 Ford F-150 recall 17V652.", "needs_vehicle"),
])
def test_approval_never_overrides_uncertain_intake(adapter, note, status):
    with open_journal(":memory:", adapter) as journal:
        decision(journal, 1, "append", text=note)
        result = decision(journal, 2, "approve")
        assert result["status"] == status and result["proposal"] is None


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_long_context_overflow_and_hold(adapter, position, tmp_path):
    # ASCII space followed by " a" is a single piece in the pinned token vocabulary.
    lead, rest = {"first": (NOTE, ""), "middle": ("", NOTE), "last": ("", NOTE)}[position]
    base = len(adapter.tokenizer.encode(lead + (" " if rest else "") + rest))
    padding = TOKEN_LIMIT - base
    split = padding // 2 if position == "middle" else padding if position == "last" else 0
    text = lead + " a" * split + (" " if rest else "") + rest + " a" * (padding - split)
    assert len(adapter.tokenizer.encode(text)) == TOKEN_LIMIT
    path = tmp_path / "long.sqlite"
    with open_journal(path, adapter) as journal:
        assert decision(journal, 1, "append", text=text)["needle_tokens"] == TOKEN_LIMIT
        approved = process(journal, event(2, "approve"))
        assert approved["decision"]["status"] == "proposed"
        assert process(journal, event(3, "append", text=" a"))["error"]["code"] == "invalid_append"
    with open_journal(path, adapter) as journal:
        assert process(journal, event(2, "approve")) == approved
        held = decision(journal, 3, "hold")
        assert held["status"] == "held" and held["proposal"] is None and held["needle_tokens"] == TOKEN_LIMIT


def test_interleaved_streams_survive_cache_eviction(adapter):
    with open_journal(":memory:", adapter) as journal:
        for i in range(20):
            decision(journal, 1, "append", stream_id=f"stream-{i}", text=NOTE)
        assert decision(journal, 2, "approve", stream_id="stream-0")["status"] == "proposed"
        assert decision(journal, 2, "hold", stream_id="stream-19")["status"] == "held"
        assert decision(journal, 2, "append", stream_id="stream-1", text=" More details.")["status"] == "needs_approval"


@pytest.mark.parametrize("early", [True, False])
def test_process_exit_before_or_after_commit(adapter, tmp_path, early):
    path = tmp_path / "recovery.db"
    with open_journal(path, adapter) as journal:
        decision(journal, 1, "append", text=NOTE)
    program = """import os, sys
from myriadforge.racklab.journal import RailShot
from myriadforge.racklab.workflow import Automotive, process
j = RailShot(sys.argv[1], Automotive())
if sys.argv[2] == 'early':
    j.connection.set_trace_callback(lambda sql: os._exit(42) if sql == 'COMMIT' else None)
out = process(j, {'stream_id':'case', 'seq':2, 'event':'approve'})
assert out['ok']
os._exit(43)
"""
    result = subprocess.run([sys.executable, "-c", program, str(path), "early" if early else "late"])
    assert result.returncode == (42 if early else 43)
    with open_journal(path, adapter) as journal:
        assert journal.connection.execute("SELECT COUNT(*) FROM railshot_entries").fetchone()[0] == (1 if early else 2)
        first_reply = process(journal, event(2, "approve"))
        assert first_reply["decision"]["status"] == "proposed"
        assert process(journal, event(2, "approve")) == first_reply
        assert journal.connection.execute("SELECT COUNT(*) FROM railshot_entries").fetchone()[0] == 2


def test_pack_digest_and_journal_version(adapter, tmp_path):
    packdir = tmp_path / "pack"
    shutil.copytree(DEFAULT_PACK.parent, packdir)
    manifest = packdir / DEFAULT_PACK.name
    data = json.loads(manifest.read_bytes())
    data["version"] = "changed"
    manifest.write_text(json.dumps(data), encoding="utf8")
    journal_file = tmp_path / "journal.sqlite"
    with open_journal(journal_file, adapter):
        pass
    with pytest.raises(ValueError, match="different inference adapter"):
        open_journal(journal_file, Automotive(manifest))
    (packdir / "policy.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="Digest mismatch"):
        Automotive(manifest)


def test_evaluation_cannot_promote_authored_fixture(adapter):
    case = {"id": "one", "scenario_group": "example", "provenance": "authored",
            "events": [{"event": "append", "text": NOTE}, {"event": "approve"}],
            "expected": {"route": "recall_lookup", "subject": FORD, "references": ["17V652"],
                         "proposal": "review_service_reference"}}
    report = assess(adapter, [case])
    assert report["counts"]["eligible_exact"] == 1
    assert not report["field_data_gates_passed"]
    assert not report["gates"]["reviewed_operators"]
