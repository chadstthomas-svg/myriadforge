"""Persist ordered inference requests and their stable results in SQLite."""

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3


RESULT_SCHEMA = "railshot.result/1"
STREAM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
MAX_SEQUENCE = (1 << 63) - 1
RESIDENT_STREAMS = 16


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class StreamError(ValueError):
    def __init__(self, code, message, expected_seq=None):
        super().__init__(message)
        self.code = code
        self.expected_seq = expected_seq


@dataclass(frozen=True)
class Request:
    stream_id: str
    seq: int
    append: str
    reset: bool

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or value.keys() - {"stream_id", "seq", "append", "reset"}:
            raise StreamError("invalid_request", "Expected stream_id, seq, append, and optional reset")
        stream_id = value.get("stream_id")
        if not isinstance(stream_id, str) or STREAM_ID.fullmatch(stream_id) is None:
            raise StreamError("invalid_request", "Invalid stream_id")
        seq = value.get("seq")
        if type(seq) is not int or not 1 <= seq <= MAX_SEQUENCE:
            raise StreamError("invalid_request", "seq must be a positive signed 64-bit integer")
        reset = value.get("reset", False)
        fragment = value.get("append", "" if reset is True else None)
        if type(reset) is not bool or not isinstance(fragment, str):
            raise StreamError("invalid_request", "Use an append string or reset=true")
        try:
            fragment.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise StreamError("invalid_request", "append must contain valid Unicode text") from exc
        return cls(stream_id, seq, fragment, reset)


class RailShot:
    def __init__(self, path, adapter):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.adapter = adapter
        self.resident = OrderedDict()
        try:
            self.connection.execute("PRAGMA synchronous = FULL")
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS railshot_metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self.connection.execute(
                    """CREATE TABLE IF NOT EXISTS railshot_entries (
                        stream_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        fragment TEXT NOT NULL,
                        reset INTEGER NOT NULL CHECK (reset IN (0, 1)),
                        response TEXT NOT NULL,
                        PRIMARY KEY (stream_id, seq)
                    ) WITHOUT ROWID"""
                )
                self.connection.execute(
                    "INSERT OR IGNORE INTO railshot_metadata VALUES ('adapter/v1', ?)", (adapter.identity,)
                )
                (identity,) = self.connection.execute(
                    "SELECT value FROM railshot_metadata WHERE name = 'adapter/v1'"
                ).fetchone()
                if identity != adapter.identity:
                    raise StreamError("adapter_mismatch", "This journal belongs to a different inference adapter")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.connection.close()

    @staticmethod
    def _decision(stream, request):
        outcome = stream.append(request.append, reset=request.reset)
        # Runtime timings are observations; durable decisions must compare exactly on replay.
        return {
            "schema": RESULT_SCHEMA,
            "ok": True,
            "stream_id": request.stream_id,
            "seq": request.seq,
            "decision": {key: value for key, value in outcome.items() if not key.endswith("_ms")},
        }

    def _restore(self, stream_id, last_seq):
        replay_after, stream = self.resident.get(stream_id, (0, None))
        if stream is None:
            stream = self.adapter.open_stream()
        if replay_after == last_seq:
            return stream
        # Reset starts a new document. Older rows still serve idempotent retries.
        (last_reset,) = self.connection.execute(
            "SELECT MAX(seq) FROM railshot_entries WHERE stream_id = ? AND reset = 1 AND seq > ?",
            (stream_id, replay_after),
        ).fetchone()
        if last_reset is not None:
            stream = self.adapter.open_stream()
            replay_after = last_reset - 1
        rows = self.connection.execute(
            """SELECT seq, fragment, reset, response FROM railshot_entries
            WHERE stream_id = ? AND seq > ? ORDER BY seq""",
            (stream_id, replay_after),
        )
        for seq, fragment, reset, recorded in rows:
            request = Request(stream_id, seq, fragment, bool(reset))
            try:
                actual = encode(self._decision(stream, request))
            except Exception as exc:
                raise StreamError("replay_failed", f"Unable to replay committed seq {seq}") from exc
            if seq != replay_after + 1 or recorded != actual:
                raise StreamError("replay_mismatch", f"Committed result disagrees at seq {seq}")
            replay_after = seq
        return stream

    def process(self, request):
        db = self.connection
        stream_id = request.stream_id
        state_used = False
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute(
                    "SELECT fragment, reset, response FROM railshot_entries WHERE stream_id = ? AND seq = ?",
                    (stream_id, request.seq),
                ).fetchone()
                if existing is not None:
                    fragment, reset, response = existing
                    if (request.append, request.reset) != (fragment, bool(reset)):
                        raise StreamError("conflicting_retry", "This seq already has a different request")
                    return json.loads(response)

                (last_seq,) = db.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM railshot_entries WHERE stream_id = ?", (stream_id,)
                ).fetchone()
                if request.seq != last_seq + 1:
                    raise StreamError("out_of_order", "Expected the next contiguous seq", last_seq + 1)

                state_used = True
                stream = self.adapter.open_stream() if request.reset else self._restore(stream_id, last_seq)
                try:
                    result = self._decision(stream, request)
                    encoded = encode(result)
                except ValueError as exc:
                    raise StreamError("invalid_append", str(exc), request.seq) from exc
                except Exception as exc:
                    raise StreamError("inference_failed", str(exc), request.seq) from exc
                db.execute(
                    "INSERT INTO railshot_entries VALUES (?, ?, ?, ?, ?)",
                    (stream_id, request.seq, request.append, request.reset, encoded),
                )
        except BaseException:
            # An inference or commit can fail after mutating a resident stream.
            # The next attempt restores it from SQLite's committed rows.
            if state_used:
                self.resident.pop(stream_id, None)
            raise

        self.resident[stream_id] = (request.seq, stream)
        self.resident.move_to_end(stream_id)
        if len(self.resident) > RESIDENT_STREAMS:
            self.resident.popitem(last=False)
        return result
