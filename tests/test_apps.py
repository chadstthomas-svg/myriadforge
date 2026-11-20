"""App-pilot tests: the WS5 contracts without hardware (NCS2_ROADMAP WS5).

Runs in the myriadforge main env, so only the stdlib/numpy parts of apps/ are
exercised: the ncs2.event/1 + ncs2.index/1 contracts, the kNN capacity
calculator, the podcast windowing/RMS gate, and synset parsing. OpenVINO-
dependent paths are verified by the pilots themselves inside the isolated
venv (tools/ncs2/demo_mvp.py), not here.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.common.events import EVENT_SCHEMA, INDEX_SCHEMA, IndexWriter, make_event
from apps.gate.synset import load_labels
from apps.knn_engine.capacity import check, max_vectors
from apps.podcast.triage import load_wav, rms_gate, triage, windows


def test_event_contract_shape() -> None:
    event = make_event(ts=1.23456, label="golden retriever", class_index=207,
                       confidence=0.8312, device="MYRIAD", latency_ms=6.90123)
    assert event["schema"] == EVENT_SCHEMA
    assert event["ts"] == 1.235
    assert event["confidence"] == 0.8312
    assert event["latency_ms"] == 6.901
    assert set(event) == {"schema", "ts", "label", "class_index",
                          "confidence", "device", "latency_ms"}


def test_index_writer_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "clip.index.jsonl"
    with IndexWriter(path, source="clip.mp4", model="mobilenetv2-12",
                     device="MYRIAD") as writer:
        writer.write(make_event(ts=0.0, label="ant", class_index=300,
                                confidence=0.5, device="MYRIAD", latency_ms=1.0))
        writer.write(make_event(ts=0.5, label="ant", class_index=300,
                                confidence=0.6, device="MYRIAD", latency_ms=1.1))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = json.loads(lines[0])
    assert header["schema"] == INDEX_SCHEMA
    assert header["source"] == "clip.mp4" and header["model"] == "mobilenetv2-12"
    events = [json.loads(ln) for ln in lines[1:]]
    assert len(events) == 2 and events[1]["ts"] == 0.5


def test_capacity_fits_and_budget_arithmetic() -> None:
    cap = check(100_000, 256)  # 100k x 256 FP16 = 48.8 MB -> fits easily
    assert cap.fits
    assert cap.as_dict()["table_mb_fp16"] == 48.8

    huge = check(10_000_000, 256)  # 4.9 GB -> over the ~1 GB budget
    assert not huge.fits
    assert huge.max_vectors == max_vectors(256)
    # max vectors must itself fit: n * dim * 2 <= budget
    assert max_vectors(256) * 256 * 2 <= int(1024**3 * 0.9)


def test_load_labels_parses_synset(tmp_path: Path) -> None:
    sidecar = tmp_path / "m.labels.txt"
    sidecar.write_text("n01440764 tench, Tinca tinca\nn01443537 goldfish\n\n",
                       encoding="utf-8")
    labels = load_labels(sidecar)
    assert labels == ["tench, Tinca tinca", "goldfish"]
    assert load_labels(tmp_path / "missing.txt") is None
    assert load_labels(None) is None


def _write_sine_wav(path: Path, rate: int = 8000, seconds: float = 2.0) -> None:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())


def test_rms_gate_and_windowing(tmp_path: Path) -> None:
    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path)
    rate, samples = load_wav(wav_path)
    assert rate == 8000 and len(samples) == 16000

    wins = list(windows(rate, samples, window_s=1.0, hop_s=0.5))
    assert len(wins) == 3  # 2.0 s audio, 1.0 s hop -> starts 0.0/0.5/1.0
    fired, rms = rms_gate(wins[0][1], threshold=500.0)
    assert fired and rms > 1000  # 20 kHz-amplitude sine is loud
    silent, rms0 = rms_gate(np.zeros(8000, dtype=np.float32), threshold=500.0)
    assert not silent and rms0 == 0.0


def test_podcast_triage_end_to_end_pure(tmp_path: Path) -> None:
    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path)
    summary = triage(wav_path, window_s=1.0, hop_s=0.5, out_dir=tmp_path / "res")
    assert summary["windows_gated"] == 3
    assert summary["events_fired"] == 3  # a loud tone fires every window
    index_path = Path(summary["index"])
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["schema"] == INDEX_SCHEMA
    events = [json.loads(ln) for ln in lines[1:]]
    assert all(ev["label"] == "speech" and ev["fired"] for ev in events)
