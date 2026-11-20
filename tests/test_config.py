"""Config tests: devices.toml parsing for the Myriad lanes."""

from __future__ import annotations

from pathlib import Path

from myriadforge.devices.registry import load_config


def test_load_config_missing_file(tmp_path: Path) -> None:
    assert load_config(tmp_path / "nope.toml") == {}


def test_load_config_parses_lanes(tmp_path: Path) -> None:
    cfg_file = tmp_path / "devices.toml"
    cfg_file.write_text(
        '[ncs1]\nrequests = 4\n\n[ncs2]\nshaves = 8\n\n[sim.myriad2]\nshave_count = 12\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg["ncs1"]["requests"] == 4
    assert cfg["ncs2"]["shaves"] == 8
    assert cfg["sim"]["myriad2"]["shave_count"] == 12
