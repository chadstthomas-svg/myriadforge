"""HuggingFace model intake, prebuilt-first.

Resolves an HF model page (URL or ``org/name``), scans it for ready-to-run
artifacts (LiteRT ``.tflite``/``.litertlm``/``.task`` for host lanes, plus
``.onnx`` for the NCS2 lane per the NCS2_ROADMAP registry note), downloads
the chosen artifact into ``models/`` and records it in ``registry.json``.
Conversion of repos without prebuilts (litert-torch / litert-lm-builder)
arrives with Phase 2 and runs in a Linux environment (WSL2 or GitHub
Actions) because Windows conversion is unsupported upstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

# Candidate model-file extensions, best rank first (ONNX ranks last: it is
# only the NCS2 lane's intake, so LiteRT prebuilts always win when present).
_FORMAT_RANK = {".litertlm": 0, ".task": 1, ".tflite": 2, ".onnx": 3}
_REGISTRY_NAME = "registry.json"


@dataclass
class IngestResult:
    repo_id: str
    status: str  # "ok" | "convert-pending"
    file: str | None = None
    local_path: Path | None = None
    format: str | None = None
    candidates: list[str] = field(default_factory=list)
    message: str = ""


def parse_repo_id(spec: str) -> str:
    """Accept a full HF URL, or a bare ``org/name`` repo id."""
    spec = spec.strip()
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/#?]+)", spec)
    if m:
        return m.group(1).rstrip("/")
    if re.fullmatch(r"[\w.-]+/[\w.-]+", spec):
        return spec
    raise ValueError(
        f"cannot parse {spec!r}: pass a huggingface.co URL or an 'org/name' repo id"
    )


def list_model_files(repo_id: str) -> list[str]:
    """All files in the repo that look like runnable model artifacts."""
    files = HfApi().list_repo_files(repo_id)
    return sorted(
        (f for f in files if Path(f).suffix.lower() in _FORMAT_RANK),
        key=lambda f: (_FORMAT_RANK[Path(f).suffix.lower()], f),
    )


def models_dir(base: Path = Path("models")) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_registry(base: Path = Path("models")) -> dict[str, Any]:
    path = base / _REGISTRY_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": "myriadforge.registry/1", "entries": {}}


def save_registry(reg: dict[str, Any], base: Path = Path("models")) -> None:
    path = base / _REGISTRY_NAME
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")


def ingest_prebuilt(spec: str, base: Path = Path("models"), pick: str | None = None) -> IngestResult:
    """Download a prebuilt model artifact from an HF repo into the registry.

    pick: explicit filename in the repo to download; defaults to the
    highest-ranked (litertlm > task > tflite > onnx), then largest, artifact.
    """
    repo_id = parse_repo_id(spec)
    candidates = list_model_files(repo_id)
    if not candidates:
        return IngestResult(
            repo_id=repo_id,
            status="convert-pending",
            message=(
                "no prebuilt artifacts (.tflite/.litertlm/.task/.onnx) in this repo. "
                "LiteRT conversion (litert-torch) requires a Linux environment - "
                "Phase 2 adds a WSL2 / GitHub Actions conversion worker."
            ),
        )

    if pick:
        if pick not in candidates:
            raise ValueError(f"{pick!r} is not a model file in {repo_id}; candidates: {candidates}")
        chosen = pick
    else:
        # highest rank, then largest file wins
        chosen = max(candidates, key=lambda f: (
            -_FORMAT_RANK[Path(f).suffix.lower()],
            _repo_file_size(repo_id, f),
        ))

    local_dir = models_dir(base) / repo_id.replace("/", "__")
    path = Path(hf_hub_download(repo_id=repo_id, filename=chosen, local_dir=local_dir))

    reg = load_registry(base)
    reg["entries"][repo_id] = {
        "file": chosen,
        "local_path": str(path),
        "format": path.suffix.lstrip("."),
        "candidates": candidates,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
    }
    save_registry(reg, base)

    return IngestResult(
        repo_id=repo_id, status="ok", file=chosen, local_path=path,
        format=path.suffix.lstrip("."), candidates=candidates,
        message=f"downloaded {chosen} -> {path}",
    )


def resolve_local_model(spec: str, base: Path = Path("models")) -> Path:
    """Resolve a bench target (path, HF URL/repo id, registry key) to a file."""
    p = Path(spec)
    if p.exists():
        return p
    try:
        repo_id = parse_repo_id(spec)
    except ValueError as exc:
        raise FileNotFoundError(f"model not found: {spec}") from exc
    reg = load_registry(base)
    entry = reg["entries"].get(repo_id)
    if entry and Path(entry["local_path"]).exists():
        return Path(entry["local_path"])
    raise FileNotFoundError(
        f"{repo_id} not in registry - run `myriadforge ingest {repo_id}` first"
    )


def _repo_file_size(repo_id: str, filename: str) -> int:
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        for sib in info.siblings or []:
            if sib.rfilename == filename:
                return sib.size or 0
    except Exception:
        pass
    return 0
