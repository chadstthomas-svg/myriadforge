"""ImageNet synset sidecar loading for gate label resolution.

The ONNX model zoo classification models label indices with the shared
``synset.txt`` (``<wnid> <comma-joined names>`` per line). We store it next
to the gate model as ``<model>.labels.txt`` and keep it out of the registry
(plain data, like the model artifacts themselves).
"""

from __future__ import annotations

from pathlib import Path


def load_labels(path: Path | str | None) -> list[str] | None:
    """Parse a synset sidecar; returns None when the file is absent."""
    if not path or not Path(path).exists():
        return None
    labels: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        labels.append(parts[1] if len(parts) > 1 else parts[0])
    return labels
