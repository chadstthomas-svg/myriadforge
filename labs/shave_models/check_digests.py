"""Verify every bundle manifest matches its artifacts (hardware-free CI gate).

Walks labs/shave_models/bundles/*/model.json: any manifest that records
onnx_sha256 must match its model.onnx exactly, and every models/registry.json
local_path that exists in the checkout must be present. Digest mismatches
fail the run - a moved or rebuilt artifact must re-register, not drift.
"""
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def main() -> int:
    failures = []
    checked = 0
    bundles = LAB / "bundles"
    for manifest_path in sorted(bundles.glob("*/model.json")):
        bundle = manifest_path.parent
        info = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = info.get("onnx_sha256")
        model = bundle / "model.onnx"
        if digest is None:
            continue  # non-ONNX bundles (needle-ui, boolean release archives)
        if not model.exists():
            failures.append(f"{bundle.name}: model.json references onnx_sha256 but model.onnx is missing")
            continue
        checked += 1
        actual = sha256(model)
        if actual != digest:
            failures.append(f"{bundle.name}: onnx digest drift - manifest {digest[:12]} vs artifact {actual[:12]}")

    registry = LAB.parent.parent / "models" / "registry.json"
    if registry.exists():
        reg = json.loads(registry.read_text(encoding="utf-8"))
        for name, entry in reg["entries"].items():
            local = Path(entry["local_path"])
            if local.exists():
                checked += 1
            elif "labs/" in entry["local_path"].replace("\\", "/"):
                # in-repo bundle artifacts must exist; downloaded cache paths
                # (models/** is git-ignored) are recorded, not enforced
                failures.append(f"registry entry {name}: in-repo local_path missing -> {local}")

    print(f"digest check: {checked} artifacts verified, {len(failures)} failures")
    for f in failures:
        print("  FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
