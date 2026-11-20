"""Extract the needed OpenVINO 2020.3.2 trees from the dev archive.

Usage: python _extract_ov.py <archive.zip> <dest_dir>
Pulls inference_engine (DLLs + myriadPlugin + usb-ma2450 firmware),
model_optimizer, ngraph (ngraph.dll), the python3.7 IE API, and setupvars.bat.
"""
import sys
import zipfile
from pathlib import Path

PREFIXES = (
    "deployment_tools/inference_engine/",
    "deployment_tools/model_optimizer/",
    "deployment_tools/ngraph/",
    "python/",
    "bin/setupvars.bat",
)


def main() -> int:
    archive, dest = Path(sys.argv[1]), Path(sys.argv[2])
    z = zipfile.ZipFile(archive)
    n = 0
    for info in z.infolist():
        if any(info.filename.startswith(p) or info.filename == p for p in PREFIXES):
            z.extract(info, dest)
            n += 1
    print(f"extracted {n} entries -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
