"""Extract the Intel C++ runtime DLLs from an intel-cmplr-lib-rt wheel.

Usage: python _extract_redist.py <wheel.whl> <dest_dir>
The 2020.3 IE DLLs (inference_engine_legacy.dll, myriadPlugin.dll) import
libmmd.dll + svml_dispmd.dll, which Intel does not bundle in the OpenVINO
archive. The intel-cmplr-lib-rt pip wheel ships them; we extract just those.
"""
import sys
import zipfile
from pathlib import Path

WANTED = {"libmmd.dll", "svml_dispmd.dll", "libircmd.dll"}


def main() -> int:
    wheel, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(wheel)
    for n in z.namelist():
        base = n.split("/")[-1]
        if base in WANTED:
            (dest / base).write_bytes(z.read(n))
            print(f"extracted {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
