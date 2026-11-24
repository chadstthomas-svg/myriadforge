"""OpenVINO 2022.3 compilation for synchronous Myriad X callers.

This module is importable in the isolated legacy Python environment.
"""

import hashlib
import json
from pathlib import Path


def prepare_static_model(core, model_path):
    from openvino.runtime import Dimension

    model = core.read_model(str(model_path))
    batch_one = {}
    for index, port in enumerate(model.inputs):
        shape = port.get_partial_shape()
        if (shape.rank.is_static and len(shape) and shape[0].is_dynamic
                and shape[0].compatible(Dimension(1))
                and all(dim.is_static for dim in list(shape)[1:])):
            shape[0] = Dimension(1)
            batch_one[index] = shape
    if batch_one:
        model.reshape(batch_one)
    return model


def allocation_properties(shaves, cmx_slices, streams):
    if type(streams) is not int or streams not in (1, 2) or not all(
        type(count) is int and 1 <= count <= 16 // streams for count in (shaves, cmx_slices)
    ) or shaves > cmx_slices:
        raise ValueError("Myriad X needs SHAVEs <= CMX slices and at most 16 slices across one or two streams")
    return {
        "MYRIAD_THROUGHPUT_STREAMS": streams,
        "MYRIAD_NUMBER_OF_SHAVES": shaves,
        "MYRIAD_NUMBER_OF_CMX_SLICES": cmx_slices,
    }


def require_myriad_x(core):
    visible = [name for name in core.available_devices if name.split(".")[0] == "MYRIAD"]
    if visible != ["MYRIAD"]:
        raise RuntimeError("Expected exactly one visible MYRIAD device, found %s" % visible)
    name = core.get_property("MYRIAD", "FULL_DEVICE_NAME")
    if "Myriad X" not in name:
        raise RuntimeError("Expected a physical Myriad X, found %s" % name)
    return name


def model_fingerprint(model_path):
    file = Path(model_path)
    files = (file, file.with_suffix(".bin")) if file.suffix.lower() == ".xml" else (file,)
    fingerprint = {}
    for part in files:
        digest = hashlib.sha256()
        with part.open("rb") as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(data)
        fingerprint[part.suffix.lower()] = digest.hexdigest()
    return fingerprint


def load_tuned_allocation(profile, model_path, core, runtime, objective):
    report = json.loads(Path(profile).read_text(encoding="utf8"))
    if report.get("schema") != "myriadforge.ncs2-tuning/1":
        raise ValueError("Expected a MyriadForge NCS2 tuning report")
    identity = (model_fingerprint(model_path), require_myriad_x(core), runtime)
    if identity != (report.get("model_sha256"), report.get("device"), report.get("runtime")):
        raise ValueError("Tune this model and OpenVINO runtime before applying the profile")
    allocation = report.get("recommended", {}).get(objective)
    if not allocation:
        raise ValueError("The report has no confirmed allocation for " + objective)
    allocation_properties(**allocation)
    return allocation


def compile_myriad_x(core, ir, shaves, *, cmx_slices=None, streams=1):
    cmx_slices = shaves if cmx_slices is None else cmx_slices
    props = allocation_properties(shaves, cmx_slices, streams)
    return core.compile_model(prepare_static_model(core, ir), "MYRIAD", props)
