import json

import pytest

from myriadforge.myriad_x import allocation_properties, load_tuned_allocation, model_fingerprint


class MyriadX:
    available_devices = ["CPU", "MYRIAD"]

    def get_property(self, device, key):
        assert (device, key) == ("MYRIAD", "FULL_DEVICE_NAME")
        return "Intel Movidius Myriad X VPU"


@pytest.mark.parametrize("allocation", [(8, 4, 1), (9, 9, 2), (0, 8, 1), (4, 8, 3)])
def test_allocation_rejects_invalid_plugin_resources(allocation):
    with pytest.raises(ValueError):
        allocation_properties(*allocation)


def test_profile_is_bound_to_companion_weights_and_runtime(tmp_path):
    xml = tmp_path / "model.xml"
    weights = xml.with_suffix(".bin")
    xml.write_bytes(b"graph")
    weights.write_bytes(b"weights")
    settings = {"shaves": 4, "cmx_slices": 8, "streams": 2}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "schema": "myriadforge.ncs2-tuning/1",
        "model_sha256": model_fingerprint(xml),
        "runtime": "2022.3.1",
        "device": "Intel Movidius Myriad X VPU",
        "recommended": {"throughput": settings},
    }), encoding="utf8")

    assert load_tuned_allocation(path, xml, MyriadX(), "2022.3.1", "throughput") == settings
    with pytest.raises(ValueError, match="OpenVINO runtime"):
        load_tuned_allocation(path, xml, MyriadX(), "another version", "throughput")
    weights.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="Tune this model"):
        load_tuned_allocation(path, xml, MyriadX(), "2022.3.1", "throughput")
