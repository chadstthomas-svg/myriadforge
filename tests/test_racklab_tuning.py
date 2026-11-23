import pytest

from myriadforge.racklab_tuning import evaluate, fit, infer


def case(key, note, route, *targets, group=None):
    return {
        "id": key, "scenario_group": group or key, "market": "fleet",
        "note": note, "route": route, "targets": list(targets),
    }


@pytest.fixture
def training_pack():
    pytest.importorskip("sklearn")
    pack = {
        "sources": {
            "brakes": ["Brakes", "https://example.com/brakes", "Brake pedal service"],
            "battery": ["Battery", "https://example.com/battery", "Battery charge level"],
        },
        "workflows": [
            {"id": "brake", "market": "fleet", "keywords": ["pedal"], "target": ["brakes"]},
            {"id": "charge", "market": "fleet", "keywords": ["battery"], "target": ["battery"]},
        ],
    }
    training = [
        case("f1", "Soft brake pedal", "brake", "brakes"),
        case("f2", "Brake pedal fluid", "brake", "brakes"),
        case("f3", "Battery charge low", "charge", "battery"),
        case("f4", "Battery pack charging", "charge", "battery"),
    ]
    selection = [
        case("s1", "Pedal needs brake service", "brake", "brakes"),
        case("s2", "Battery needs overnight charge", "charge", "battery"),
    ]
    return pack, training, selection


def test_binary_adapter_ties_keep_baseline_and_quantized_score_matches(training_pack):
    model = fit(*training_pack)
    assert model["selection"]["route"] == "keywords"
    assert model["weights"]["router_fp32_bytes"] == model["weights"]["router_int8_bytes"] * 4
    for mode in ("selected", "learned", "int8"):
        result = infer(model, "fleet", "Battery charge level dropped", mode, "selected")
        assert result["suggested_route"] == "charge"
        assert result["references"][0]["id"] == "battery"
        assert result["requires_policy_review"] is True
    assert infer(model, "fleet", "unknown unrelated words", "learned", "selected")["suggested_route"] is None


def test_evaluation_rejects_shared_group_and_normalized_fit_note(training_pack):
    model = fit(*training_pack)
    with pytest.raises(ValueError, match="overlap detected for group"):
        evaluate(model, [case("e1", "Inspect the brake pedal", "brake", "brakes", group="f1")])
    with pytest.raises(ValueError, match="overlap detected for note"):
        evaluate(model, [case("e2", "SOFT, brake pedal!", "brake", "brakes")])
    result = evaluate(model, [case("e3", "Inspect the brake pedal", "brake", "brakes")])
    assert result["cases"] == result["by_market"]["fleet"]["learned_routes"] == 1
