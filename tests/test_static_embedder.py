import numpy as np
import pytest

from apps.static_embedder import StaticEncoder, fit_lexical_weights, lexical_counts
from apps.static_embedder.contextual import (SCHEMA, adapter_shapes, grouped_features, load_adapter,
                                              numpy_context)
from tools.ncs2.measure_contextual_embedder import relevance
from tools.ncs2.measure_static_embedder import retrieval


def test_frequency_weights_and_signed_counts():
    single = lexical_counts("Filter")
    repeated = lexical_counts("filter filter filter filter")
    np.testing.assert_array_equal(repeated, 2 * single)
    bucket = np.flatnonzero(single)[0]
    weights = fit_lexical_weights(["filter", "filter"])
    assert weights[bucket] < weights.max()


def test_hybrid_reference_has_unit_norm_without_lexical_terms():
    encoder = StaticEncoder.__new__(StaticEncoder)
    encoder.table = np.array([[3, 4]], dtype=np.float16)
    encoder.lexical = fit_lexical_weights(["filter", "rotor"])
    for text in ("filter", "8"):
        vector = encoder.encode_cpu([0], lexical_counts(text))
        assert np.isclose(np.linalg.norm(vector), 1)
    np.testing.assert_allclose(encoder.encode_cpu([0], lexical_counts("8"))[:2], [0.6, 0.8])


def test_two_positive_documents_contribute_to_recall_and_ndcg():
    rows = [{"id": name} for name in ("a", "b", "c")] + [{"id": "q", "kind": "query"}]
    labels = [{"query-id": "q", "corpus-id": name} for name in ("b", "c")]
    embeddings = np.array([[1, 0], [0.5, 0.5], [0, 1], [1, 0]])
    report, _ = retrieval(rows, labels, embeddings)
    assert report["top1_correct"] == 0
    assert report["mrr_at_10"] == 0.5
    assert report["recall_at_10"] == 1
    assert np.isclose(report["ndcg_at_10"], (1 / np.log2(3) + 0.5) / (1 + 1 / np.log2(3)))


def test_context_features_distinguish_order_and_respect_capacity():
    forward, mask = grouped_features([1, 2, 3, 4])
    reverse, _ = grouped_features([4, 3, 2, 1])
    np.testing.assert_array_equal(forward[:, :128], reverse[:, :128])
    assert not np.array_equal(forward[:, 128:], reverse[:, 128:])
    assert mask.sum() == 1
    first = grouped_features(list(range(128)))
    longer = grouped_features(list(range(160)))
    np.testing.assert_array_equal(first[0], longer[0])
    np.testing.assert_array_equal(first[1], longer[1])


def test_tied_adapter_roundtrip_and_padding(tmp_path):
    rng = np.random.default_rng(42)
    tied = adapter_shapes("tied")
    assert "q1" not in tied and "q1" in adapter_shapes("untied")
    bundle = tmp_path / "adapter.npz"
    arrays = {name: rng.normal(0, .05, shape).astype(np.float32) for name, shape in tied.items()}
    np.savez(bundle, schema=SCHEMA, variant="tied", **arrays)
    features, mask = grouped_features([1, 2, 3])
    adapter = load_adapter(bundle)
    expected = numpy_context(features, mask, adapter)
    features[mask[:, 0] == 0, :3] = 1
    np.testing.assert_allclose(numpy_context(features, mask, adapter), expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(expected), 1)
    del arrays["q0"]
    np.savez(bundle, schema=SCHEMA, variant="tied", **arrays)
    with pytest.raises(ValueError, match="missing required weights"):
        load_adapter(bundle)


def test_context_benchmark_accounts_for_graded_relevance():
    fixture = {"rows": [{"id": "higher", "kind": "document"}, {"id": "lower", "kind": "document"},
                        {"id": "q", "kind": "query"}],
               "qrels": [{"query-id": "q", "corpus-id": "higher", "score": "2"},
                         {"query-id": "q", "corpus-id": "lower", "score": "1"}]}
    report = relevance(fixture, np.array([[0.5, 0.9]]))
    assert report["recall_at_10"] == 1
    assert np.isclose(report["ndcg_at_10"], (1 + 2 / np.log2(3)) / (2 + 1 / np.log2(3)))
