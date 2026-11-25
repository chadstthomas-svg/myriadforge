import numpy as np

from apps.static_embedder import StaticEncoder, fit_lexical_weights, lexical_counts
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
