import numpy as np
import pytest

from apps.embeddinggemma2.__main__ import compare, normalize, prompt, rank


def test_retrieval_prompts_and_kind():
    assert prompt({"kind": "query", "text": "Open recall?"}) == "task: search result | query: Open recall?"
    assert prompt({"text": "Read this"}) == "title: none | text: Read this"
    assert prompt({"text": "Read this", "title": "Note"}) == "title: Note | text: Read this"
    with pytest.raises(ValueError, match="kind"):
        prompt({"kind": "other", "text": "Read this"})


def test_mrl_truncates_before_normalizing():
    values = np.array([[3, 4, 12], [0, 2, 3]], dtype=np.float32)
    np.testing.assert_allclose(normalize(values, 2), [[.6, .8], [0, 1]])
    with pytest.raises(ArithmeticError, match="zero"):
        normalize(np.array([[0, 0, 1]]), 2)


def test_ranking_excludes_queries_and_keeps_stable_ties():
    rows = [{"id": "first"}, {"id": "second"}, {"id": "query", "kind": "query", "expected": "first"}]
    values = np.zeros((3, 768), np.float32)
    values[:, 0] = 1
    assert [hit["id"] for hit in rank(rows, values, 2)["query"]] == ["first", "second"]
    evidence = compare(rows, values, values.copy())
    assert all(row["ranking_top1_matches"] == row["labeled_top1_matches"] == 1 for row in evidence.values())


def test_query_only_parity_has_no_ranking_gate():
    rows = [{"id": "query", "kind": "query"}]
    values = np.ones((1, 768), dtype=np.float32)
    evidence = compare(rows, values, values.copy())
    assert all(row["ranking_top1_matches"] == row["queries_with_catalog"] == 0 for row in evidence.values())
