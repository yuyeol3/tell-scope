import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from score import is_korean, ko_detect, passage, plain, runs  # noqa: E402


def test_runs_merges_adjacent_tokens_and_sorts_by_strength():
    assert runs([(5, 1.0), (2, 0.5), (3, 2.0), (9, 1.5)]) == [(2, 3, 2.0), (9, 9, 1.5), (5, 5, 1.0)]


def test_passage_marks_core_with_context_and_flattens_newlines():
    tokens = ["a", "b", "\n", "c", "d", "e"]
    assert passage(tokens, 3, 4, 2) == "...b / [[cd]]e..."


def test_plain_strips_markdown_and_carriage_returns():
    text = "# 제목\r\n\r\n- [x] **굵게** 항목\r\n1. `코드` [링크](http://x)\r\n---\r\n본문"
    assert plain(text) == "제목\n\n굵게 항목\n코드 링크\n본문"


def test_is_korean():
    assert is_korean("이 PR은 버그를 고칩니다. fix typo")
    assert not is_korean("This PR fixes a bug in the 파서.")
    assert not is_korean("1234 !!")


def test_ko_detect_importance_is_weight_times_log1p_activation():
    d = 6
    x = torch.zeros(1, d)
    x[0, 1], x[0, 4] = 2.0, 0.5
    token_feats = torch.zeros(3, d)
    token_feats[0, 1], token_feats[2, 1], token_feats[1, 4] = 2.0, 1.0, 0.5
    detector = SimpleNamespace(extract=lambda text: SimpleNamespace(x=x, token_feats=token_feats.to_sparse(), tokens=["가", "나", "다"]))
    head = {"bias": -0.3, "features": [
        {"id": 1, "weight": 0.8, "description": "AI쪽"},
        {"id": 4, "weight": -1.2, "description": "사람쪽"},
        {"id": 5, "weight": 9.0, "description": "안 켜짐"},
    ]}
    logit, feats, tokens = ko_detect(detector, head, "본문")
    assert math.isclose(logit, -0.3 + 0.8 * math.log1p(2.0) - 1.2 * math.log1p(0.5), rel_tol=1e-6)
    assert [f.id for f in feats] == [1, 4]  # inactive feature dropped, |importance| descending
    assert math.isclose(feats[0].importance, 0.8 * math.log1p(2.0), rel_tol=1e-6)
    assert sorted(feats[0].highlights) == [(0, 2.0), (2, 1.0)]
    assert feats[1].highlights == [(1, 0.5)]
    assert tokens == ["가", "나", "다"]
