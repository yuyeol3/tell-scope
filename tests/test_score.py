import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from score import is_korean, ko_detect, passage, plain, rule_tells, runs  # noqa: E402


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


def test_plain_strips_emoji_including_joiners():
    assert plain("# 🚀 시작 ⚙️\n휴식 🙇🏻‍♂️ 필요") == "시작\n휴식  필요"


def test_rule_tells_flags_korean_punctuation_emoji_and_stock_phrases():
    text = "오늘날 개발은 중요합니다 — 정말로.\n설계·구현 🎯\n결론적으로, 성장을 응원합니다."
    rules = [(t["rule"], t["match"]) for t in rule_tells(text)]
    assert rules == [("stock-phrase", "오늘날"), ("em-dash", "—"), ("middle-dot", "·"), ("emoji", "🎯"),
                     ("stock-phrase", "결론적으로")]
    assert rule_tells(text)[1]["line"] == "오늘날 개발은 중요합니다 — 정말로."


def test_rule_tells_english_openers_only_at_sentence_start():
    rules = [t["match"] for t in rule_tells("We leverage it. Moreover, it works. It is not moreover, fine.")]
    assert rules == ["leverage", ". Moreover,"]


def test_rule_tells_clean_text():
    assert rule_tells("그냥 평범하게 쓴 문장이다. 숫자 1-2와 하이픈은 괜찮다.") == []


def test_rule_tells_counts_a_composed_emoji_once():
    assert [t["match"] for t in rule_tells("설정 ⚙️ 완료 🙇🏻‍♂️")] == ["⚙️", "🙇🏻‍♂️"]


def test_is_korean_ignores_code():
    text = "레포 폴더를 연결하세요.\n\n```bash\nln -s \"$PWD\" ~/.claude/skills/tell-scope\nln -s \"$PWD\" ~/.codex/skills/tell-scope\n```\n\n그다음 `python scripts/score.py draft.txt`로 채점합니다."
    assert is_korean(text)
