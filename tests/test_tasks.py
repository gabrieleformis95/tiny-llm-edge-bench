"""Tests for eval task implementations."""

from __future__ import annotations

import pytest

from src.tasks.json_following import parse_json_lenient


def test_rouge_l_identical():
    from src.tasks.ragas_industrial import _rouge_l_f1
    assert _rouge_l_f1("the cat sat", "the cat sat") == 1.0


def test_rouge_l_partial():
    from src.tasks.ragas_industrial import _rouge_l_f1
    score = _rouge_l_f1("the cat", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_rouge_l_no_overlap():
    from src.tasks.ragas_industrial import _rouge_l_f1
    assert _rouge_l_f1("dog barks loudly", "cat sits quietly") == 0.0


def test_rouge_l_empty():
    from src.tasks.ragas_industrial import _rouge_l_f1
    assert _rouge_l_f1("", "reference text") == 0.0


def test_parse_json_lenient_plain():
    result = parse_json_lenient('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_lenient_fenced():
    result = parse_json_lenient('```json\n{"key": 1}\n```')
    assert result == {"key": 1}


def test_parse_json_lenient_invalid():
    result = parse_json_lenient("not json at all")
    assert result is None


def test_json_following_score_valid(tmp_path):
    import json
    from src.tasks.json_following import JSONFollowingTask

    data = [{"id": "j1", "instruction": "Return JSON with key 'name'.", "required_keys": ["name"]}]
    p = tmp_path / "jf.json"
    p.write_text(json.dumps(data))
    task = JSONFollowingTask(p)
    samples = list(task.iter_samples())
    assert len(samples) == 1
    score = task.score('{"name": "Alice"}', json.dumps(["name"]))
    assert score == 1.0


def test_json_following_score_missing_key(tmp_path):
    import json
    from src.tasks.json_following import JSONFollowingTask

    data = [{"id": "j2", "instruction": ".", "required_keys": ["name", "age"]}]
    p = tmp_path / "jf2.json"
    p.write_text(json.dumps(data))
    task = JSONFollowingTask(p)
    score = task.score('{"name": "Alice"}', json.dumps(["name", "age"]))
    assert score == 0.5


def test_instruction_follow_word_count(tmp_path):
    import json
    from src.tasks.instruction_follow import InstructionFollowTask

    data = [{"id": "i1", "prompt": "Say hi in 3 words.", "constraints": [{"type": "word_count_max", "value": 3}]}]
    p = tmp_path / "if.json"
    p.write_text(json.dumps(data))
    task = InstructionFollowTask(p)
    score = task.score("Hello there friend", json.dumps([{"type": "word_count_max", "value": 3}]))
    assert score == 1.0

    score_fail = task.score("Hello there my dear friend and colleague", json.dumps([{"type": "word_count_max", "value": 3}]))
    assert score_fail == 0.0


def test_mmlu_score(tmp_path):
    import json
    from src.tasks.mmlu_subset import MMLUSubsetTask

    data = [{"id": "m1", "question": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": "B"}]
    p = tmp_path / "mmlu.json"
    p.write_text(json.dumps(data))
    task = MMLUSubsetTask(p, n_samples=1)
    assert task.score("B", "B") == 1.0
    assert task.score("A", "B") == 0.0
    assert task.score("b is the answer", "B") == 1.0
