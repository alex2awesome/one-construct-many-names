from pathlib import Path

import pytest

from methods.codability.lexicon import partial_regroup as p


def test_jsonl_loader_rejects_non_objects(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("[]\n")
    with pytest.raises(ValueError):
        p._load_jsonl(path)


def test_jsonl_loader_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "ok.jsonl"
    path.write_text('\n{"a": 1}\n')
    assert p._load_jsonl(path) == [{"a": 1}]
