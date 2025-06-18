import pytest
from agentic.utils.shared import ensure_list, flatten_dict

def test_ensure_list():
    assert ensure_list([1, 2]) == [1, 2]
    assert ensure_list(1) == [1]
    assert ensure_list(None) == []

def test_flatten_dict():
    d = {"a": 1, "b": {"c": 2, "d": 3}}
    flat = flatten_dict(d)
    assert flat == {"a": 1, "b.c": 2, "b.d": 3}
