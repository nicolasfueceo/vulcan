import pytest
from agentic.utils.dict_utils import deep_merge, dict_diff

def test_deep_merge():
    a = {"x": 1, "y": {"z": 2}}
    b = {"y": {"z": 3, "w": 4}, "k": 5}
    merged = deep_merge(a, b)
    assert merged == {"x": 1, "y": {"z": 3, "w": 4}, "k": 5}


def test_dict_diff():
    a = {"x": 1, "y": 2}
    b = {"x": 1, "y": 3, "z": 4}
    diff = dict_diff(a, b)
    assert diff == {"y": (2, 3), "z": (None, 4)}
