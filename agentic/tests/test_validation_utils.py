import pytest
from agentic.utils.validation import require_keys

def test_require_keys_pass():
    @require_keys(["foo", "bar"])
    def f(d):
        return True
    assert f({"foo": 1, "bar": 2}) is True

def test_require_keys_fail():
    @require_keys(["foo", "bar"])
    def f(d):
        return True
    with pytest.raises(KeyError):
        f({"foo": 1})
