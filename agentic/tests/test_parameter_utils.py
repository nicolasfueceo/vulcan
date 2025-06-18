import pytest
from agentic.utils.parameter import extract_params

def test_extract_params_defaults():
    params = {"a": 1, "b": 2}
    schema = {"a": 0, "b": 0, "c": 99}
    result = extract_params(params, schema)
    assert result == {"a": 1, "b": 2, "c": 99}

def test_extract_params_types():
    params = {"a": "5", "b": 2.2}
    schema = {"a": 0, "b": 0.0}
    result = extract_params(params, schema, enforce_types=True)
    assert result == {"a": 5, "b": 2.2}

    # If type cannot be cast, fallback to default
    params = {"a": "bad"}
    schema = {"a": 42}
    result = extract_params(params, schema, enforce_types=True)
    assert result == {"a": 42}
