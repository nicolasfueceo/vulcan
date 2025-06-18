import pytest
from agentic.utils.autogen_backend import AutogenBackend

def test_autogen_backend_run():
    class DummyAutoGen:
        def __call__(self, prompt, context):
            return f"autogen:{prompt}:{context}"
    backend = AutogenBackend(autogen_func=DummyAutoGen())
    result = backend.run(prompt="foo", context={"bar": 1})
    assert result == "autogen:foo:{'bar': 1}"
