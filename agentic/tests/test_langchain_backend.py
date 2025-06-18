import pytest
from agentic.utils.langchain_backend import LangChainBackend

def test_langchain_backend_run():
    class DummyLangChain:
        def __call__(self, prompt, context):
            return f"langchain:{prompt}:{context}"
    backend = LangChainBackend(langchain_func=DummyLangChain())
    result = backend.run(prompt="foo", context={"bar": 2})
    assert result == "langchain:foo:{'bar': 2}"
