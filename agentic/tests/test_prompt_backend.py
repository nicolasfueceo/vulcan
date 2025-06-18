import pytest
from agentic.utils.prompt_backend import PromptBackend

def test_prompt_backend_run():
    backend = PromptBackend(template="Hello {name}!", engine=lambda prompt: prompt.upper())
    result = backend.run(prompt="", context={"name": "Alice"})
    assert result == "HELLO ALICE!"

def test_prompt_backend_missing_key():
    backend = PromptBackend(template="{foo}", engine=lambda prompt: prompt)
    with pytest.raises(KeyError):
        backend.run(prompt="", context={})
