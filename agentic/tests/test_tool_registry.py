import pytest
from agentic.core.tool_registry import ToolRegistry

def test_tool_registration_and_call():
    registry = ToolRegistry()
    def foo(x):
        return x + 1
    registry.register("foo", foo)
    assert registry.call("foo", 3) == 4

    with pytest.raises(KeyError):
        registry.call("bar", 1)

    # Overwrite
    registry.register("foo", lambda x: x * 2)
    assert registry.call("foo", 3) == 6
