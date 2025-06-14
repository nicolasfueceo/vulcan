from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    required_args: List[str]
    optional_args: Optional[List[str]] = None


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        required_args: List[str],
        optional_args: Optional[List[str]] = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._tools[name] = Tool(
                name=name,
                description=description,
                func=wrapper,
                required_args=required_args,
                optional_args=optional_args or [],
            )
            return wrapper

        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_description(self, name: str) -> Optional[str]:
        tool = self.get_tool(name)
        return tool.description if tool else None

    def execute_tool(self, name: str, **kwargs) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")

        # Validate required arguments
        missing_args = [arg for arg in tool.required_args if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments for {name}: {missing_args}")

        # Remove any arguments that aren't required or optional
        valid_args = set(tool.required_args + (tool.optional_args or []))
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

        return tool.func(**filtered_kwargs)


# Create global registry instance
registry = ToolRegistry()
