from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Feature:
    name: str
    description: str
    type: str  # 'code' or 'row'
    code: Optional[str] = None  # generated code (for code-based features)
    prompt: Optional[str] = None  # prompt used (for row-based features)
    reasoning_steps: List[str] = field(default_factory=list)  # chain-of-thought steps
    values: Optional[List[Any]] = None  # computed row-level values (for row-based)
    # ... (other fields as needed) ...
