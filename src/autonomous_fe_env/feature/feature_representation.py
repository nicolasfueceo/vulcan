"""Feature representation classes for the autonomous feature engineering system."""

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataRequirement:
    """Specifies data needed beyond the current instance for a feature."""

    type: str  # e.g., 'horizontal', 'vertical'
    entity_type: str  # e.g., 'user', 'book'
    columns: List[str]  # Columns needed from the related data
    # ID field in the *current* instance data used to find related data
    # For horizontal: typically the entity's own ID field (e.g., 'user_id')
    # For vertical: typically the context ID field (e.g., 'book_id')
    lookup_id_field: str
    # Optional: Name of the primary ID column in the *related* data table (e.g., 'review_id')
    # Used to potentially exclude the current instance from related data.
    instance_id_field: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "entity_type": self.entity_type,
            "columns": self.columns,
            "lookup_id_field": self.lookup_id_field,
            "instance_id_field": self.instance_id_field,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRequirement":
        """Create from dictionary representation."""
        return cls(
            type=data["type"],
            entity_type=data["entity_type"],
            columns=data["columns"],
            lookup_id_field=data["lookup_id_field"],
            instance_id_field=data.get("instance_id_field"),
        )


@dataclass
class FeatureDefinition:
    """Represents a feature proposed by an agent.

    Attributes:
        name: A unique identifier/name for the feature.
        description: A natural language description of what the feature does.
        code: The Python code (as a string) that implements the feature calculation.
              This code should define a function that accepts specific arguments.
        data_requirements: Optional specification of additional data needed.
        required_input_columns: List of columns from the primary data instance needed by the code.
        output_column_name: The name of the column this feature will produce.
        error_handling_strategy: How errors during execution should be handled
                                   (e.g., 'return_default', 'raise_exception', 'skip_instance').
        default_value: The value to return if execution fails and strategy is 'return_default'.
        metadata: Optional additional metadata about the feature.
    """

    name: str
    description: str
    code: str
    required_input_columns: List[str]
    output_column_name: str
    data_requirements: Optional[DataRequirement] = None
    error_handling_strategy: str = "return_default"
    default_value: Any = 0.0  # Sensible default might be float for regression tasks
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Basic validation
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Feature name must be a non-empty string.")
        if not self.code or not isinstance(self.code, str):
            raise ValueError("Feature code must be a non-empty string.")
        if not self.required_input_columns or not isinstance(
            self.required_input_columns, list
        ):
            raise ValueError("required_input_columns must be a non-empty list.")
        if not self.output_column_name or not isinstance(self.output_column_name, str):
            raise ValueError("output_column_name must be a non-empty string.")
        if self.error_handling_strategy not in [
            "return_default",
            "raise_exception",
            "skip_instance",
        ]:
            raise ValueError(
                f"Invalid error_handling_strategy: {self.error_handling_strategy}"
            )
        if self.data_requirements and not isinstance(
            self.data_requirements, DataRequirement
        ):
            raise ValueError(
                "data_requirements must be an instance of DataRequirement or None."
            )

    def get_function_name(self) -> str:
        """Extracts the function name from the code string.

        Uses AST parsing for reliable extraction.

        Returns:
            The name of the feature function defined in the code.
        """
        try:
            # Parse code with AST
            tree = ast.parse(self.code)

            # Find function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name

            # Fallback to regex if AST parsing didn't find a function
            match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", self.code)
            if match:
                return match.group(1)

            # If still no match, use a default name
            return "feature_function"

        except SyntaxError:
            # If code has syntax errors, fall back to regex
            match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", self.code)
            if match:
                return match.group(1)
            return "feature_function"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "required_input_columns": self.required_input_columns,
            "output_column_name": self.output_column_name,
            "error_handling_strategy": self.error_handling_strategy,
            "default_value": self.default_value,
            "metadata": self.metadata,
        }

        if self.data_requirements:
            result["data_requirements"] = self.data_requirements.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDefinition":
        """Create from dictionary representation."""
        # Handle data requirements if present
        data_req = None
        if "data_requirements" in data and data["data_requirements"]:
            data_req = DataRequirement.from_dict(data["data_requirements"])

        return cls(
            name=data["name"],
            description=data["description"],
            code=data["code"],
            required_input_columns=data["required_input_columns"],
            output_column_name=data["output_column_name"],
            data_requirements=data_req,
            error_handling_strategy=data.get(
                "error_handling_strategy", "return_default"
            ),
            default_value=data.get("default_value", 0.0),
            metadata=data.get("metadata", {}),
        )
