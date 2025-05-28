"""Feature validator for checking and validating feature definitions."""

import ast
import re
from typing import List, Optional, Set, Tuple

from .feature_representation import FeatureDefinition


class FeatureValidator:
    """Validates feature definitions for syntax and safety."""

    def __init__(
        self,
        allowed_modules: Optional[Set[str]] = None,
        allowed_builtins: Optional[Set[str]] = None,
    ):
        """
        Initialize the feature validator.

        Args:
            allowed_modules: Set of module names that are allowed to be imported.
                            If None, a default safe set will be used.
            allowed_builtins: Set of built-in functions that are allowed.
                             If None, a default safe set will be used.
        """
        # Default allowed modules (can be overridden)
        self.allowed_modules = allowed_modules or {
            "numpy",
            "pandas",
            "math",
            "statistics",
            "datetime",
            "collections",
            "functools",
            "re",
            "string",
            "itertools",
        }

        # Default allowed builtins (can be overridden)
        self.allowed_builtins = allowed_builtins or {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "chr",
            "complex",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "hasattr",
            "hash",
            "hex",
            "id",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "oct",
            "ord",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "slice",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "zip",
        }

        # Banned operations that might be dangerous
        self.banned_operations = {
            "eval",
            "exec",
            "compile",
            "globals",
            "locals",
            "getattr",
            "setattr",
            "delattr",
            "open",
            "input",
            "__import__",
            "os.system",
            "subprocess",
            "sys.modules",
            "importlib",
            "__builtins__",
            "builtins",
            "file",
            "pickle",
        }

    def validate_feature(self, feature: FeatureDefinition) -> Tuple[bool, List[str]]:
        """
        Validate a feature definition.

        Args:
            feature: The feature definition to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Basic structural validation
        if not feature.name or not isinstance(feature.name, str):
            errors.append("Feature name must be a non-empty string")

        if not feature.description or not isinstance(feature.description, str):
            errors.append("Feature description must be a non-empty string")

        if not feature.code or not isinstance(feature.code, str):
            errors.append("Feature code must be a non-empty string")

        if not feature.required_input_columns or not isinstance(
            feature.required_input_columns, list
        ):
            errors.append("Required input columns must be a non-empty list")

        if not feature.output_column_name or not isinstance(
            feature.output_column_name, str
        ):
            errors.append("Output column name must be a non-empty string")

        # If basic validation fails, don't continue with code validation
        if errors:
            return False, errors

        # Validate the code
        syntax_valid, syntax_errors = self._validate_syntax(feature.code)
        if not syntax_valid:
            errors.extend(syntax_errors)

        # Check for unsafe operations
        safety_valid, safety_errors = self._validate_safety(feature.code)
        if not safety_valid:
            errors.extend(safety_errors)

        # Check that the function exists and has correct parameters
        func_valid, func_errors = self._validate_function(feature.code)
        if not func_valid:
            errors.extend(func_errors)

        return len(errors) == 0, errors

    def _validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate the syntax of the code.

        Args:
            code: The Python code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")

        return len(errors) == 0, errors

    def _validate_safety(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate the code for safety concerns.

        Args:
            code: The Python code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Parse the code
            tree = ast.parse(code)

            # Check for banned operations and imports
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name not in self.allowed_modules:
                            errors.append(
                                f"Import of module '{name.name}' is not allowed"
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module not in self.allowed_modules:
                        errors.append(
                            f"Import from module '{node.module}' is not allowed"
                        )

                # Check for calls to banned functions
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.banned_operations:
                            errors.append(
                                f"Use of banned function '{func_name}' is not allowed"
                            )

                    elif isinstance(node.func, ast.Attribute):
                        # Check for calls like os.system()
                        if isinstance(node.func.value, ast.Name):
                            attr_source = node.func.value.id
                            attr_name = node.func.attr
                            full_name = f"{attr_source}.{attr_name}"

                            if full_name in self.banned_operations:
                                errors.append(
                                    f"Use of banned operation '{full_name}' is not allowed"
                                )

                    # Also check for calls to __builtins__ or builtins
                    if isinstance(node.func, ast.Attribute) and isinstance(
                        node.func.value, ast.Name
                    ):
                        if node.func.value.id in ["__builtins__", "builtins"]:
                            errors.append("Direct access to builtins is not allowed")

            # Search for dangerous patterns in the raw code
            for pattern in self.banned_operations:
                if re.search(r"\b" + re.escape(pattern) + r"\b", code):
                    errors.append(f"Potentially unsafe operation '{pattern}' detected")

        except SyntaxError:
            # Syntax errors are handled in _validate_syntax
            pass

        return len(errors) == 0, errors

    def _validate_function(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate that the code defines a proper feature function.

        Args:
            code: The Python code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Parse the code
            tree = ast.parse(code)

            # Look for function definitions
            func_defs = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]

            if not func_defs:
                errors.append("No function definition found in the code")
                return False, errors

            # Check the main function (assume it's the first one defined)
            main_func = func_defs[0]

            # Check function parameters
            if len(main_func.args.args) < 1:
                errors.append(
                    f"Function '{main_func.name}' must have at least one parameter (current_review_data)"
                )

            # Check function body for common issues
            has_return = any(
                isinstance(node, ast.Return) for node in ast.walk(main_func)
            )
            if not has_return:
                errors.append(
                    f"Function '{main_func.name}' must have at least one return statement"
                )

        except SyntaxError:
            # Syntax errors are handled in _validate_syntax
            pass

        return len(errors) == 0, errors
