"""
Safe code execution sandbox for feature engineering.
"""

import ast
import logging
import signal
import traceback
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Exception raised when code execution times out."""

    pass


class CodeSandbox:
    """
    Safe execution environment for feature engineering code.

    Provides restricted execution with timeout protection and
    validation of allowed imports and functions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the code sandbox.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.timeout = config.get("timeout", 30)
        self.max_memory = config.get("max_memory", 104857600)  # 100MB

        # Allowed imports and functions
        self.allowed_imports = {
            "pandas",
            "numpy",
            "math",
            "statistics",
            "datetime",
            "re",
            "collections",
            "itertools",
            "functools",
            "operator",
        }

        self.allowed_functions = {
            "len",
            "sum",
            "min",
            "max",
            "abs",
            "round",
            "sorted",
            "reversed",
            "enumerate",
            "zip",
            "range",
            "list",
            "dict",
            "set",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
        }

        self.logger = logger

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code for safety before execution.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with validation results
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Check for dangerous operations
            validator = CodeValidator(self.allowed_imports, self.allowed_functions)
            validator.visit(tree)

            if validator.errors:
                return {"valid": False, "errors": validator.errors}

            return {"valid": True, "errors": []}

        except SyntaxError as e:
            return {"valid": False, "errors": [f"Syntax error: {str(e)}"]}
        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}

    def execute_code(
        self, code: str, context: Dict[str, Any], function_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code safely in a restricted environment.

        Args:
            code: Python code to execute
            context: Context variables to make available
            function_name: Name of function to call after execution

        Returns:
            Dictionary with execution results
        """
        # Validate code first
        validation = self.validate_code(code)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Code validation failed: {validation['errors']}",
                "result": None,
            }

        try:
            # Set up timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Code execution timed out after {self.timeout} seconds"
                )

            # Set timeout (only works on Unix-like systems)
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)

            # Create restricted execution environment
            restricted_globals = self._create_restricted_globals()
            restricted_locals = context.copy()

            try:
                # Execute the code
                exec(code, restricted_globals, restricted_locals)

                # Call the specified function if provided
                if function_name:
                    if function_name in restricted_locals:
                        func = restricted_locals[function_name]
                        if callable(func):
                            # Extract function arguments from context
                            result = self._call_function_safely(func, context)
                        else:
                            result = func
                    else:
                        return {
                            "success": False,
                            "error": f"Function '{function_name}' not found in executed code",
                            "result": None,
                        }
                else:
                    # Return the entire local namespace
                    result = restricted_locals

                return {"success": True, "error": None, "result": result}

            finally:
                # Clear timeout
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

        except TimeoutError as e:
            return {"success": False, "error": str(e), "result": None}
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}\n{traceback.format_exc()}",
                "result": None,
            }

    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for code execution."""
        import collections
        import datetime
        import functools
        import itertools
        import math
        import operator
        import re
        import statistics

        import numpy as np
        import pandas as pd

        restricted_globals = {
            "__builtins__": {
                name: getattr(__builtins__, name)
                for name in self.allowed_functions
                if hasattr(__builtins__, name)
            },
            "pd": pd,
            "np": np,
            "math": math,
            "statistics": statistics,
            "datetime": datetime,
            "re": re,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "operator": operator,
        }

        return restricted_globals

    def _call_function_safely(self, func: callable, context: Dict[str, Any]) -> Any:
        """
        Call a function safely with appropriate arguments.

        Args:
            func: Function to call
            context: Available context variables

        Returns:
            Function result
        """
        import inspect

        try:
            # Get function signature
            sig = inspect.signature(func)

            # Prepare arguments based on function signature
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in context:
                    kwargs[param_name] = context[param_name]

            # Call function with available arguments
            return func(**kwargs)

        except Exception as e:
            raise Exception(f"Error calling function: {str(e)}")


class CodeValidator(ast.NodeVisitor):
    """AST visitor to validate code for safety."""

    def __init__(self, allowed_imports: Set[str], allowed_functions: Set[str]):
        self.allowed_imports = allowed_imports
        self.allowed_functions = allowed_functions
        self.errors: List[str] = []

    def visit_Import(self, node):
        """Check import statements."""
        for alias in node.names:
            if alias.name not in self.allowed_imports:
                self.errors.append(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from-import statements."""
        if node.module and node.module not in self.allowed_imports:
            self.errors.append(f"Import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check function calls for dangerous operations."""
        # Check for dangerous built-in functions
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            dangerous_functions = {
                "eval",
                "exec",
                "compile",
                "open",
                "input",
                "raw_input",
                "__import__",
                "getattr",
                "setattr",
                "delattr",
                "hasattr",
                "globals",
                "locals",
                "vars",
                "dir",
                "help",
                "reload",
            }

            if func_name in dangerous_functions:
                self.errors.append(f"Dangerous function call: {func_name}")

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Check attribute access for dangerous operations."""
        # Check for dangerous attribute access
        if isinstance(node.attr, str):
            dangerous_attrs = {
                "__class__",
                "__bases__",
                "__subclasses__",
                "__mro__",
                "__globals__",
                "__code__",
                "__func__",
                "__self__",
            }

            if node.attr in dangerous_attrs:
                self.errors.append(f"Dangerous attribute access: {node.attr}")

        self.generic_visit(node)
