"""
Tool schema for function-calling-based feature realization in the Strategy Team.
Defines the tool for the EngineerAgent to submit Python feature functions.
"""

write_feature_function_tool_schema = {
    "type": "function",
    "function": {
        "name": "write_feature_function",
        "description": "Submit the Python code for a feature function. The code must define a single function with the exact required name and signature, and a docstring explaining its logic and dependencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The exact name of the feature function (snake_case)."
                },
                "python_code": {
                    "type": "string",
                    "description": "The full Python code for the feature function, including the function definition and docstring."
                }
            },
            "required": ["function_name", "python_code"]
        }
    }
}
