from typing import Dict, Any

def extract_params(params: Dict[str, Any], schema: Dict[str, Any], enforce_types: bool = False) -> Dict[str, Any]:
    out = {}
    for k, default in schema.items():
        v = params.get(k, default)
        if enforce_types and not isinstance(v, type(default)):
            try:
                v = type(default)(v)
            except Exception:
                v = default
        out[k] = v
    return out
