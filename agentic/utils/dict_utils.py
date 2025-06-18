from typing import Dict, Any

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def dict_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, tuple]:
    diff = {}
    keys = set(a) | set(b)
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            diff[k] = (va, vb)
    return diff
