import json
import hashlib
from typing import Dict, Any, Optional, Tuple, Callable
import types

class FeatureFunctionRecord:
    """
    Stores feature function code, metadata, and provides dynamic loading.
    """
    def __init__(self, name: str, code: str, scope: str, params: Dict[str, Any], feature_id: Optional[str] = None):
        self.name = name
        self.code = code
        self.scope = scope  # 'user' or 'item'
        self.params = params
        self.feature_id = feature_id or self.compute_hash()

    def compute_hash(self) -> str:
        m = hashlib.sha256()
        m.update(self.code.encode())
        m.update(json.dumps(self.params, sort_keys=True).encode())
        m.update(self.scope.encode())
        return m.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "code": self.code,
            "scope": self.scope,
            "params": self.params,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeatureFunctionRecord":
        return FeatureFunctionRecord(
            name=d["name"],
            code=d["code"],
            scope=d["scope"],
            params=d["params"],
            feature_id=d.get("feature_id"),
        )

    def load_function(self) -> Tuple[Callable, str]:
        """
        Dynamically loads the function object from the code string.
        Returns (function_object, scope)
        """
        local_ns = {}
        exec(self.code, {}, local_ns)
        # Convention: feature function must be named 'feature_func'
        func = local_ns["feature_func"]
        return func, self.scope

class FeatureRegistry:
    """
    Manages storage and loading of feature functions from JSON.
    """
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.records: Dict[str, FeatureFunctionRecord] = {}
        self.load()

    def load(self):
        import json
        try:
            with open(self.json_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
            for entry in data:
                rec = FeatureFunctionRecord.from_dict(entry)
                self.records[rec.feature_id] = rec
        except FileNotFoundError:
            self.records = {}

    def save(self):
        with open(self.json_path, "w") as f:
            json.dump([rec.to_dict() for rec in self.records.values()], f, indent=2)

    def register(self, record: FeatureFunctionRecord):
        self.records[record.feature_id] = record
        self.save()

    def get(self, feature_id: str) -> Optional[FeatureFunctionRecord]:
        return self.records.get(feature_id)

    def list_features(self):
        return list(self.records.values())
