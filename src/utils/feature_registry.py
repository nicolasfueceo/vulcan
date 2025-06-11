class FeatureRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, feature_data: dict):
        """Registers a feature function and its metadata."""
        if name in self._registry:
            print(f"Warning: Feature {name} is already registered. Overwriting.")
        self._registry[name] = feature_data

    def get(self, name: str) -> dict:
        """Retrieves a feature function and its metadata."""
        return self._registry.get(name)

    def get_all(self) -> dict:
        """Retrieves the entire registry."""
        return self._registry.copy()


# Global instance of the registry
feature_registry = FeatureRegistry()


def get_feature(name: str):
    """Public method to get a feature from the global registry."""
    feature_data = feature_registry.get(name)
    if feature_data:
        return feature_data.get("func")
    return None
