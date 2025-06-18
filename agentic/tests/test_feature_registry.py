from agentic.utils.feature_registry import FeatureFunctionRecord, FeatureRegistry
import tempfile
import os

def test_feature_registry():
    code = """
def feature_func(params, data_loader):
    return params['x'] + 1, 'user'
"""
    record = FeatureFunctionRecord(
        name="test_feature",
        code=code,
        scope="user",
        params={"x": 5}
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
        path = tf.name
    try:
        reg = FeatureRegistry(path)
        reg.register(record)
        reg2 = FeatureRegistry(path)
        rec2 = reg2.get(record.feature_id)
        assert rec2 is not None
        func, scope = rec2.load_function()
        out, returned_scope = func({"x": 2}, None)
        assert out == 3
        assert returned_scope == "user"
        print("FeatureRegistry test passed.")
    finally:
        os.remove(path)
