from agentic.bo.reward_function_manager import RewardFunctionManager

def test_register_and_get():
    mgr = RewardFunctionManager()
    def dummy(x):
        return x * 2
    mgr.register("double", dummy)
    fn = mgr.get("double")
    assert fn(3) == 6
    assert "double" in mgr.list_functions()


def test_get_missing():
    mgr = RewardFunctionManager()
    try:
        mgr.get("not_exist")
    except ValueError as e:
        assert "No reward function registered" in str(e)
    else:
        assert False, "Expected ValueError"
