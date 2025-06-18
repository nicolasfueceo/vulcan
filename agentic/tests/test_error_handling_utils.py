import pytest
from agentic.utils.error_handling import catch_and_log

class DummyLogger:
    def __init__(self):
        self.errors = []
    def error(self, msg):
        self.errors.append(msg)

def test_catch_and_log_success():
    logger = DummyLogger()
    @catch_and_log(logger)
    def f(x):
        return x + 1
    assert f(1) == 2
    assert logger.errors == []

def test_catch_and_log_exception():
    logger = DummyLogger()
    @catch_and_log(logger)
    def f(x):
        raise ValueError("fail!")
    assert f(1) is None
    assert any("fail!" in msg for msg in logger.errors)
