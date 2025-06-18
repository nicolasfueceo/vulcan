import pytest
from agentic.utils.logger import AgenticLogger

def test_logger_info_and_error(capsys):
    logger = AgenticLogger()
    logger.info("hello info")
    logger.error("oops error")
    out, err = capsys.readouterr()
    assert "INFO" in out and "hello info" in out
    assert "ERROR" in out and "oops error" in out

def test_logger_levels(capsys):
    logger = AgenticLogger(level="ERROR")
    logger.info("should not appear")
    logger.error("should appear")
    out, _ = capsys.readouterr()
    assert "should not appear" not in out
    assert "should appear" in out
