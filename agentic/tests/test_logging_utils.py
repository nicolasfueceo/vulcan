import pytest
from agentic.utils.logger import AgenticLogger

def test_logger_info_and_error(capsys):
    logger = AgenticLogger()
    logger.info("info message")
    logger.error("error message")
    out, err = capsys.readouterr()
    assert "INFO" in out and "info message" in out
    assert "ERROR" in out and "error message" in out

def test_logger_level_filtering(capsys):
    logger = AgenticLogger(level="ERROR")
    logger.info("should not be printed")
    logger.error("should be printed")
    out, _ = capsys.readouterr()
    assert "should not be printed" not in out
    assert "should be printed" in out
