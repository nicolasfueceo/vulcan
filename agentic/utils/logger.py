from typing import Literal
import sys

class AgenticLogger:
    LEVELS = {"INFO": 20, "ERROR": 40}
    def __init__(self, level: Literal["INFO", "ERROR"] = "INFO"):
        self.level = level

    def info(self, msg: str) -> None:
        if self.LEVELS[self.level] <= 20:
            print(f"[INFO] {msg}", file=sys.stdout)

    def error(self, msg: str) -> None:
        if self.LEVELS[self.level] <= 40:
            print(f"[ERROR] {msg}", file=sys.stdout)
