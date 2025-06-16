import json
import threading
from datetime import datetime
from pathlib import Path

class RunLogger:
    """
    Logs every tool call (input/output) and every group chat message to a JSON file incrementally.
    Thread-safe for multi-agent, multi-process use.
    """
    def __init__(self, run_dir: Path, filename: str = "run_transcript.json"):
        self.log_path = Path(run_dir) / filename
        self.lock = threading.Lock()
        # Create the file if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def log_event(self, event_type: str, payload: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **payload
        }
        with self.lock:
            # Read, append, and write back (atomic for small files)
            with open(self.log_path, 'r+') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

    def log_message(self, sender, recipient, content, role=None, extra=None):
        self.log_event("message", {
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "role": role,
            "extra": extra or {}
        })

    def log_tool_call(self, tool_name, input_args, output, agent=None, extra=None):
        self.log_event("tool_call", {
            "tool_name": tool_name,
            "input_args": input_args,
            "output": output,
            "agent": agent,
            "extra": extra or {}
        })
