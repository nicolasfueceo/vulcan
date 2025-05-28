"""Memory system for feature engineering reflections."""

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReflectionEntry:
    """A single reflection entry about a feature or feature engineering decision."""

    timestamp: str
    entry_type: str  # 'feature_proposal', 'feature_evaluation', 'strategy', etc.
    content: str  # The reflection text
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls, entry_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "ReflectionEntry":
        """Create a new reflection entry with current timestamp."""
        timestamp = datetime.datetime.now().isoformat()
        return cls(
            timestamp=timestamp,
            entry_type=entry_type,
            content=content,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionEntry":
        """Create from dictionary representation."""
        return cls(
            timestamp=data["timestamp"],
            entry_type=data["entry_type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class ReflectionMemory:
    """Memory system for storing and retrieving reflections about feature engineering."""

    def __init__(self, memory_dir: str = "memory", max_entries: int = 100):
        """
        Initialize the reflection memory.

        Args:
            memory_dir: Directory to save memory files to
            max_entries: Maximum number of entries to keep in memory
        """
        self.memory_dir = memory_dir
        self.max_entries = max_entries
        self.entries: List[ReflectionEntry] = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)

    def add_entry(
        self, entry_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ReflectionEntry:
        """
        Add a new reflection entry to memory.

        Args:
            entry_type: Type of reflection ('feature_proposal', 'feature_evaluation', etc.)
            content: The reflection text
            metadata: Optional additional metadata

        Returns:
            The created reflection entry
        """
        entry = ReflectionEntry.create(entry_type, content, metadata)
        self.entries.append(entry)

        # Trim memory if it exceeds max size
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

        # Save to disk
        self._save_memory()

        return entry

    def get_entries_by_type(self, entry_type: str) -> List[ReflectionEntry]:
        """
        Get all entries of a specific type.

        Args:
            entry_type: Type of reflection to retrieve

        Returns:
            List of matching reflection entries
        """
        return [entry for entry in self.entries if entry.entry_type == entry_type]

    def get_entries_with_metadata(self, key: str, value: Any) -> List[ReflectionEntry]:
        """
        Get all entries that have a specific metadata key-value pair.

        Args:
            key: Metadata key to match
            value: Metadata value to match

        Returns:
            List of matching reflection entries
        """
        return [
            entry
            for entry in self.entries
            if key in entry.metadata and entry.metadata[key] == value
        ]

    def get_recent_entries(self, n: int = 5) -> List[ReflectionEntry]:
        """
        Get the N most recent entries.

        Args:
            n: Number of entries to retrieve

        Returns:
            List of the N most recent entries
        """
        return self.entries[-n:]

    def clear(self) -> None:
        """Clear all entries from memory."""
        self.entries = []
        self._save_memory()

    def save_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save the current memory to a specific file.

        Args:
            filename: Optional filename to save to (default: uses session ID)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"memory_{self.session_id}.json"

        filepath = os.path.join(self.memory_dir, filename)

        with open(filepath, "w") as f:
            data = {
                "session_id": self.session_id,
                "entries": [entry.to_dict() for entry in self.entries],
                "timestamp": datetime.datetime.now().isoformat(),
            }
            json.dump(data, f, indent=2)

        return filepath

    def load_from_file(self, filepath: str) -> bool:
        """
        Load memory from a file.

        Args:
            filepath: Path to the memory file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.session_id = data.get("session_id", self.session_id)
            self.entries = [
                ReflectionEntry.from_dict(entry_data)
                for entry_data in data.get("entries", [])
            ]

            return True

        except Exception as e:
            print(f"Error loading memory from {filepath}: {e}")
            return False

    def _save_memory(self) -> None:
        """Save the current memory state to the default session file."""
        self.save_to_file(f"memory_{self.session_id}.json")

    def get_formatted_history(
        self, n: Optional[int] = None, entry_type: Optional[str] = None
    ) -> str:
        """
        Get a formatted string of memory entries for inclusion in prompts.

        Args:
            n: Optional number of entries to include (default: all)
            entry_type: Optional type of entries to include

        Returns:
            Formatted string of memory entries
        """
        entries = self.entries

        # Filter by type if specified
        if entry_type:
            entries = [entry for entry in entries if entry.entry_type == entry_type]

        # Limit to N entries if specified
        if n is not None:
            entries = entries[-n:]

        # Format the entries
        formatted = []
        for entry in entries:
            timestamp = datetime.datetime.fromisoformat(entry.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            formatted.append(
                f"[{timestamp}] {entry.entry_type.upper()}: {entry.content}"
            )

        return "\n\n".join(formatted)
