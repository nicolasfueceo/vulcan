"""
SQL view management utilities for the VULCAN pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import duckdb
from loguru import logger


class ViewManager:
    """Manages SQL views and their documentation."""

    def __init__(self, db_path: str, views_dir: Path):
        """
        Initialize the view manager.

        Args:
            db_path: Path to the DuckDB database
            views_dir: Directory to store view metadata
        """
        self.db_path = db_path
        self.views_dir = views_dir
        self.views_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = views_dir / "view_metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load view metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save view metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _get_next_version(self, base_name: str) -> str:
        """
        Get the next available version number for a view name.

        Args:
            base_name: Base name of the view

        Returns:
            Next available version number
        """
        version = 1
        while f"{base_name}_v{version}" in self.metadata:
            version += 1
        return version

    def create_view(
        self,
        view_name: str,
        sql: str,
        rationale: str,
        force: bool = False,
    ) -> str:
        """
        Create a SQL view with documentation.

        Args:
            view_name: Name of the view to create
            sql: SQL query to create the view
            rationale: Explanation of what the view is for
            force: If True, overwrite existing view

        Returns:
            Actual view name used (may have version suffix)
        """
        # Check if view exists
        with duckdb.connect(self.db_path) as conn:
            existing_views = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            existing_views = [v[0] for v in existing_views]

        # Handle view name collision
        actual_name = view_name
        if view_name in existing_views:
            if force:
                logger.warning(f"Overwriting existing view: {view_name}")
            else:
                version = self._get_next_version(view_name)
                actual_name = f"{view_name}_v{version}"
                logger.info(f"View {view_name} exists, using {actual_name} instead")

        # Create the view
        with duckdb.connect(self.db_path) as conn:
            conn.execute(f"CREATE OR REPLACE VIEW {actual_name} AS {sql}")

        # Save metadata
        self.metadata[actual_name] = {
            "original_name": view_name,
            "sql": sql,
            "rationale": rationale,
            "created_at": str(Path(self.db_path).stat().st_mtime),
        }
        self._save_metadata()

        logger.info(f"Created view {actual_name}")
        return actual_name

    def get_view_metadata(self, view_name: str) -> Optional[Dict]:
        """
        Get metadata for a view.

        Args:
            view_name: Name of the view

        Returns:
            View metadata if found, None otherwise
        """
        return self.metadata.get(view_name)

    def list_views(self) -> Dict[str, Dict]:
        """
        List all views and their metadata.

        Returns:
            Dictionary mapping view names to their metadata
        """
        return self.metadata.copy()

    def drop_view(self, view_name: str) -> None:
        """
        Drop a view and its metadata.

        Args:
            view_name: Name of the view to drop
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")

        if view_name in self.metadata:
            del self.metadata[view_name]
            self._save_metadata()
            logger.info(f"Dropped view {view_name}")


# Global view manager instance
view_manager = None


def init_view_manager(db_path: str, views_dir: Path) -> None:
    """
    Initialize the global view manager.

    Args:
        db_path: Path to the DuckDB database
        views_dir: Directory to store view metadata
    """
    global view_manager
    view_manager = ViewManager(db_path, views_dir)


def create_analysis_view(
    view_name: str,
    sql: str,
    rationale: str,
    force: bool = False,
) -> str:
    """
    Create a SQL view with documentation.

    Args:
        view_name: Name of the view to create
        sql: SQL query to create the view
        rationale: Explanation of what the view is for
        force: If True, overwrite existing view

    Returns:
        Actual view name used (may have version suffix)
    """
    if view_manager is None:
        raise RuntimeError(
            "View manager not initialized. Call init_view_manager first."
        )
    return view_manager.create_view(view_name, sql, rationale, force)


def get_view_metadata(view_name: str) -> Optional[Dict]:
    """
    Get metadata for a view.

    Args:
        view_name: Name of the view

    Returns:
        View metadata if found, None otherwise
    """
    if view_manager is None:
        raise RuntimeError(
            "View manager not initialized. Call init_view_manager first."
        )
    return view_manager.get_view_metadata(view_name)


def list_views() -> Dict[str, Dict]:
    """
    List all views and their metadata.

    Returns:
        Dictionary mapping view names to their metadata
    """
    if view_manager is None:
        raise RuntimeError(
            "View manager not initialized. Call init_view_manager first."
        )
    return view_manager.list_views()


def drop_view(view_name: str) -> None:
    """
    Drop a view and its metadata.

    Args:
        view_name: Name of the view to drop
    """
    if view_manager is None:
        raise RuntimeError(
            "View manager not initialized. Call init_view_manager first."
        )
    view_manager.drop_view(view_name)
