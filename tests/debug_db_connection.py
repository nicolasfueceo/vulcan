"""
A simple script to debug DuckDB connection issues with CVDataManager outside of pytest.
"""

import sys
from pathlib import Path

# Add src to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.cv_data_manager import CVDataManager
from loguru import logger

def main():
    """Initializes CVDataManager and tries to load folds to test the DB connection."""
    logger.info("Starting DB connection debug script...")
    data_dir = Path("data")
    cv_data = None

    try:
        logger.info(f"Attempting to initialize CVDataManager with data_dir: {data_dir.resolve()}")
        cv_data = CVDataManager(data_dir=data_dir)
        logger.info("CVDataManager initialized. Attempting to load CV folds...")
        cv_data.load_cv_folds()
        logger.success("Successfully loaded CV folds and connected to the database!")
        summary = cv_data.get_fold_summary()
        logger.info(f"Fold summary: {summary}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

    finally:
        logger.info("Attempting to close the global connection pool...")
        CVDataManager.close_global_connection_pool()
        logger.info("Global connection pool closed.")

if __name__ == "__main__":
    main()
