# Data Curation Pipeline

This directory contains the scripts to process the raw data from `data/goodreads_raw.duckdb` and create the curated database `data/goodreads_curated.duckdb`.

## Main Pipeline

The main pipeline is orchestrated by `run.py`. It executes a series of SQL scripts located in the `sql/` directory in numerical order.

To run the pipeline:

```bash
python data_curation/run.py
```

This will create a fresh version of `data/goodreads_curated.duckdb`.

## Additional Scripts

The `steps/` directory contains additional Python scripts for analysis, verification, and cleaning. These are intended for manual execution and are not part of the main automated pipeline.

- **`analyze_db.py`**: Performs various analyses on the database tables.
- **`clean_data.py`**: Contains functions for data cleaning tasks.
- **`drop_useless_tables.py`**: Removes specific tables and columns that have been identified as redundant or useless. This is idempotent and can be run multiple times.
- **`get_curated_schema.py`**: Prints the schema of the curated database.
- **`inspect_raw_dates.py`**: A script to inspect date formats in the raw data.
- **`verify_curated_dates.py`**: Verifies date formats in the curated data. 