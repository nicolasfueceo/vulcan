-- Attach the raw database and install necessary extensions
INSTALL httpfs;
LOAD httpfs;
ATTACH 'data/goodreads_raw.duckdb' AS raw (READ_ONLY); 