import duckdb

DB_PATH = "data/goodreads_curated.duckdb"

con = duckdb.connect(DB_PATH)
views = con.execute("SELECT table_name FROM information_schema.views WHERE table_schema NOT IN ('information_schema', 'pg_catalog')").fetchall()
for v in views:
    vname = v[0]
    if vname != 'interactions':
        con.execute(f'DROP VIEW IF EXISTS {vname}')
        print(f'Dropped view: {vname}')
con.close()
