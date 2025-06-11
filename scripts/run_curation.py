import os

import duckdb

# Define paths
db_path = "data/goodreads_curated.duckdb"
curation_steps_dir = "scripts/curation_steps"

# Ensure the old DB is removed
if os.path.exists(db_path):
    os.remove(db_path)

# Get the list of SQL files to execute, sorted numerically
sql_files = sorted([f for f in os.listdir(curation_steps_dir) if f.endswith(".sql")])

try:
    with duckdb.connect(database=db_path) as con:
        for sql_file in sql_files:
            file_path = os.path.join(curation_steps_dir, sql_file)
            print(f"Executing {file_path}...")
            with open(file_path, "r") as f:
                sql_script = f.read()

            # For the indexing script, we need to split by semicolon
            if sql_file == "10_indexing.sql":
                index_statements = sql_script.split(";")
                for stmt in index_statements:
                    if stmt.strip():
                        con.execute(stmt)
            else:
                con.execute(sql_script)
            print(f"✅ Successfully executed {sql_file}")

    print(f"\n🎉 Successfully created and populated {db_path}")

except Exception as e:
    print(f"\n❌ Curation script failed on file: {sql_file}")
    print(e)
    raise
