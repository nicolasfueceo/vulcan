import os
import sys
import duckdb

print("--- Starting data curation script ---")
sys.stdout.flush()

# Define paths
db_path = "data/goodreads_curated.duckdb"
curation_steps_dir = "data_curation/sql"
print(f"Database path: {db_path}")
print(f"SQL directory: {curation_steps_dir}")
sys.stdout.flush()

# Ensure the old DB is removed
if os.path.exists(db_path):
    print(f"Removing existing database: {db_path}")
    sys.stdout.flush()
    try:
        os.remove(db_path)
        print("Database removed successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"!!! FAILED to remove database: {e}")
        sys.stdout.flush()
        sys.exit(1) # Exit if we can't remove the old DB

# Get the list of SQL files to execute, sorted numerically
try:
    print("Listing SQL files...")
    sys.stdout.flush()
    sql_files = sorted([f for f in os.listdir(curation_steps_dir) if f.endswith(".sql")])
    print(f"Found SQL files: {sql_files}")
    sys.stdout.flush()
except Exception as e:
    print(f"Error listing SQL files: {e}")
    sys.stdout.flush()
    raise

sql_file = "N/A"
statement_num = 0
try:
    print(f"Connecting to database: {db_path}")
    sys.stdout.flush()
    with duckdb.connect(database=db_path) as con:
        print("Database connection successful.")
        sys.stdout.flush()
        for sql_file in sql_files:
            file_path = os.path.join(curation_steps_dir, sql_file)
            print(f"Executing {file_path}...")
            sys.stdout.flush()
            with open(file_path, "r", encoding="utf-8") as f:
                sql_script = f.read()

            statements = sql_script.split(';')
            
            statement_num = 0
            for statement in statements:
                statement_num += 1
                if statement.strip():
                    print(f"  Executing statement {statement_num} from {sql_file}...", end="")
                    sys.stdout.flush()
                    con.execute(statement)
                    print(" ✅")
                    sys.stdout.flush()

            print(f"✅ Successfully executed all statements in {sql_file}")
            sys.stdout.flush()

    print("\n🎉 Successfully created and populated {db_path}")
    sys.stdout.flush()

except Exception as e:
    print(f"\n❌ Curation script failed on file: {sql_file}, statement: {statement_num}")
    print(e)
    sys.stdout.flush()
    raise
