import duckdb
import pandas as pd

# Configure pandas for better display
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 150)

DB_PATH = "data/goodreads_curated.duckdb"

print(f"--- Verifying Curated Dates in {DB_PATH} ---")

try:
    with duckdb.connect(database=DB_PATH, read_only=True) as con:
        print(
            "\n[+] Checking 'publication_date' in curated_books (should be DATE type)"
        )
        con.sql("DESCRIBE curated_books;").show()

        print(
            "\n[+] Sampling 'publication_date' from curated_books (should not be all NULL)"
        )
        books_dates_df = con.sql("""
            SELECT publication_date
            FROM curated_books
            WHERE publication_date IS NOT NULL
            LIMIT 10;
        """).df()
        print(books_dates_df)
        assert not books_dates_df.empty, "FAIL: All publication_date values are NULL!"
        print("✅ PASS: Found non-NULL publication dates.")

        print(
            "\n[+] Checking 'date_added' in curated_reviews (should be TIMESTAMP type)"
        )
        con.sql("DESCRIBE curated_reviews;").show()

        print(
            "\n[+] Sampling 'date_added' from curated_reviews (should not be all NULL)"
        )
        reviews_dates_df = con.sql("""
            SELECT date_added
            FROM curated_reviews
            WHERE date_added IS NOT NULL
            LIMIT 10;
        """).df()
        print(reviews_dates_df)
        assert not reviews_dates_df.empty, "FAIL: All date_added values are NULL!"
        print("✅ PASS: Found non-NULL date_added timestamps.")


except Exception as e:
    print(f"\n❌ Verification failed: {e}")

print("\n--- Verification Complete ---")
