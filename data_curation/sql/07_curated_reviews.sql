CREATE TABLE curated_reviews AS
SELECT
    review_id,
    user_id,
    book_id,
    rating::TINYINT AS rating,
    review_text,
    try_cast(date_added AS TIMESTAMP) AS date_added,
    try_cast(date_updated AS TIMESTAMP) AS date_updated,
    try_cast(read_at AS TIMESTAMP) AS read_at,
    try_cast(started_at AS TIMESTAMP) AS started_at,
    try_cast(n_votes AS INTEGER) AS n_votes,
    try_cast(n_comments AS INTEGER) AS n_comments
FROM raw.reviews; 