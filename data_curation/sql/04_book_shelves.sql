CREATE TABLE book_shelves AS
SELECT
    book_id::BIGINT AS book_id,
    s.name AS shelf,
    try_cast(s.count AS INTEGER) AS cnt
FROM raw.books,
UNNEST(popular_shelves) AS t(s)
WHERE s.name IS NOT NULL; 