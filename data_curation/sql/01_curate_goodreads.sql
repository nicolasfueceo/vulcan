-- Attach the raw database and install necessary extensions
INSTALL httpfs;
LOAD httpfs;
ATTACH 'data/goodreads_raw.duckdb' AS raw (READ_ONLY);

----------------------------------------------------
--  Curated Tables
----------------------------------------------------

-- 1. curated_books
CREATE TABLE curated_books AS
SELECT
    book_id,
    work_id::BIGINT AS work_id,
    title,
    title_without_series,
    description,
    language_code,
    country_code,
    format,
    is_ebook::BOOLEAN AS is_ebook,
    try_cast(num_pages AS SMALLINT) AS num_pages,
    -- Safely construct and cast date, handling invalid components
    try_strptime(
        concat(
            try_cast(publication_year as VARCHAR),
            '-',
            lpad(coalesce(try_cast(publication_month as VARCHAR), '1'), 2, '0'),
            '-',
            lpad(coalesce(try_cast(publication_day as VARCHAR), '1'), 2, '0')
        ),
        '%Y-%m-%d'
    ) AS publication_date,
    try_cast(average_rating AS FLOAT) AS avg_rating,
    try_cast(ratings_count AS INTEGER) AS ratings_count,
    try_cast(text_reviews_count AS INTEGER) AS text_reviews_count,
    publisher AS publisher_name
FROM raw.books;

-- 2. curated_reviews
CREATE TABLE curated_reviews AS
SELECT
    review_id,
    user_id,
    book_id,
    rating::TINYINT AS rating,
    review_text,
    -- Use try_strptime for robust date parsing of the specific format
    try_strptime(date_added, '%a %b %d %H:%M:%S %z %Y') AS date_added,
    try_strptime(date_updated, '%a %b %d %H:%M:%S %z %Y') AS date_updated,
    try_strptime(read_at, '%a %b %d %H:%M:%S %z %Y') AS read_at,
    try_strptime(started_at, '%a %b %d %H:%M:%S %z %Y') AS started_at,
    try_cast(n_votes AS INTEGER) AS n_votes,
    try_cast(n_comments AS INTEGER) AS n_comments
FROM raw.reviews;

-- 3. users
CREATE TABLE users AS
SELECT user_id FROM raw.users;


----------------------------------------------------
--  Normalized Long Tables (from unnesting)
----------------------------------------------------

-- 4. book_series
CREATE TABLE book_series AS
SELECT
    book_id,
    series_elem AS series_name,
    row_number() OVER (PARTITION BY book_id) AS series_pos
FROM raw.books, UNNEST(series) AS t(series_elem);

-- 5. book_shelves
CREATE TABLE book_shelves AS
SELECT
    book_id,
    s.name AS shelf,
    try_cast(s.count AS INTEGER) AS cnt
FROM raw.books, UNNEST(popular_shelves) AS t(s)
WHERE s.name IS NOT NULL;

-- 6. book_authors
CREATE TABLE book_authors AS
SELECT
    book_id,
    try_cast(s.author_id AS BIGINT) AS author_id,
    s.role AS role
FROM raw.books, UNNEST(authors) AS t(s);

-- 7. book_similars
CREATE TABLE book_similars AS
SELECT
    book_id,
    similar_elem AS similar_book_id,
    row_number() OVER (PARTITION BY book_id) AS rank
FROM raw.books, UNNEST(similar_books) AS t(similar_elem)
WHERE similar_elem IS NOT NULL;

----------------------------------------------------
--  Views and Indexes
----------------------------------------------------

-- 8. user_stats_daily (VIEW)
CREATE VIEW user_stats_daily AS
SELECT
    user_id,
    DATE_TRUNC('day', date_added) AS day,
    COUNT(*) AS n_ratings,
    AVG(rating) AS mean_rating,
    STDDEV_SAMP(rating) AS var_rating
FROM curated_reviews
GROUP BY user_id, DATE_TRUNC('day', date_added);

-- 9. Indexing
CREATE INDEX idx_reviews_user ON curated_reviews(user_id);
CREATE INDEX idx_reviews_book ON curated_reviews(book_id);
CREATE INDEX idx_books_rating ON curated_books(avg_rating); 