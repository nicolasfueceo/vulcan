CREATE VIEW user_stats_daily AS
SELECT
    user_id,
    DATE_TRUNC('day', date_added) AS day,
    COUNT(*) AS n_ratings,
    AVG(rating) AS mean_rating,
    STDDEV_SAMP(rating) AS var_rating
FROM curated_reviews
GROUP BY user_id, DATE_TRUNC('day', date_added); 