-- First, identify users with less than 5 reviews
WITH low_activity_users AS (
    SELECT user_id
    FROM reviews
    GROUP BY user_id
    HAVING COUNT(*) < 5
)

-- Delete reviews from these users
DELETE FROM reviews
WHERE user_id IN (SELECT user_id FROM low_activity_users);

-- Note: We don't need to delete from the books table as it's independent
-- and we want to keep all books for future reviews