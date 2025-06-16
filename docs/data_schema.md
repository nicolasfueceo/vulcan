# Data Schema

VULCAN uses a DuckDB database with the following tables:

## curated_books
| Column            | Type    | Description                                  |
|-------------------|---------|----------------------------------------------|
| book_id           | INTEGER | Unique identifier for each book              |
| title             | TEXT    | Title of the book                           |
| author            | TEXT    | Author name(s)                              |
| genre             | TEXT    | Genre/category                              |
| publication_year  | INTEGER | Year the book was published                 |
| avg_rating        | FLOAT   | Average rating                              |
| num_ratings       | INTEGER | Number of ratings                           |
| num_reviews       | INTEGER | Number of reviews                           |

## curated_reviews
| Column         | Type    | Description                                 |
|----------------|---------|---------------------------------------------|
| review_id      | INTEGER | Unique identifier for each review           |
| book_id        | INTEGER | Foreign key to curated_books.book_id        |
| user_id        | INTEGER | Foreign key to users.user_id                |
| rating         | FLOAT   | User's rating for the book                  |
| review_text    | TEXT    | The review content                          |
| review_date    | DATE    | Date of the review                          |

## users
| Column     | Type    | Description                  |
|------------|---------|------------------------------|
| user_id    | INTEGER | Unique user identifier       |
| name       | TEXT    | User's display name          |
