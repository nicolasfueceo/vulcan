import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score

def run_popularity_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Recommend the most popular items (books) in the training set to all users in the test set.
    Returns only the top-N popular books and the number of recommendations made.
    """
    # Compute most popular books by count of ratings in train set
    pop_books = (
        train_df.groupby('book_id')['rating'].count()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    user_ids = test_df['user_id'].unique()
    recommendations = {user_id: pop_books for user_id in user_ids}
    # For RMSE, predict the mean rating of the popular books for each user-item pair in test set
    mean_pop_rating = train_df[train_df['book_id'].isin(pop_books)]['rating'].mean()
    if np.isnan(mean_pop_rating):
        mean_pop_rating = train_df['rating'].mean()

    y_true = test_df['rating'].values
    y_pred = np.full_like(y_true, fill_value=mean_pop_rating, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # NDCG@10 calculation
    all_items = train_df['book_id'].unique()
    true_relevance = np.isin(np.array(all_items)[:, None], test_df['book_id'].values).astype(int)
    # Popularity score: 1 for top-N, 0 for others
    scores = np.isin(np.array(all_items)[:, None], np.array(pop_books)).astype(int)
    if len(all_items) > 1:
        ndcg_at_5 = ndcg_score(true_relevance, scores, k=5)
    else:
        ndcg_at_5 = float('nan')
    result = {
        'top_n_books': pop_books,
        'num_users': len(user_ids),
        'num_recommendations': len(user_ids) * len(pop_books),
        'rmse': rmse,
        'mae': mae,
        'ndcg_at_5': ndcg_at_5
    }
    return result
