import pandas as pd
from src.evaluation.ranking_metrics import evaluate_ranking_metrics

def run_popularity_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 10, k_list=[5, 10, 20]) -> dict:
    """
    Recommend the most popular items (books) in the training set to all users in the test set.
    Returns NDCG@10 and the list of most popular books.
    """
    # Compute most popular books by count of ratings in train set
    pop_books = (
        train_df.groupby('book_id')['rating'].count()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    # For each user in test set, recommend the same top-N popular books
    user_ids = test_df['user_id'].unique()
    recommendations = {user_id: pop_books for user_id in user_ids}

    # Prepare ground truth for ranking metrics
    ground_truth = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
    ranking_metrics = evaluate_ranking_metrics(recommendations, ground_truth, k_list=k_list)
    result = dict(ranking_metrics)
    result['top_n_books'] = pop_books
    return result
