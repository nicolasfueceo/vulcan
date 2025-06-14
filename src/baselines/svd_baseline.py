import pandas as pd
from loguru import logger
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse

from src.baselines.ranking_utils import (
    calculate_ndcg,
    get_top_n_recommendations,
)


def run_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Runs the SVD baseline, evaluating with RMSE, MAE, and NDCG@10.
    """
    logger.info("Starting SVD baseline...")

    # 1. Load Data
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    testset = list(test_df[['user_id', 'book_id', 'rating']].itertuples(index=False, name=None))

    # Build an anti-test set for generating predictions for items not in the training set
    anti_testset = trainset.build_anti_testset()

    # 2. Train Model
    logger.info("Training SVD model...")
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)
    model.fit(trainset)

    # 3. Evaluate for Accuracy (RMSE, MAE)
    logger.info("Evaluating model for accuracy (RMSE, MAE)...")
    accuracy_predictions = model.test(testset)
    rmse_score = rmse(accuracy_predictions, verbose=False)
    mae_score = mae(accuracy_predictions, verbose=False)
    logger.info(f"SVD baseline RMSE: {rmse_score:.4f}, MAE: {mae_score:.4f}")

    # 4. Evaluate for Ranking (NDCG)
    logger.info("Evaluating model for ranking (NDCG@10)...")
    ranking_predictions = model.test(anti_testset)

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(
        ranking_predictions,
        columns=["user_id", "book_id", "true_rating", "rating", "details"],
    )

    top_n = get_top_n_recommendations(predictions_df, n=10)
    ndcg_score = calculate_ndcg(top_n, test_df, n=10)
    logger.info(f"SVD baseline NDCG@10: {ndcg_score:.4f}")

    # 5. Return Metrics
    metrics = {"rmse": rmse_score, "mae": mae_score, "ndcg@10": ndcg_score}
    logger.success("SVD baseline finished successfully.")
    return metrics
