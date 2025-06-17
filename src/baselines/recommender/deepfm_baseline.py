import itertools

import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from src.evaluation.ranking_metrics import evaluate_ranking_metrics
from .ranking_utils import get_top_n_recommendations


def run_deepfm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k_list=[5, 10, 20]) -> dict:
    """
    Runs the DeepFM baseline for recommendation.

    This function preprocesses the data, defines feature columns for DeepCTR, and then
    trains and evaluates the DeepFM model.

    Args:
        train_df: DataFrame for training. Expected columns: ['user_id', 'book_id', 'rating'].
        test_df: DataFrame for testing. Expected columns: ['user_id', 'book_id', 'rating'].

    Returns:
        A dictionary containing the final evaluation metrics (MSE and NDCG@10).
    """
    logger.info("Starting DeepFM baseline...")

    # 1. Data Preprocessing
    logger.info("Preprocessing data for DeepFM...")
    data = pd.concat([train_df, test_df], ignore_index=True)
    sparse_features = ["user_id", "book_id"]
    target = "rating"

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2. Define Feature Columns
    logger.info("Defining feature columns for DeepCTR...")
    feat_voc_size = {feat: data[feat].nunique() for feat in sparse_features}
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=feat_voc_size[feat], embedding_dim=4)
        for feat in sparse_features
    ]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3. Split data for training and testing
    train = data.iloc[: len(train_df)]
    test = data.iloc[len(train_df) :]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    train_labels = train[target].values
    test_labels = test[target].values

    # 4. Instantiate and Train Model
    logger.info("Instantiating and training DeepFM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        task="regression",
        device=device,
    )
    model.compile("adam", "mse", metrics=["mse"])
    model.fit(
        train_model_input,
        train_labels,
        batch_size=256,
        epochs=10,
        verbose=1,
        validation_data=(test_model_input, test_labels),
    )

    # 5. Evaluate for Ranking (NDCG, Precision@K, Recall@K) using RankerEval
    logger.info("Evaluating model for ranking metrics (RankerEval)...")
    all_users = data["user_id"].unique()
    all_items = data["book_id"].unique()
    all_pairs = pd.DataFrame(
        list(itertools.product(all_users, all_items)), columns=["user_id", "book_id"]
    )
    train_pairs = train[["user_id", "book_id"]].drop_duplicates()
    anti_test_df = pd.merge(
        all_pairs, train_pairs, on=["user_id", "book_id"], how="left", indicator=True
    )
    anti_test_df = anti_test_df[anti_test_df["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    anti_test_model_input = {name: anti_test_df[name] for name in feature_names}
    anti_test_predictions = model.predict(anti_test_model_input, batch_size=256)
    anti_test_df["rating"] = anti_test_predictions
    # Get top-N for all K
    top_n = get_top_n_recommendations(anti_test_df, n=20)
    # Convert to {user_id: [item_id, ...]}
    recommendations = {user: [item for item, _ in items] for user, items in top_n.items()}
    ground_truth = test.groupby('user_id')['book_id'].apply(list).to_dict()
    ranking_metrics = evaluate_ranking_metrics(recommendations, ground_truth, k_list=k_list)
    logger.info(f"DeepFM ranking metrics: {ranking_metrics}")

    # 6. Evaluate for Accuracy (MSE)
    logger.info("Evaluating model on the test set...")
    predictions = model.predict(test_model_input, batch_size=256)
    import numpy as np
    mse = np.mean((test_labels - predictions) ** 2)
    rmse = np.sqrt(mse)
    logger.info(f"DeepFM baseline RMSE: {rmse:.4f}")
    metrics = {"mse": mse, "rmse": rmse}
    metrics.update(ranking_metrics)
    logger.info(f"DeepFM metrics: {metrics}")
    logger.success("DeepFM baseline finished successfully.")
    return metrics
