import itertools

import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from loguru import logger
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc
import psutil

import os
from datetime import datetime

def run_deepfm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k_list=[5, 10, 20], model_save_path: str = None) -> dict:
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
    
    # Memory monitoring
    process = psutil.Process(os.getpid())
    logger.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 1. Data Preprocessing
    logger.info("Preprocessing data for DeepFM...")
    logger.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
    
    # Validate input data
    required_columns = ['user_id', 'book_id', 'rating']
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in train_df")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in test_df")
    
    # Check for null values
    logger.info(f"Train null values: {train_df.isnull().sum().to_dict()}")
    logger.info(f"Test null values: {test_df.isnull().sum().to_dict()}")
    
    # Remove any null values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.info(f"After removing nulls - Train: {train_df.shape}, Test: {test_df.shape}")
    
    try:
        data = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Combined data shape: {data.shape}")
        logger.info(f"Memory after concat: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        sparse_features = ["user_id", "book_id"]
        target = "rating"

        # Encode features with better memory management
        for feat in sparse_features:
            logger.info(f"Encoding feature: {feat}")
            unique_vals = data[feat].nunique()
            logger.info(f"Unique values in {feat}: {unique_vals}")
            
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat].astype(str))
            logger.info(f"Encoded {feat}, range: {data[feat].min()} to {data[feat].max()}")
            
            # Force garbage collection
            gc.collect()

        logger.info(f"Memory after encoding: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 2. Define Feature Columns
        logger.info("Defining feature columns for DeepCTR...")
        feat_voc_size = {feat: data[feat].nunique() for feat in sparse_features}
        logger.info(f"Feature vocabulary sizes: {feat_voc_size}")
        
        # Use smaller embedding dimensions to reduce memory usage
        embedding_dim = min(4, max(2, int(np.sqrt(min(feat_voc_size.values())))))
        logger.info(f"Using embedding dimension: {embedding_dim}")
        
        fixlen_feature_columns = [
            SparseFeat(feat, vocabulary_size=feat_voc_size[feat], embedding_dim=embedding_dim)
            for feat in sparse_features
        ]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        logger.info(f"Feature names: {feature_names}")

        # 3. Split data for training and testing
        logger.info("Splitting data...")
        train = data.iloc[: len(train_df)].copy()
        test = data.iloc[len(train_df) :].copy()
        
        logger.info(f"Train split shape: {train.shape}, Test split shape: {test.shape}")
        
        # Convert to appropriate data types
        for name in feature_names:
            train[name] = train[name].astype(np.int32)
            test[name] = test[name].astype(np.int32)
        
        train_model_input = {name: train[name].values for name in feature_names}
        test_model_input = {name: test[name].values for name in feature_names}
        train_labels = train[target].values.astype(np.float32)
        test_labels = test[target].values.astype(np.float32)
        
        logger.info(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
        logger.info(f"Test labels shape: {test_labels.shape}, dtype: {test_labels.dtype}")
        logger.info(f"Memory after data preparation: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # Clear unnecessary data
        del data, train, test
        gc.collect()

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

    # 4. Instantiate and Train Model
    logger.info("Instantiating and training DeepFM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        model = DeepFM(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            task="regression",
            device=device,
        )
        logger.info("Model instantiated successfully")
        
        model.compile("adam", "mse", metrics=["mse"])
        logger.info("Model compiled successfully")
        
        logger.info(f"Memory before training: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        model.fit(
            train_model_input,
            train_labels,
            batch_size=256,
            epochs=100,  # Keep reduced epochs for testing
            verbose=1,
            validation_data=(test_model_input, test_labels),
        )
        logger.info("Model training completed")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

    # 6. Evaluate for Accuracy (MSE)
    logger.info("Evaluating model on the test set...")
    try:
        predictions = model.predict(test_model_input, batch_size=256)
        mse = np.mean((test_labels - predictions.flatten()) ** 2)
        rmse = np.sqrt(mse)
        logger.info(f"DeepFM baseline RMSE: {rmse:.4f}")
        metrics = {"mse": mse, "rmse": rmse}
        logger.info(f"DeepFM metrics: {metrics}")
        logger.success("DeepFM baseline finished successfully.")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
