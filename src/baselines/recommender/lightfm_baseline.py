import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from sklearn.metrics import ndcg_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import coo_matrix
from loguru import logger
import gc


def run_lightfm_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_epochs=20,
    learning_rate=0.05,
    loss='warp',
    user_features=None,
    item_features=None,
    num_threads=2,
    no_components=50,
    item_alpha=1e-6,
) -> dict:
    """
    Runs the LightFM baseline for recommendation as in the LightFM documentation.
    Supports both pure collaborative filtering (cold-start) and hybrid (with side features) scenarios.
    Args:
        train_df: DataFrame for training. ['user_id', 'book_id', 'rating']
        test_df: DataFrame for testing. ['user_id', 'book_id', 'rating']
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the model.
        loss: Loss function ('warp', 'bpr', etc.)
        user_features: User feature matrix (optional, for hybrid recommender)
        item_features: Item feature matrix (optional, for hybrid recommender)
        num_threads: Number of threads to use (LightFM default: 1)
        no_components: Embedding dimension
        item_alpha: Regularization for item features
    Returns:
        Dictionary of evaluation metrics (RMSE, NDCG@10, precision@k, AUC, etc.)
    """
    logger.info("Starting LightFM baseline (docs-style, supports hybrid and cold-start)...")
    k_list = [5, 10, 20]  # Used for precision@k and can be changed as needed
    k_list = [5, 10, 20]  # Used for precision@k and can be changed as needed
    """
    Runs the LightFM baseline for recommendation.
    
    Args:
        train_df: DataFrame for training. Expected columns: ['user_id', 'book_id', 'rating'].
        test_df: DataFrame for testing. Expected columns: ['user_id', 'book_id', 'rating'].
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the model.
        loss: Loss function ('warp', 'bpr', 'logistic', 'regression').
        
    Returns:
        A dictionary containing evaluation metrics.
    """
    logger.info("Starting LightFM baseline...")
    
    # Validate input data
    required_columns = ['user_id', 'book_id', 'rating']
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in train_df")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in test_df")
    
    # Remove null values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    try:
        # Create dataset
        dataset = Dataset()
        
        # Get all unique users and items
        all_users = set(train_df['user_id'].unique()) | set(test_df['user_id'].unique())
        all_items = set(train_df['book_id'].unique()) | set(test_df['book_id'].unique())
        
        logger.info(f"Total users: {len(all_users)}, Total items: {len(all_items)}")
        
        # Fit the dataset
        dataset.fit(users=all_users, items=all_items)
        
        # Build interaction matrices
        def build_interactions(df):
            interactions, weights = dataset.build_interactions(
                [(row['user_id'], row['book_id'], row['rating']) for _, row in df.iterrows()]
            )
            return interactions, weights
        
        train_interactions, train_weights = build_interactions(train_df)
        test_interactions, test_weights = build_interactions(test_df)
        # Ensure test_interactions has only users/items present in train_interactions
        train_user_count, train_item_count = train_interactions.shape
        test_user_count, test_item_count = test_interactions.shape
        if test_user_count > train_user_count or test_item_count > train_item_count:
            logger.warning(f"Filtering test_interactions from shape {test_interactions.shape} to match train_interactions {train_interactions.shape}")
            test_interactions = test_interactions[:train_user_count, :train_item_count]
            if test_weights is not None:
                test_weights = test_weights[:train_user_count, :train_item_count]
        # Convert to CSR for batch slicing
        train_interactions_csr = train_interactions.tocsr()
        train_weights_csr = train_weights.tocsr() if train_weights is not None else None
        
        logger.info(f"Train interactions shape: {train_interactions.shape}")
        logger.info(f"Test interactions shape: {test_interactions.shape}")
        
        # Initialize and train model (see LightFM docs)
        model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            no_components=no_components,
            item_alpha=item_alpha,
            random_state=42
        )
        logger.info(f"Training LightFM model for {num_epochs} epochs (loss={loss}) using fit (as in docs)...")
        if user_features is not None or item_features is not None:
            model.fit(
                train_interactions,
                user_features=user_features,
                item_features=item_features,
                epochs=num_epochs,
                num_threads=num_threads,
                verbose=True,
            )
        else:
            model.fit(
                train_interactions,
                epochs=num_epochs,
                num_threads=num_threads,
                verbose=True,
            )
        
        # Evaluate model
        metrics = {}

        # Compute RMSE on the test set
        user_map, item_map, *_ = dataset.mapping()
        inv_user_map = {v: k for k, v in user_map.items()}
        inv_item_map = {v: k for k, v in item_map.items()}

        # Prepare arrays of predictions and true ratings
        y_true = []
        y_pred = []
        for _, row in test_df.iterrows():
            u_raw, i_raw, rating = row['user_id'], row['book_id'], row['rating']
            if u_raw not in user_map or i_raw not in item_map:
                # Skip unknown user/item combos
                continue
            u_idx = user_map[u_raw]
            i_idx = item_map[i_raw]
            pred = model.predict(u_idx, i_idx)
            y_true.append(rating)
            y_pred.append(pred)
        
        if len(y_true) == 0:
            rmse = np.nan
            logger.warning("No valid pairs for RMSE calculation in LightFM baseline. Skipping RMSE computation.")
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = rmse
        logger.info(f"LightFM RMSE: {rmse:.4f}")
        
        # NDCG@10 calculation (docs-style: use precision_at_k, auc_score, etc.)
        metrics['ndcg_at_10'] = np.nan
        try:
            for k in k_list:
                precision = precision_at_k(model, test_interactions, k=k, train_interactions=train_interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
                metrics[f'precision_at_{k}'] = precision
            # AUC (docs-style)
            train_auc = auc_score(model, train_interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
            test_auc = auc_score(model, test_interactions, train_interactions=train_interactions, user_features=user_features, item_features=item_features, num_threads=num_threads).mean()
            metrics['train_auc'] = train_auc
            metrics['test_auc'] = test_auc
            logger.info(f"LightFM train AUC: {train_auc:.4f}, test AUC: {test_auc:.4f}")
            # NDCG@5 (approximate, using scores)
            n_users, n_items = test_interactions.shape
            max_user = min(n_users, model.user_embeddings.shape[0])
            valid_users = np.arange(max_user)
            true_relevance_mat = test_interactions[:max_user].toarray()
            user_ids = np.repeat(valid_users, n_items)
            item_ids = np.tile(np.arange(n_items), max_user)
            if user_features is not None or item_features is not None:
                scores_flat = model.predict(user_ids, item_ids, user_features=user_features, item_features=item_features)
            else:
                scores_flat = model.predict(user_ids, item_ids)
            scores_mat = scores_flat.reshape(max_user, n_items)
            ndcg_scores = []
            for u in range(max_user):
                true_rel = true_relevance_mat[u]
                if np.sum(true_rel) == 0:
                    continue
                ndcg = ndcg_score([true_rel], [scores_mat[u]], k=5)
                ndcg_scores.append(ndcg)
            ndcg_at_5 = np.mean(ndcg_scores) if ndcg_scores else np.nan
            metrics['ndcg_at_5'] = ndcg_at_5
            logger.info(f"LightFM NDCG@5: {ndcg_at_5:.4f} (fast batch, skipped {n_users-max_user} users not in model)")
        except Exception as ndcg_e:
            logger.warning(f"Could not compute LightFM metrics: {ndcg_e}")
        
        logger.info(f"LightFM metrics: {metrics}")
        logger.success("LightFM baseline finished successfully.")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in LightFM baseline: {e}")
        raise
    finally:
        # Clean up memory
        gc.collect()
