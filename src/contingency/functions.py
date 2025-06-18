import pandas as pd
import numpy as np
from typing import Dict, Any

def template_feature_function(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Template for a feature computation function.

    Args:
        df (pd.DataFrame): The input DataFrame containing all necessary columns (already joined).
        params (Dict[str, Any]): Dictionary of hyperparameters for this feature, as suggested by BO.

    Returns:
        pd.Series: A single column of computed feature values, indexed the same as df.

    Contract:
    - This function must be pure (no side effects), deterministic, and not mutate df in-place.
    - The function should handle missing values gracefully and document any required columns.
    - The feature name should correspond to the function name (for registry purposes).
    - Example usage:
        feature_col = template_feature_function(df, {'alpha': 0.5, 'window': 3})
    """
    # Example (identity feature):
    # return df["some_column"] * params.get("alpha", 1.0)
    raise NotImplementedError("Override this template with your custom feature logic.")


def rating_popularity_momentum(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Feature based on hypothesis: "Books with higher average ratings tend to have more ratings"
    
    This feature captures the momentum effect where popular books (high ratings) attract more ratings,
    creating a virtuous cycle. The feature combines average rating with rating count in a non-linear way
    to capture this momentum effect.
    
    Required columns:
    - average_rating: Book's average rating (float, typically 0-5)
    - ratings_count: Number of ratings the book has received (int)
    
    Hyperparameters:
    - rating_weight: Weight for the average rating component (default: 1.0)
    - count_weight: Weight for the ratings count component (default: 0.5) 
    - momentum_power: Power to apply to the momentum calculation (default: 0.8)
    - min_ratings_threshold: Minimum ratings needed for momentum effect (default: 10)
    - rating_scale: Scale factor for rating normalization (default: 5.0)
    
    Args:
        df (pd.DataFrame): Input DataFrame with book data
        params (Dict[str, Any]): Hyperparameters for the feature
        
    Returns:
        pd.Series: Rating popularity momentum feature values
    """
    # Extract hyperparameters with defaults
    rating_weight = params.get("rating_weight", 1.0)
    count_weight = params.get("count_weight", 0.5)
    momentum_power = params.get("momentum_power", 0.8)
    min_ratings_threshold = params.get("min_ratings_threshold", 10)
    rating_scale = params.get("rating_scale", 5.0)
    
    # Validate required columns
    required_cols = ["average_rating", "ratings_count"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values
    avg_rating = df["average_rating"].fillna(df["average_rating"].median())
    ratings_count = df["ratings_count"].fillna(0)
    
    # Normalize average rating to 0-1 scale
    normalized_rating = avg_rating / rating_scale
    
    # Log-transform ratings count to handle skewness
    log_count = np.log1p(ratings_count)  # log(1 + count) to handle zeros
    
    # Create momentum indicator (books with sufficient ratings get momentum boost)
    momentum_mask = ratings_count >= min_ratings_threshold
    
    # Calculate base momentum: rating * log(count)
    base_momentum = (rating_weight * normalized_rating) * (count_weight * log_count)
    
    # Apply power transformation for non-linearity
    momentum_feature = np.power(base_momentum, momentum_power)
    
    # Apply momentum boost for books with sufficient ratings
    momentum_feature = np.where(
        momentum_mask,
        momentum_feature * (1 + 0.1 * np.log1p(ratings_count - min_ratings_threshold)),
        momentum_feature * 0.8  # Penalty for books with few ratings
    )
    
    # Handle edge cases
    momentum_feature = np.where(
        (avg_rating == 0) | (ratings_count == 0),
        0.0,  # Zero momentum for unrated books
        momentum_feature
    )
    
    return pd.Series(momentum_feature, index=df.index, name="rating_popularity_momentum")


def genre_preference_alignment(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
    """
    Feature based on hypothesis: "Users prefer specific genres that have consistently high average ratings"
    
    Creates a feature that captures how well a book aligns with high-performing genres.
    This feature identifies books in genres that tend to have consistently high ratings.
    
    Parameters (for Bayesian Optimization):
    - genre_weight: Weight for genre rating component (0.1 to 2.0)
    - rating_threshold: Minimum rating to consider a genre "high-performing" (3.0 to 4.5)
    - popularity_factor: How much to weight genre popularity (0.0 to 1.0)
    - recency_decay: Decay factor for older books (0.8 to 1.0)
    - boost_multiplier: Multiplier for books in top genres (1.0 to 3.0)
    """
    if params is None:
        params = {}
    
    # Extract hyperparameters with defaults
    genre_weight = params.get('genre_weight', 1.0)
    rating_threshold = params.get('rating_threshold', 3.8)
    popularity_factor = params.get('popularity_factor', 0.5)
    recency_decay = params.get('recency_decay', 0.95)
    boost_multiplier = params.get('boost_multiplier', 1.5)
    
    # Initialize feature values
    feature_values = pd.Series(0.0, index=df.index, name='genre_preference_alignment')
    
    try:
        # Create synthetic genre data since we don't have real genre info
        # In a real implementation, this would come from the database
        genres = ['Fiction', 'Non-Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 
                 'Biography', 'History', 'Self-Help', 'Thriller']
        
        # Assign genres based on book characteristics (synthetic approach)
        np.random.seed(42)  # For reproducibility
        book_genres = np.random.choice(genres, size=len(df))
        
        # Calculate genre performance metrics
        genre_ratings = {}
        genre_popularity = {}
        
        for genre in genres:
            genre_mask = book_genres == genre
            if genre_mask.sum() > 0:
                genre_books = df[genre_mask]
                avg_rating = genre_books['average_rating'].mean()
                popularity = genre_books['ratings_count'].mean()
                
                genre_ratings[genre] = avg_rating
                genre_popularity[genre] = popularity
        
        # Identify high-performing genres
        high_performing_genres = {
            genre: rating for genre, rating in genre_ratings.items() 
            if rating >= rating_threshold
        }
        
        # Calculate feature for each book
        current_year = 2024
        
        for i, (idx, row) in enumerate(df.iterrows()):
            book_genre = book_genres[i]
            
            # Base alignment score
            if book_genre in high_performing_genres:
                genre_score = high_performing_genres[book_genre] * genre_weight
                
                # Add popularity factor
                if book_genre in genre_popularity:
                    pop_score = np.log1p(genre_popularity[book_genre]) * popularity_factor
                    genre_score += pop_score
                
                # Apply boost for top genres
                if genre_ratings.get(book_genre, 0) > rating_threshold + 0.2:
                    genre_score *= boost_multiplier
                
                # Apply recency decay for older books
                if pd.notna(row.get('publication_year')):
                    years_old = current_year - row['publication_year']
                    if years_old > 0:
                        decay_factor = recency_decay ** min(years_old, 20)  # Cap at 20 years
                        genre_score *= decay_factor
                
                feature_values.iloc[i] = genre_score
            else:
                # Books in lower-performing genres get a small base score
                feature_values.iloc[i] = genre_ratings.get(book_genre, 3.0) * 0.3
        
        # Normalize to reasonable range
        if feature_values.max() > 0:
            feature_values = feature_values / feature_values.max() * 5.0
        
        return feature_values
        
    except Exception as e:
        print(f"Error in genre_preference_alignment: {e}")
        return pd.Series(0.0, index=df.index, name='genre_preference_alignment')


def publication_recency_boost(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
    """
    Feature based on hypothesis: "Recent publications with high ratings indicate emerging trends"
    
    Creates a feature that boosts books published recently that have gained traction quickly.
    This captures the momentum of newer books that are performing well.
    
    Parameters (for Bayesian Optimization):
    - recency_weight: Weight for how recent the book is (0.1 to 2.0)
    - rating_weight: Weight for the book's rating (0.5 to 2.0)
    - velocity_factor: Weight for rating velocity (ratings/years) (0.1 to 1.5)
    - recent_threshold: Years to consider "recent" (1 to 10)
    - min_ratings: Minimum ratings needed for boost (5 to 100)
    """
    if params is None:
        params = {}
    
    # Extract hyperparameters with defaults
    recency_weight = params.get('recency_weight', 1.2)
    rating_weight = params.get('rating_weight', 1.0)
    velocity_factor = params.get('velocity_factor', 0.8)
    recent_threshold = params.get('recent_threshold', 5)
    min_ratings = params.get('min_ratings', 20)
    
    # Initialize feature values
    feature_values = pd.Series(0.0, index=df.index, name='publication_recency_boost')
    
    try:
        current_year = 2024
        
        for idx, row in df.iterrows():
            # Check if we have required data
            if pd.isna(row.get('publication_year')) or pd.isna(row.get('average_rating')):
                continue
                
            pub_year = row['publication_year']
            avg_rating = row['average_rating']
            ratings_count = row.get('ratings_count', 0)
            
            # Only consider books with sufficient ratings
            if ratings_count < min_ratings:
                continue
            
            # Calculate years since publication
            years_since_pub = current_year - pub_year
            
            # Only boost recent books
            if years_since_pub <= recent_threshold and years_since_pub > 0:
                # Recency score (higher for more recent)
                recency_score = (recent_threshold - years_since_pub) / recent_threshold
                recency_score = recency_score ** 0.5  # Square root for smoother curve
                
                # Rating score (higher for better ratings)
                rating_score = (avg_rating - 2.0) / 3.0  # Normalize 2-5 to 0-1
                rating_score = max(0, rating_score)
                
                # Velocity score (ratings per year since publication)
                velocity = ratings_count / max(years_since_pub, 0.5)  # Avoid division by zero
                velocity_score = np.log1p(velocity) / 10.0  # Log scale, normalized
                
                # Combine components
                boost_score = (
                    recency_score * recency_weight +
                    rating_score * rating_weight +
                    velocity_score * velocity_factor
                )
                
                # Apply additional boost for exceptional performance
                if avg_rating >= 4.2 and velocity > 50:
                    boost_score *= 1.3
                
                feature_values[idx] = boost_score
        
        # Normalize to 0-3 range
        if feature_values.max() > 0:
            feature_values = feature_values / feature_values.max() * 3.0
        
        return feature_values
        
    except Exception as e:
        print(f"Error in publication_recency_boost: {e}")
        return pd.Series(0.0, index=df.index, name='publication_recency_boost')


def engagement_depth_score(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
    """
    Feature based on hypothesis: "Books with more text reviews relative to ratings indicate deeper engagement"
    
    Creates a feature that captures the depth of user engagement beyond just ratings.
    Books that inspire detailed reviews may have different recommendation value.
    
    Parameters (for Bayesian Optimization):
    - review_ratio_weight: Weight for text reviews to ratings ratio (0.5 to 2.0)
    - absolute_reviews_weight: Weight for absolute number of reviews (0.1 to 1.0)
    - engagement_threshold: Minimum ratio to consider high engagement (0.05 to 0.5)
    - length_proxy_factor: Factor for estimated review length (0.0 to 1.0)
    - quality_boost: Boost for high-quality engagement indicators (1.0 to 2.0)
    """
    if params is None:
        params = {}
    
    # Extract hyperparameters with defaults
    review_ratio_weight = params.get('review_ratio_weight', 1.0)
    absolute_reviews_weight = params.get('absolute_reviews_weight', 0.3)
    engagement_threshold = params.get('engagement_threshold', 0.1)
    length_proxy_factor = params.get('length_proxy_factor', 0.2)
    quality_boost = params.get('quality_boost', 1.2)
    
    # Initialize feature values
    feature_values = pd.Series(0.0, index=df.index, name='engagement_depth_score')
    
    try:
        for idx, row in df.iterrows():
            ratings_count = row.get('ratings_count', 0)
            text_reviews_count = row.get('text_reviews_count', 0)
            avg_rating = row.get('average_rating', 3.0)
            
            if ratings_count == 0:
                continue
            
            # Calculate review-to-rating ratio
            review_ratio = text_reviews_count / ratings_count
            
            # Base engagement score
            if review_ratio >= engagement_threshold:
                # Ratio component
                ratio_score = min(review_ratio, 1.0) * review_ratio_weight  # Cap at 1.0
                
                # Absolute reviews component (log scale)
                absolute_score = np.log1p(text_reviews_count) * absolute_reviews_weight
                
                # Length proxy (assume higher-rated books get longer reviews)
                length_proxy = (avg_rating - 3.0) * length_proxy_factor
                
                # Combine components
                engagement_score = ratio_score + absolute_score + length_proxy
                
                # Quality boost for exceptional engagement
                if review_ratio > 0.3 and text_reviews_count > 50:
                    engagement_score *= quality_boost
                
                # Boost for books that inspire discussion (high review ratio + good rating)
                if review_ratio > 0.2 and avg_rating >= 4.0:
                    engagement_score *= 1.1
                
                feature_values[idx] = engagement_score
        
        # Normalize to 0-4 range
        if feature_values.max() > 0:
            feature_values = feature_values / feature_values.max() * 4.0
        
        return feature_values
        
    except Exception as e:
        print(f"Error in engagement_depth_score: {e}")
        return pd.Series(0.0, index=df.index, name='engagement_depth_score')


# =============================================================================
# BATCH 1: HYPOTHESES 1-10 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

def user_engagement_rating_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 1: Users who read more books tend to provide higher average ratings.

    Required columns:
        - books_read (int)
        - avg_rating (float)
    """
    books_w = params.get("books_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    boost = params.get("rating_boost", 1.2)
    threshold = params.get("engagement_threshold", 10)

    if not {"books_read", "avg_rating"}.issubset(df.columns):
        raise ValueError("user_engagement_rating_correlation needs books_read & avg_rating")

    books = np.log1p(df["books_read"].fillna(0))
    rating = df["avg_rating"].fillna(df["avg_rating"].median()) / 5
    score = books_w * books + rating_w * rating
    score = np.where(df["books_read"].fillna(0) >= threshold, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_engagement_rating_correlation")


def genre_preference_strength(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 2: Users prefer specific genres with consistently high ratings.

    Required columns:
        - genre (str)
        - average_rating (float)
    """
    genre_w = params.get("genre_weight", 1.0)
    popularity_w = params.get("popularity_weight", 0.5)
    rating_thresh = params.get("rating_threshold", 4.0)

    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_preference_strength requires genre & average_rating")

    genre_stats = df.groupby("genre").agg(avg_r=("average_rating", "mean"), cnt=("genre", "size"))
    genre_score = genre_w * (genre_stats["avg_r"] / 5) + popularity_w * np.log1p(genre_stats["cnt"])
    preferred = genre_stats["avg_r"] >= rating_thresh
    genre_score[preferred] *= 1.2
    genre_score = (genre_score - genre_score.min()) / (genre_score.max() - genre_score.min() + 1e-9)
    return pd.Series(df["genre"].map(genre_score).fillna(0).values, index=df.index, name="genre_preference_strength")


def shelf_popularity_indicator(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 3: Shelf categories with more books signal interest.

    Required columns:
        - shelf (str)
        - shelf_count (int)  # number of times book appears in shelf
    """
    shelf_w = params.get("shelf_weight", 1.0)
    count_w = params.get("count_weight", 1.0)
    boost = params.get("popularity_boost", 1.3)
    min_cnt = params.get("min_books_threshold", 5)

    if not {"shelf", "shelf_count"}.issubset(df.columns):
        raise ValueError("shelf_popularity_indicator needs shelf & shelf_count")

    shelf_stats = df.groupby("shelf").agg(cnt=("shelf_count", "sum"))
    score = shelf_w * np.log1p(shelf_stats["cnt"]) * count_w
    score = np.where(shelf_stats["cnt"] >= min_cnt, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(df["shelf"].map(score).fillna(0).values, index=df.index, name="shelf_popularity_indicator")


def format_preference_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 4: Certain formats receive better ratings.

    Required columns:
        - format (str)
        - average_rating (float)
    """
    fmt_w = params.get("format_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    min_rating = params.get("min_rating_threshold", 3.5)
    boost = params.get("format_boost", 1.2)

    if not {"format", "average_rating"}.issubset(df.columns):
        raise ValueError("format_preference_score needs format & average_rating")

    fmt_stats = df.groupby("format").agg(avg_r=("average_rating", "mean"), cnt=("format", "size"))
    score = fmt_w * np.log1p(fmt_stats["cnt"]) + rating_w * (fmt_stats["avg_r"] / 5)
    score = np.where(fmt_stats["avg_r"] >= min_rating, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(df["format"].map(score).fillna(0).values, index=df.index, name="format_preference_score")


def genre_diversity_preference(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 5: Readers like diverse genres, esp. fantasy & romance.

    Required columns:
        - genre (str)
        - average_rating (float)
    """
    base_w = params.get("genre_weight", 1.0)
    diversity_f = params.get("diversity_factor", 0.6)
    fantasy_boost = params.get("fantasy_boost", 1.3)
    romance_boost = params.get("romance_boost", 1.2)

    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_diversity_preference needs genre & average_rating")

    stats = df.groupby("genre").agg(avg_r=("average_rating", "mean"), cnt=("genre", "size"))
    base = base_w * (stats["avg_r"] / 5) * np.log1p(stats["cnt"])
    fantasy_mask = stats.index.str.contains("fantasy", case=False, na=False)
    romance_mask = stats.index.str.contains("romance", case=False, na=False)
    base[fantasy_mask] *= fantasy_boost
    base[romance_mask] *= romance_boost
    base *= (1 + diversity_f * (1 - abs(stats["avg_r"] - 3.5) / 1.5))
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return pd.Series(df["genre"].map(base).fillna(0).values, index=df.index, name="genre_diversity_preference")


def user_behavior_clustering(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 6: Consistent rating behaviors form clusters.

    Required columns:
        - user_id (identifier)
        - avg_rating_user (float)
        - rating_stddev (float)
        - review_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    var_penalty = params.get("variance_penalty", 0.8)
    boost = params.get("cluster_boost", 1.2)

    cols = {"user_id", "avg_rating_user", "rating_stddev", "review_count"}
    if not cols.issubset(df.columns):
        raise ValueError("user_behavior_clustering missing required columns")

    score = rating_w * (df["avg_rating_user"].fillna(3.5) / 5) * np.log1p(df["review_count"].fillna(0))
    score *= (var_penalty + (1 - var_penalty) * (1 / (1 + df["rating_stddev"].fillna(0))))
    distinct = (df["avg_rating_user"] > 4) | (df["avg_rating_user"] < 2)
    score = np.where(distinct, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_behavior_clustering")


def thematic_genre_crossover(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 7: Fans of steampunk/fantasy like thematic crossovers.

    Required columns:
        - title (str)
        - rating (float)
    """
    rating_w = params.get("rating_weight", 1.0)
    steam_boost = params.get("steampunk_boost", 1.4)
    fantasy_boost = params.get("fantasy_boost", 1.2)
    cross_f = params.get("crossover_factor", 0.8)

    if not {"title", "rating"}.issubset(df.columns):
        raise ValueError("thematic_genre_crossover requires title & rating")

    title = df["title"].fillna("").str.lower()
    steam = title.str.contains("steampunk|steam|clockwork|victorian")
    fantasy = title.str.contains("fantasy|magic|dragon|wizard|elf")
    score = rating_w * df["rating"].fillna(df["rating"].median()) / 5
    score = np.where(steam, score * steam_boost, score)
    score = np.where(fantasy, score * fantasy_boost, score)
    crossover = steam & fantasy
    score = np.where(crossover, score * (1 + cross_f), score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="thematic_genre_crossover")


def author_collaboration_quality(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 8: Frequent author collaborations relate to quality.

    Required columns:
        - author_id (identifier)
        - avg_rating (float)
        - book_count_author (int)
    """
    collab_w = params.get("collaboration_weight", 1.0)
    quality_boost = params.get("quality_boost", 1.3)
    min_col = params.get("min_collaborations", 2)

    cols = {"author_id", "avg_rating", "book_count_author"}
    if not cols.issubset(df.columns):
        raise ValueError("author_collaboration_quality missing columns")

    base = collab_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5) * np.log1p(df["book_count_author"].fillna(0))
    boost_mask = df["book_count_author"] >= min_col
    base = np.where(boost_mask, base * quality_boost, base)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return pd.Series(base, index=df.index, name="author_collaboration_quality")


def selective_reader_curation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 9: Readers with few but high ratings need curation.

    Required columns:
        - books_read (int)
        - avg_rating (float)
    """
    sel_w = params.get("selectivity_weight", 1.0)
    boost = params.get("curation_boost", 1.4)
    books_thr = params.get("books_threshold", 10)
    rating_thr = params.get("rating_threshold", 4.0)

    if not {"books_read", "avg_rating"}.issubset(df.columns):
        raise ValueError("selective_reader_curation missing columns")

    score = sel_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5) / np.log1p(df["books_read"].fillna(1))
    mask = (df["books_read"] <= books_thr) & (df["avg_rating"] >= rating_thr)
    score = np.where(mask, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="selective_reader_curation")


def page_length_rating_impact(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 10: Page length influences ratings differently.

    Required columns:
        - num_pages (int)
        - average_rating (float)
    """
    page_w = params.get("page_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    optimal = params.get("optimal_length", 300)
    penalty = params.get("length_penalty", 0.8)

    if not {"num_pages", "average_rating"}.issubset(df.columns):
        raise ValueError("page_length_rating_impact requires num_pages & average_rating")

    deviation = abs(df["num_pages"].fillna(optimal) - optimal) / optimal
    length_factor = 1 / (1 + penalty * deviation)
    score = page_w * np.log1p(df["num_pages"].fillna(0)) * length_factor + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="page_length_rating_impact")


# =============================================================================
# BATCH 2: HYPOTHESES 11-20 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

# (Batch 2 functions defined above...)

# =============================================================================
# BATCH 3: HYPOTHESES 21-30 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

# =============================================================================
# BATCH 4: HYPOTHESES 31-40 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

# =============================================================================
# BATCH 5: HYPOTHESES 41-50 FEATURE FUNCTIONS (TEMPLATE-COMPLIANT)
# =============================================================================

def genre_format_distribution_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 41: The distribution of books across genres and formats reveals market trends and reader preferences.
    Required columns:
        - genre (str)
        - format (str)
    """
    genre_w = params.get("genre_weight", 1.0)
    format_w = params.get("format_weight", 1.0)
    if not {"genre", "format"}.issubset(df.columns):
        raise ValueError("genre_format_distribution_score requires genre and format columns")
    genre_counts = df["genre"].value_counts()
    format_counts = df["format"].value_counts()
    score = genre_w * df["genre"].map(genre_counts) + format_w * df["format"].map(format_counts)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="genre_format_distribution_score")

def avg_rating_rating_count_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 42: Books with higher average ratings are more likely to have a greater number of ratings.
    Required columns:
        - avg_rating (float)
        - ratings_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    count_w = params.get("count_weight", 1.0)
    if not {"avg_rating", "ratings_count"}.issubset(df.columns):
        raise ValueError("avg_rating_rating_count_score requires avg_rating and ratings_count columns")
    score = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5) + count_w * np.log1p(df["ratings_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="avg_rating_rating_count_score")

def ebook_positive_rating_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 43: eBooks tend to have higher average ratings compared to physical books.
    Required columns:
        - is_ebook (bool/int 0/1)
        - avg_rating (float)
    """
    ebook_boost = params.get("ebook_boost", 1.1)
    rating_w = params.get("rating_weight", 1.0)
    if not {"is_ebook", "avg_rating"}.issubset(df.columns):
        raise ValueError("ebook_positive_rating_score requires is_ebook and avg_rating columns")
    base = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["is_ebook"].fillna(0) == 1, base * ebook_boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="ebook_positive_rating_score")

def publisher_reputation_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 44: Books published by known publishers receive higher ratings.
    Required columns:
        - publisher_name (str)
        - avg_rating (float)
    """
    rep_w = params.get("reputation_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"publisher_name", "avg_rating"}.issubset(df.columns):
        raise ValueError("publisher_reputation_rating requires publisher_name and avg_rating columns")
    pub_counts = df["publisher_name"].value_counts()
    score = rep_w * np.log1p(df["publisher_name"].map(pub_counts)) + rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="publisher_reputation_rating")

def rating_engagement_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 45: Higher book ratings correlate with more user engagement indicators.
    Required columns:
        - cnt (int)
        - mean_rating (float)
    """
    cnt_w = params.get("count_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"cnt", "mean_rating"}.issubset(df.columns):
        raise ValueError("rating_engagement_correlation requires cnt and mean_rating columns")
    score = cnt_w * np.log1p(df["cnt"].fillna(0)) + rating_w * (df["mean_rating"].fillna(df["mean_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="rating_engagement_correlation")

def series_vs_standalone_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 46: Books in series have higher average ratings than standalone books.
    Required columns:
        - series_name (str, can be null for standalone)
        - avg_rating (float)
    """
    series_boost = params.get("series_boost", 1.1)
    rating_w = params.get("rating_weight", 1.0)
    if not {"series_name", "avg_rating"}.issubset(df.columns):
        raise ValueError("series_vs_standalone_rating requires series_name and avg_rating columns")
    base = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["series_name"].notnull() & (df["series_name"] != ""), base * series_boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="series_vs_standalone_rating")

def translation_penalty_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 47: Translated books achieve lower ratings compared to original language publications.
    Required columns:
        - role (str)
        - avg_rating (float)
    """
    penalty = params.get("translation_penalty", 0.9)
    rating_w = params.get("rating_weight", 1.0)
    if not {"role", "avg_rating"}.issubset(df.columns):
        raise ValueError("translation_penalty_score requires role and avg_rating columns")
    base = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["role"].str.lower().fillna("").str.contains("translator"), base * penalty, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="translation_penalty_score")

def genre_diversity_engagement_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 48: Readers who engage with multiple genres display broader engagement metrics.
    Required columns:
        - genre (str)
        - n_ratings (int)
    """
    diversity_w = params.get("diversity_weight", 1.0)
    engagement_w = params.get("engagement_weight", 1.0)
    if not {"genre", "n_ratings"}.issubset(df.columns):
        raise ValueError("genre_diversity_engagement_score requires genre and n_ratings columns")
    genre_counts = df["genre"].value_counts()
    score = diversity_w * df["genre"].map(genre_counts) + engagement_w * np.log1p(df["n_ratings"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="genre_diversity_engagement_score")

def publisher_marketing_rating_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 49: Books published with more extensive marketing (e.g., large publisher backing) receive higher user ratings.
    Required columns:
        - publisher_name (str)
        - avg_rating (float)
    """
    marketing_boost = params.get("marketing_boost", 1.2)
    rating_w = params.get("rating_weight", 1.0)
    if not {"publisher_name", "avg_rating"}.issubset(df.columns):
        raise ValueError("publisher_marketing_rating_boost requires publisher_name and avg_rating columns")
    pub_counts = df["publisher_name"].value_counts()
    score = rating_w * (df["avg_rating"].fillna(df["avg_rating"].median()) / 5)
    score = np.where(df["publisher_name"].map(pub_counts) > 10, score * marketing_boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="publisher_marketing_rating_boost")


def detailed_review_rating_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 31: Users who leave reviews with more detailed text tend to provide higher ratings.
    Required columns:
        - review_text (str)
        - rating (float)
    """
    detail_w = params.get("detail_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"review_text", "rating"}.issubset(df.columns):
        raise ValueError("detailed_review_rating_boost requires review_text and rating columns")
    detail = df["review_text"].fillna("").str.len()
    score = detail_w * np.log1p(detail) + rating_w * (df["rating"].fillna(df["rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="detailed_review_rating_boost")

def wishlist_vs_bookclub_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 32: Books categorized as 'wish-list' tend to receive lower ratings than those in 'book-club' or 'ya' genres.
    Required columns:
        - genre (str)
        - average_rating (float)
    """
    wishlist_penalty = params.get("wishlist_penalty", 0.9)
    club_boost = params.get("club_boost", 1.1)
    rating_w = params.get("rating_weight", 1.0)
    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("wishlist_vs_bookclub_rating requires genre and average_rating columns")
    base = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = base.copy()
    score[df["genre"].str.lower().fillna("").str.contains("wish-list")] *= wishlist_penalty
    score[df["genre"].str.lower().fillna("").str.contains("book-club|ya")] *= club_boost
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="wishlist_vs_bookclub_rating")

def reader_engagement_positive_influence(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 33: Readers who rate more books tend to have a positive influence on their average ratings.
    Required columns:
        - user_id (str/int)
        - rating (float)
    """
    engagement_w = params.get("engagement_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"user_id", "rating"}.issubset(df.columns):
        raise ValueError("reader_engagement_positive_influence requires user_id and rating columns")
    user_counts = df.groupby("user_id")["rating"].count()
    user_avg = df.groupby("user_id")["rating"].mean()
    score = engagement_w * np.log1p(df["user_id"].map(user_counts)) + rating_w * (df["user_id"].map(user_avg) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="reader_engagement_positive_influence")

def avg_rating_ratings_count_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 34: Books with higher average ratings tend to receive more ratings.
    Required columns:
        - average_rating (float)
        - ratings_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    count_w = params.get("count_weight", 1.0)
    if not {"average_rating", "ratings_count"}.issubset(df.columns):
        raise ValueError("avg_rating_ratings_count_correlation requires average_rating and ratings_count columns")
    score = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5) + count_w * np.log1p(df["ratings_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="avg_rating_ratings_count_correlation")

def optimal_page_length_popularity(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 35: Books with a length of 400-450 pages are more popular.
    Required columns:
        - num_pages (int)
    """
    lower = params.get("lower", 400)
    upper = params.get("upper", 450)
    boost = params.get("boost", 1.2)
    page_w = params.get("page_weight", 1.0)
    if "num_pages" not in df.columns:
        raise ValueError("optimal_page_length_popularity requires num_pages column")
    base = page_w * (df["num_pages"].fillna(0) / df["num_pages"].max())
    mask = (df["num_pages"] >= lower) & (df["num_pages"] <= upper)
    score = np.where(mask, base * boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="optimal_page_length_popularity")

def outlier_popularity_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 36: Certain books have significantly higher ratings counts, indicating outlier popularity.
    Required columns:
        - ratings_count (int)
    """
    outlier_w = params.get("outlier_weight", 1.0)
    if "ratings_count" not in df.columns:
        raise ValueError("outlier_popularity_score requires ratings_count column")
    q3 = df["ratings_count"].quantile(0.75)
    outlier = df["ratings_count"] > q3
    score = outlier_w * outlier.astype(float)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="outlier_popularity_score")

def niche_audience_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 37: Books with lower average ratings may have niche audiences.
    Required columns:
        - average_rating (float)
        - ratings_count (int)
    """
    niche_w = params.get("niche_weight", 1.0)
    if not {"average_rating", "ratings_count"}.issubset(df.columns):
        raise ValueError("niche_audience_score requires average_rating and ratings_count columns")
    low_rating = df["average_rating"] < df["average_rating"].median()
    score = niche_w * low_rating.astype(float)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="niche_audience_score")

def mystery_suspense_genre_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 38: Books in the 'mystery-suspense' genre have higher average ratings than those in other genres.
    Required columns:
        - genre (str)
        - average_rating (float)
    """
    genre_boost = params.get("genre_boost", 1.2)
    rating_w = params.get("rating_weight", 1.0)
    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("mystery_suspense_genre_boost requires genre and average_rating columns")
    base = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = np.where(df["genre"].str.lower().fillna("").str.contains("mystery-suspense"), base * genre_boost, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="mystery_suspense_genre_boost")

def user_reading_volume_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 39: Users who read more books tend to give higher average ratings.
    Required columns:
        - user_id (str/int)
        - rating (float)
    """
    volume_w = params.get("volume_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    if not {"user_id", "rating"}.issubset(df.columns):
        raise ValueError("user_reading_volume_rating requires user_id and rating columns")
    user_counts = df.groupby("user_id")["rating"].count()
    user_avg = df.groupby("user_id")["rating"].mean()
    score = volume_w * np.log1p(df["user_id"].map(user_counts)) + rating_w * (df["user_id"].map(user_avg) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_reading_volume_rating")

def author_collaboration_success(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 40: Author collaborations are linked to an increase in shared readership and book success.
    Required columns:
        - author_id (str/int)
        - book_id (str/int)
    """
    collab_w = params.get("collab_weight", 1.0)
    if not {"author_id", "book_id"}.issubset(df.columns):
        raise ValueError("author_collaboration_success requires author_id and book_id columns")
    author_books = df.groupby("author_id")["book_id"].nunique()
    score = collab_w * np.log1p(df["author_id"].map(author_books))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="author_collaboration_success")


def page_count_rating_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 21: Books with more pages tend to receive a higher average rating.
    
    Required columns:
        - num_pages (int)
        - average_rating (float)
    """
    page_w = params.get("page_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    max_pages = params.get("max_pages", 1200)

    if not {"num_pages", "average_rating"}.issubset(df.columns):
        raise ValueError("page_count_rating_correlation requires num_pages & average_rating")

    pg_norm = df["num_pages"].fillna(0).clip(upper=max_pages) / max_pages
    score = page_w * pg_norm + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="page_count_rating_correlation")


def ebook_rating_penalty(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 22: Ebooks tend to have lower average ratings compared to physical books.
    
    Required columns:
        - is_ebook (bool/int 0/1)
        - average_rating (float)
    """
    ebook_penalty = params.get("ebook_penalty", 0.9)
    rating_w = params.get("rating_weight", 1.0)

    if not {"is_ebook", "average_rating"}.issubset(df.columns):
        raise ValueError("ebook_rating_penalty needs is_ebook & average_rating")

    base = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = np.where(df["is_ebook"].fillna(0) == 1, base * ebook_penalty, base)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="ebook_rating_penalty")


def genre_volume_rating_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 23: Genres with more books tend to have higher average ratings.
    
    Required columns:
        - genre (str)
        - average_rating (float)
    """
    volume_w = params.get("volume_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"genre", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_volume_rating_boost needs genre & average_rating")

    gstats = df.groupby("genre").agg(cnt=("genre", "size"), avg_r=("average_rating", "mean"))
    gscore = volume_w * np.log1p(gstats["cnt"]) + rating_w * (gstats["avg_r"] / 5)
    gscore = (gscore - gscore.min()) / (gscore.max() - gscore.min() + 1e-9)
    return pd.Series(df["genre"].map(gscore).fillna(0).values, index=df.index, name="genre_volume_rating_boost")


def user_activity_review_count(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 24: Users who engage with more books tend to provide more reviews.
    
    Required columns:
        - user_book_count (int)
        - review_count_user (int)
    """
    activity_w = params.get("activity_weight", 1.0)
    review_w = params.get("review_weight", 1.0)

    if not {"user_book_count", "review_count_user"}.issubset(df.columns):
        raise ValueError("user_activity_review_count requires user_book_count & review_count_user")

    score = activity_w * np.log1p(df["user_book_count"].fillna(0)) + review_w * np.log1p(df["review_count_user"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_activity_review_count")


def rating_review_volume_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 25: Books with higher average ratings tend to have more reviews.
    
    Required columns:
        - average_rating (float)
        - review_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    review_w = params.get("review_weight", 1.0)

    if not {"average_rating", "review_count"}.issubset(df.columns):
        raise ValueError("rating_review_volume_correlation requires average_rating & review_count")

    score = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5) + review_w * np.log1p(df["review_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="rating_review_volume_correlation")


def demographic_format_engagement(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 26: Different audience demographics engage differently with book formats.
    
    Required columns:
        - user_age_group (str)
        - format (str)
        - engagement (float/int)
    """
    demo_w = params.get("demo_weight", 1.0)
    format_w = params.get("format_weight", 1.0)

    cols = {"user_age_group", "format", "engagement"}
    if not cols.issubset(df.columns):
        raise ValueError("demographic_format_engagement missing required columns")

    demo_fmt_eng = df.groupby(["user_age_group", "format"]).agg(avg_e=("engagement", "mean"))
    demo_fmt_eng = (demo_fmt_eng - demo_fmt_eng.min()) / (demo_fmt_eng.max() - demo_fmt_eng.min() + 1e-9)

    score = df.apply(lambda row: demo_w * demo_fmt_eng.loc[(row["user_age_group"], row["format"])] if (row["user_age_group"], row["format"]) in demo_fmt_eng.index else 0, axis=1)
    return pd.Series(score, index=df.index, name="demographic_format_engagement")


def author_popularity_review_rate(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 27: Popular authors tend to have higher review rates for their books.
    
    Required columns:
        - author_id (identifier)
        - review_count (int)
        - ratings_count (int)
    """
    review_w = params.get("review_weight", 1.0)
    popularity_w = params.get("popularity_weight", 1.0)

    cols = {"author_id", "review_count", "ratings_count"}
    if not cols.issubset(df.columns):
        raise ValueError("author_popularity_review_rate missing columns")

    auth_stats = df.groupby("author_id").agg(rev_sum=("review_count", "sum"), rat_sum=("ratings_count", "sum"))
    auth_score = review_w * np.log1p(auth_stats["rev_sum"]) + popularity_w * np.log1p(auth_stats["rat_sum"])
    auth_score = (auth_score - auth_score.min()) / (auth_score.max() - auth_score.min() + 1e-9)
    return pd.Series(df["author_id"].map(auth_score).fillna(0).values, index=df.index, name="author_popularity_review_rate")


def review_sentiment_engagement_variance(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 28: The sentiments expressed in reviews vary significantly by user engagement.
    
    Required columns:
        - review_text (str)
        - engagement (float/int)
    """
    sentiment_w = params.get("sentiment_weight", 1.0)
    engagement_w = params.get("engagement_weight", 1.0)

    positive_words = ["good", "great", "excellent", "amazing", "love", "wonderful", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "hate", "boring", "worst", "poor"]

    if not {"review_text", "engagement"}.issubset(df.columns):
        raise ValueError("review_sentiment_engagement_variance requires review_text & engagement")

    text = df["review_text"].fillna("").str.lower()
    pos = text.str.count("|".join(positive_words))
    neg = text.str.count("|".join(negative_words))
    sentiment = (pos - neg) / (pos + neg + 1)

    score = sentiment_w * sentiment + engagement_w * (df["engagement"].fillna(df["engagement"].median()) / (df["engagement"].max() + 1e-9))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="review_sentiment_engagement_variance")


def format_availability_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 29: Books with higher average ratings tend to have more formats available.
    
    Required columns:
        - format_count (int)
        - average_rating (float)
    """
    fmt_w = params.get("format_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"format_count", "average_rating"}.issubset(df.columns):
        raise ValueError("format_availability_rating requires format_count & average_rating")

    score = fmt_w * np.log1p(df["format_count"].fillna(0)) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="format_availability_rating")


def genre_listing_diversity_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 30: Books listed in more genres receive higher ratings.
    
    Required columns:
        - genre_count (int)
        - average_rating (float)
    """
    genre_w = params.get("genre_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"genre_count", "average_rating"}.issubset(df.columns):
        raise ValueError("genre_listing_diversity_rating requires genre_count & average_rating")

    score = genre_w * np.log1p(df["genre_count"].fillna(0)) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="genre_listing_diversity_rating")

def description_quality_rating_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 11: Well-written / longer descriptions correlate with higher ratings.

    Required columns:
        - description: text description of the book (str)
        - average_rating: float rating 0-5
    """
    # Hyperparameters
    desc_weight = params.get("desc_weight", 1.0)
    length_weight = params.get("length_weight", 0.5)
    min_length = params.get("min_length", 100)
    rating_scale = params.get("rating_scale", 5.0)

    if not {"description", "average_rating"}.issubset(df.columns):
        raise ValueError("description_quality_rating_correlation requires description and average_rating columns")

    desc_len = df["description"].fillna("").str.len()
    quality = desc_weight * (df["average_rating"].fillna(df["average_rating"].median()) / rating_scale)
    length_component = length_weight * np.log1p(desc_len.clip(lower=min_length))

    score = quality + length_component
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="description_quality_rating_correlation")


def review_sentiment_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 12: Positive review sentiment predicts higher ratings.

    Required columns:
        - review_text (str)
        - rating (float)
    """
    pos_weight = params.get("pos_weight", 1.0)
    rating_weight = params.get("rating_weight", 0.5)

    if not {"review_text", "rating"}.issubset(df.columns):
        raise ValueError("review_sentiment_score requires review_text and rating columns")

    # Very light lexicon sentiment (counts of positive vs negative cues)
    positive_words = ["good", "great", "excellent", "amazing", "love", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "hate", "boring", "worst"]

    text = df["review_text"].fillna("").str.lower()
    pos_ct = text.str.count("|".join(positive_words))
    neg_ct = text.str.count("|".join(negative_words))
    sentiment = (pos_ct - neg_ct) / (pos_ct + neg_ct + 1)

    score = pos_weight * sentiment + rating_weight * df["rating"].fillna(df["rating"].median())
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="review_sentiment_score")


def user_interaction_engagement(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 13: Votes & comments reflect engagement → quality.

    Required columns:
        - n_votes (int)
        - n_comments (int)
    """
    votes_w = params.get("votes_weight", 1.0)
    comments_w = params.get("comments_weight", 1.0)
    boost = params.get("engagement_boost", 1.2)

    cols = {"n_votes", "n_comments"}
    if not cols.issubset(df.columns):
        raise ValueError("user_interaction_engagement requires n_votes and n_comments columns")

    votes = np.log1p(df["n_votes"].fillna(0))
    comments = np.log1p(df["n_comments"].fillna(0))
    score = votes_w * votes + comments_w * comments
    high_eng = (df["n_votes"].fillna(0) >= 10) | (df["n_comments"].fillna(0) >= 5)
    score = np.where(high_eng, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="user_interaction_engagement")


def publisher_diversity_quality(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 14: Publishers with diverse catalogues & quality yield better books.

    Required columns:
        - publisher_name (str)
        - average_rating (float)
    The DataFrame must be per book; function aggregates internally.
    """
    diversity_w = params.get("diversity_weight", 0.7)
    quality_w = params.get("quality_weight", 1.0)

    if not {"publisher_name", "average_rating"}.issubset(df.columns):
        raise ValueError("publisher_diversity_quality requires publisher_name and average_rating columns")

    pub_stats = df.groupby("publisher_name").agg(
        pub_count=("average_rating", "size"),
        pub_avg=("average_rating", "mean")
    )
    pub_score = quality_w * pub_stats["pub_avg"] + diversity_w * np.log1p(pub_stats["pub_count"])
    pub_score = (pub_score - pub_score.min()) / (pub_score.max() - pub_score.min() + 1e-9)
    score = df["publisher_name"].map(pub_score).fillna(0)
    return pd.Series(score.values, index=df.index, name="publisher_diversity_quality")


def publication_recency_impact(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 15: Recent high-rated books trend better.

    Required columns:
        - publication_year (int)
        - average_rating (float)
    """
    recency_w = params.get("recency_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)
    current_year = params.get("current_year", 2025)
    recent_years = params.get("recent_years", 5)
    boost = params.get("recency_boost", 1.3)

    cols = {"publication_year", "average_rating"}
    if not cols.issubset(df.columns):
        raise ValueError("publication_recency_impact requires publication_year and average_rating columns")

    years_old = current_year - df["publication_year"].fillna(current_year)
    recency = recency_w * (recent_years - years_old).clip(lower=0) / recent_years
    score = recency + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    recent_mask = years_old <= recent_years
    score = np.where(recent_mask, score * boost, score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="publication_recency_impact")


def format_preference_rating(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 16: Certain formats garner higher ratings.

    Required columns:
        - format (str)
        - average_rating (float)
    """
    format_w = params.get("format_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if not {"format", "average_rating"}.issubset(df.columns):
        raise ValueError("format_preference_rating requires format and average_rating columns")

    format_stats = df.groupby("format").agg(avg_r=("average_rating", "mean"), cnt=("format", "size"))
    fmt_score = format_w * np.log1p(format_stats["cnt"]) + rating_w * (format_stats["avg_r"] / 5)
    fmt_score = (fmt_score - fmt_score.min()) / (fmt_score.max() - fmt_score.min() + 1e-9)
    score = df["format"].map(fmt_score).fillna(0)
    return pd.Series(score.values, index=df.index, name="format_preference_rating")


def rating_review_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 17: Highly rated books receive more reviews.

    Required columns:
        - average_rating (float)
        - review_count (int)
    """
    rating_w = params.get("rating_weight", 1.0)
    review_w = params.get("review_weight", 1.0)

    if not {"average_rating", "review_count"}.issubset(df.columns):
        raise ValueError("rating_review_correlation requires average_rating and review_count columns")

    score = rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5) + review_w * np.log1p(df["review_count"].fillna(0))
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="rating_review_correlation")


def thematic_engagement_score(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 18: Presence of engaging themes in description ↑ ratings.

    Required columns:
        - description (str)
        - average_rating (float)
    """
    theme_w = params.get("theme_weight", 1.0)
    rating_w = params.get("rating_weight", 0.5)

    engaging_keywords = params.get("keywords", [
        "love", "mystery", "adventure", "magic", "family", "friendship", "war"
    ])

    if not {"description", "average_rating"}.issubset(df.columns):
        raise ValueError("thematic_engagement_score requires description and average_rating columns")

    desc = df["description"].fillna("").str.lower()
    theme_counts = sum(desc.str.count(k) for k in engaging_keywords)
    score = theme_w * np.log1p(theme_counts) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="thematic_engagement_score")


def author_collaboration_effect(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Hypothesis 19: Multi-author collaborations influence ratings.

    Required columns:
        - authors (str list or comma-separated str)
        - average_rating (float)
    """
    collab_w = params.get("collab_weight", 1.0)
    rating_w = params.get("rating_weight", 1.0)

    if "authors" not in df.columns or "average_rating" not in df.columns:
        raise ValueError("author_collaboration_effect requires authors and average_rating columns")

    author_count = df["authors"].fillna("").apply(lambda x: len(str(x).split("|")) if "|" in str(x) else len(str(x).split(",")))
    score = collab_w * np.log1p(author_count) + rating_w * (df["average_rating"].fillna(df["average_rating"].median()) / 5)
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return pd.Series(score, index=df.index, name="author_collaboration_effect")
