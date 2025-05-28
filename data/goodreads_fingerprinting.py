def generate_rating_pattern_fingerprints(min_ratings=5, max_users=None):
    """
    Generate rating pattern fingerprints for ALL users with ultra-efficient processing.
    Performs deep analysis of how users distribute their ratings across the 1-5 scale.

    Args:
        min_ratings: Minimum number of ratings a user must have (default: 5)
        max_users: Optional limit on number of users to analyze (default: None = all users)
    """
    import sqlite3
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    start_time = time.time()
    print("Starting ultra-efficient rating pattern analysis on full dataset...")

    # Connect to the database and optimize with indexes
    try:
        db_path = "data/goodreads.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure all needed indexes exist for maximum performance
        indexes = {
            "idx_reviews_user_id": "CREATE INDEX idx_reviews_user_id ON reviews(user_id)",
            "idx_reviews_rating": "CREATE INDEX idx_reviews_rating ON reviews(rating)",
            "idx_reviews_book_id": "CREATE INDEX idx_reviews_book_id ON reviews(book_id)",
        }

        for idx_name, idx_sql in indexes.items():
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='index' AND name='{idx_name}'"
            )
            if not cursor.fetchone():
                print(f"Creating index {idx_name} (one-time operation)...")
                cursor.execute(idx_sql)
                conn.commit()

        # Enable pragmas for faster queries
        cursor.execute("PRAGMA temp_store = MEMORY;")
        cursor.execute("PRAGMA cache_size = -50000;")  # 50MB cache
        cursor.execute("PRAGMA journal_mode = WAL;")
        conn.commit()
    except Exception as e:
        print(f"Failed to connect to database or create index: {e}")
        return None

    # Step 1: Find eligible users with minimum rating count (optimized query)
    print(f"Step 1/6: Finding users with at least {min_ratings} ratings...")
    try:
        # Using COUNT(*) with GROUP BY is more efficient than subqueries for this use case
        eligible_users_query = f"""
        SELECT user_id, COUNT(*) as rating_count
        FROM reviews
        WHERE rating IS NOT NULL
        GROUP BY user_id
        HAVING COUNT(*) >= {min_ratings}
        """

        if max_users:
            eligible_users_query += f" ORDER BY rating_count DESC LIMIT {max_users}"

        # Use pandas with efficient chunksize for large results
        eligible_users_df = pd.read_sql_query(eligible_users_query, conn)
        user_count = len(eligible_users_df)

        if user_count == 0:
            print("No users found with specified criteria.")
            conn.close()
            return None

        print(
            f"Found {user_count:,} eligible users with {eligible_users_df['rating_count'].sum():,} total ratings"
        )

        # Output user statistics
        print(f"  Min ratings per user: {eligible_users_df['rating_count'].min()}")
        print(f"  Max ratings per user: {eligible_users_df['rating_count'].max()}")
        print(
            f"  Median ratings per user: {eligible_users_df['rating_count'].median()}"
        )
        print(
            f"  Mean ratings per user: {eligible_users_df['rating_count'].mean():.1f}"
        )
    except Exception as e:
        print(f"Error finding eligible users: {e}")
        conn.close()
        return None

    # Step 2: Get rating distributions with a single efficient query
    print("Step 2/6: Retrieving rating patterns...")
    try:
        # Ultra-optimized query using Common Table Expression and direct GROUP BY
        patterns_query = f"""
        WITH eligible_users AS (
            SELECT user_id
            FROM reviews
            WHERE rating IS NOT NULL
            GROUP BY user_id
            HAVING COUNT(*) >= {min_ratings}
            {f"ORDER BY COUNT(*) DESC LIMIT {max_users}" if max_users else ""}
        )
        SELECT 
            r.user_id,
            r.rating,
            COUNT(*) as rating_count
        FROM reviews r
        JOIN eligible_users e ON r.user_id = e.user_id
        WHERE r.rating IS NOT NULL
        GROUP BY r.user_id, r.rating
        """

        print("Running optimized query...")
        start_query = time.time()
        rating_patterns_raw = pd.read_sql_query(patterns_query, conn)
        print(f"Query completed in {time.time() - start_query:.2f} seconds")

        # Process rating percentages efficiently with vectorized operations
        print("Calculating rating percentages...")
        user_totals = (
            rating_patterns_raw.groupby("user_id")["rating_count"].sum().reset_index()
        )
        user_totals.columns = ["user_id", "total_ratings"]

        # Merge with optimized parameters
        rating_patterns = pd.merge(
            rating_patterns_raw, user_totals, on="user_id", how="left"
        )
        rating_patterns["rating_percentage"] = (
            rating_patterns["rating_count"] / rating_patterns["total_ratings"]
        )

        unique_users = rating_patterns["user_id"].nunique()
        print(
            f"Retrieved {len(rating_patterns):,} rating patterns for {unique_users:,} users"
        )
    except Exception as e:
        print(f"Error retrieving rating patterns: {e}")
        conn.close()
        return None

    # Close database connection
    conn.close()

    # Step 3: Create user fingerprints (optimized pivot)
    print("Step 3/6: Creating fingerprints...")
    try:
        # Pivot to create user fingerprints (memory optimized)
        pattern_matrix = rating_patterns.pivot(
            index="user_id", columns="rating", values="rating_percentage"
        ).fillna(0)

        # Ensure all ratings 1-5 exist
        for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
            if rating not in pattern_matrix.columns:
                pattern_matrix[rating] = 0

        # Reorder columns for consistency
        pattern_matrix = pattern_matrix.reindex(columns=[1.0, 2.0, 3.0, 4.0, 5.0])

        print(f"Created fingerprints matrix with shape: {pattern_matrix.shape}")
    except Exception as e:
        print(f"Error creating fingerprints: {e}")
        return None

    # Step 4: Calculate comprehensive metrics
    print("Step 4/6: Calculating advanced metrics...")
    try:
        # Initialize results DataFrame
        metrics = pd.DataFrame(index=pattern_matrix.index)

        # Basic metrics
        metrics["primary_rating"] = pattern_matrix.idxmax(axis=1)
        metrics["primary_rating_pct"] = pattern_matrix.max(axis=1)

        # Entropy metric (rating diversity)
        epsilon = 1e-10
        log_vals = np.log(pattern_matrix.clip(lower=epsilon))
        entropy_values = -(pattern_matrix * log_vals).sum(axis=1)
        metrics["rating_entropy"] = entropy_values

        # Rating bias metrics
        metrics["high_rating_bias"] = pattern_matrix[4.0] + pattern_matrix[5.0]
        metrics["low_rating_bias"] = pattern_matrix[1.0] + pattern_matrix[2.0]
        metrics["rating_skew"] = (
            metrics["high_rating_bias"] - metrics["low_rating_bias"]
        )

        # New metrics
        # Gini coefficient (measure of inequality in rating distribution)
        def gini(x):
            # Assumes x is already sorted
            n = len(x)
            s = x.sum()
            if s == 0 or n <= 1:
                return 0
            # Normalized Gini calculation
            return (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * x / s)) / n

        # Apply gini to each user's rating distribution
        metrics["rating_gini"] = pattern_matrix.apply(
            lambda x: gini(x.sort_values()), axis=1
        )

        # Rating volatility (standard deviation of rating distribution)
        rating_values = np.array([1, 2, 3, 4, 5])

        def weighted_std(x):
            # Calculate weighted standard deviation of ratings
            weights = np.array([x[1.0], x[2.0], x[3.0], x[4.0], x[5.0]])
            if sum(weights) == 0:
                return 0
            avg = sum(weights * rating_values) / sum(weights)
            variance = sum(weights * (rating_values - avg) ** 2) / sum(weights)
            return np.sqrt(variance)

        metrics["rating_volatility"] = pattern_matrix.apply(weighted_std, axis=1)

        # Rating extremity (preference for extreme ratings vs middle ratings)
        metrics["rating_extremity"] = (
            pattern_matrix[1.0] + pattern_matrix[5.0]
        ) - pattern_matrix[3.0]

        # Add rating count from user_totals
        metrics = pd.merge(metrics.reset_index(), user_totals, on="user_id").set_index(
            "user_id"
        )

        # Categorize users by primary rating
        metrics["primary_rating_category"] = metrics["primary_rating"].map(
            {
                1.0: "Very Negative",
                2.0: "Negative",
                3.0: "Neutral",
                4.0: "Positive",
                5.0: "Very Positive",
            }
        )

        # Categorize rating entropy
        metrics["entropy_category"] = pd.qcut(
            metrics["rating_entropy"],
            q=5,
            labels=["Very Focused", "Focused", "Moderate", "Diverse", "Very Diverse"],
        )

        # Combine with pattern matrix for final fingerprints
        fingerprints = pd.merge(
            pattern_matrix.reset_index(), metrics.reset_index(), on="user_id"
        )

        print(
            f"Calculated {len(metrics.columns)} metrics for {len(fingerprints):,} users"
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

    # Step 5: Advanced clustering and dimensionality reduction
    print("Step 5/6: Performing advanced pattern analysis...")

    try:
        # Use only the rating pattern columns for analysis
        pattern_cols = [1.0, 2.0, 3.0, 4.0, 5.0]
        X = pattern_matrix[pattern_cols].values

        # Scale the data for better clustering
        scaler = StandardScaler()
        scaled_patterns = scaler.fit_transform(X)

        # PCA for dimensionality reduction and visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_patterns)
        print(f"PCA variance explained: {sum(pca.explained_variance_ratio_):.1%}")

        # Add PCA results to fingerprints
        fingerprints["pca_x"] = pca_result[:, 0]
        fingerprints["pca_y"] = pca_result[:, 1]

        # Get optimal number of clusters using simplified silhouette analysis
        # (simplified for speed - normally would use silhouette scores)
        k = 5  # Use fixed k=5 for efficiency

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_patterns)

        # Calculate silhouette score for the clustering
        from sklearn.metrics import silhouette_score

        try:
            sil_score = silhouette_score(scaled_patterns, clusters)
            print(f"K-means silhouette score: {sil_score:.3f}")
        except:
            print("Couldn't calculate silhouette score")

        # Add cluster information to results
        fingerprints["cluster"] = clusters

        # Alternative clustering with DBSCAN for outlier detection
        # Only if dataset is not too large
        if len(pattern_matrix) < 50000:
            try:
                print("Performing DBSCAN clustering for outlier detection...")
                dbscan = DBSCAN(eps=0.5, min_samples=10)
                dbscan_clusters = dbscan.fit_predict(scaled_patterns)

                # Add DBSCAN results
                fingerprints["dbscan_cluster"] = dbscan_clusters
                outliers = (dbscan_clusters == -1).sum()
                print(
                    f"DBSCAN found {outliers} outliers ({outliers / len(fingerprints):.1%})"
                )
            except Exception as e:
                print(f"Skipping DBSCAN: {e}")

        print("Advanced pattern analysis complete")
    except Exception as e:
        print(f"Error during advanced pattern analysis: {e}")
        print("Continuing without advanced pattern analysis")

    # Step 6: Generate comprehensive visualizations
    print("Step 6/6: Generating visualizations...")

    # 1. Primary rating distribution
    plt.figure(figsize=(12, 7))
    primary_counts = metrics["primary_rating"].value_counts().sort_index()
    ax = sns.barplot(x=primary_counts.index, y=primary_counts.values, palette="viridis")

    # Add percentage labels on bars
    total = primary_counts.sum()
    for i, count in enumerate(primary_counts):
        ax.text(i, count + 0.1, f"{count / total:.1%}", ha="center")

    plt.xlabel("Primary Rating")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Users by Primary Rating")
    plt.xticks(range(len(primary_counts)), [int(x) for x in primary_counts.index])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # 2. Rating entropy distribution with annotations
    plt.figure(figsize=(12, 7))
    sns.histplot(metrics["rating_entropy"], bins=50, kde=True)

    # Add vertical lines for quantiles
    quantiles = metrics["rating_entropy"].quantile([0.25, 0.5, 0.75])
    for q, qval in quantiles.items():
        plt.axvline(
            qval,
            color=["green", "red", "blue"][int(q * 4) - 1],
            linestyle="--",
            label=f"{int(q * 100)}%: {qval:.2f}",
        )

    plt.xlabel("Rating Entropy")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Rating Entropy (Higher = More Diverse Ratings)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # 3. Average rating patterns with standard error
    plt.figure(figsize=(12, 7))
    avg_pattern = pattern_matrix.mean()
    std_pattern = pattern_matrix.std() / np.sqrt(len(pattern_matrix))  # Standard error

    x = np.arange(len(avg_pattern))
    plt.bar(x, avg_pattern, yerr=std_pattern, capsize=10, color="skyblue", alpha=0.7)

    # Add percentage labels on bars
    for i, v in enumerate(avg_pattern):
        plt.text(i, v + 0.01, f"{v:.1%}", ha="center")

    plt.xlabel("Rating")
    plt.ylabel("Average Percentage")
    plt.title("Average Rating Distribution Across All Users")
    plt.xticks(x, [int(x) for x in avg_pattern.index])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # 4. Rating skew distribution
    plt.figure(figsize=(12, 7))
    sns.histplot(metrics["rating_skew"], bins=50, kde=True)
    plt.axvline(0, color="red", linestyle="--", label="Neutral")
    plt.axvline(
        metrics["rating_skew"].mean(),
        color="green",
        linestyle="-",
        label=f"Mean: {metrics['rating_skew'].mean():.2f}",
    )
    plt.xlabel("Rating Skew (High - Low Ratings)")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Rating Skew (Positive = More High Ratings)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # 5. New plot: Rating patterns by cluster
    plt.figure(figsize=(15, 10))

    # Get average pattern for each cluster
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_, columns=pattern_matrix.columns
    )

    # Plot each cluster's average pattern
    for i in range(k):
        plt.subplot(2, 3, i + 1)
        cluster_size = sum(fingerprints["cluster"] == i)
        plt.bar(range(1, 6), cluster_centers.iloc[i], color=f"C{i}", alpha=0.7)
        plt.title(
            f"Cluster {i + 1} (n={cluster_size}, {cluster_size / len(fingerprints):.1%})"
        )
        plt.xlabel("Rating")
        plt.ylabel("Percentage")
        plt.ylim(0, max(1.0, cluster_centers.values.max() + 0.1))
        plt.xticks(range(1, 6))
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle("Rating Patterns by Cluster", fontsize=16, y=1.02)
    plt.show()

    # 6. New plot: PCA visualization of rating patterns
    plt.figure(figsize=(12, 10))

    # Create colormap based on primary rating
    plt.scatter(
        fingerprints["pca_x"],
        fingerprints["pca_y"],
        c=fingerprints["primary_rating"],
        cmap="viridis",
        alpha=0.5,
        s=5,
    )

    plt.colorbar(label="Primary Rating")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("PCA of User Rating Patterns")
    plt.grid(alpha=0.3)
    plt.show()

    # 7. New plot: Rating Gini coefficient distribution
    plt.figure(figsize=(12, 7))
    sns.histplot(metrics["rating_gini"], bins=50, kde=True)
    plt.axvline(
        metrics["rating_gini"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {metrics['rating_gini'].mean():.3f}",
    )
    plt.axvline(
        metrics["rating_gini"].median(),
        color="green",
        linestyle="-",
        label=f"Median: {metrics['rating_gini'].median():.3f}",
    )
    plt.xlabel("Rating Gini Coefficient")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Rating Inequality (Higher = More Unequal Distribution)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # 8. New plot: Rating Extremity vs. Entropy
    plt.figure(figsize=(12, 7))
    plt.scatter(metrics["rating_extremity"], metrics["rating_entropy"], alpha=0.3, s=5)
    plt.xlabel("Rating Extremity (Extreme - Middle)")
    plt.ylabel("Rating Entropy")
    plt.title("Relationship Between Rating Extremity and Diversity")
    plt.grid(True, alpha=0.3)

    # Add a best fit line
    try:
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            metrics["rating_extremity"], metrics["rating_entropy"]
        )
        x = np.linspace(
            metrics["rating_extremity"].min(), metrics["rating_extremity"].max(), 100
        )
        plt.plot(
            x,
            intercept + slope * x,
            "r-",
            label=f"r = {r_value:.2f}, p = {p_value:.3g}",
        )
        plt.legend()
    except:
        pass

    plt.show()

    # Final timing info
    print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    print(f"Total number of users analyzed: {len(fingerprints):,}")

    return fingerprints


def generate_rating_bias_fingerprints(min_ratings=5):
    """
    Calculate user rating bias fingerprints for all users with optimization.
    Uses the entire dataset but with efficient queries.

    Args:
        min_ratings: Minimum number of ratings a user must have (default: 5)
    """
    import sqlite3
    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    start_time = time.time()
    print("Starting rating bias analysis on full dataset...")

    # Connect to the database
    try:
        db_path = "data/goodreads.db"
        conn = sqlite3.connect(db_path)

        # First check if index exists on user_id, if not create it
        # This will dramatically speed up queries that filter by user_id
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_reviews_user_id'"
        )
        if not cursor.fetchone():
            print(
                "Creating index on user_id (one-time operation, may take a minute)..."
            )
            cursor.execute("CREATE INDEX idx_reviews_user_id ON reviews(user_id)")
            conn.commit()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_reviews_rating'"
        )
        if not cursor.fetchone():
            print("Creating index on rating (one-time operation, may take a minute)...")
            cursor.execute("CREATE INDEX idx_reviews_rating ON reviews(rating)")
            conn.commit()
    except Exception as e:
        print(f"Failed to connect to database or create index: {e}")
        return None

    # Step 1: Calculate global average rating
    print("Step 1/4: Calculating global average...")
    try:
        global_avg_query = """
        SELECT AVG(rating) FROM reviews WHERE rating IS NOT NULL
        """
        global_avg = pd.read_sql_query(global_avg_query, conn).iloc[0, 0]
        print(f"Global average rating: {global_avg:.2f}")
    except Exception as e:
        print(f"Error calculating global average: {e}")
        conn.close()
        return None

    # Step 2: Find eligible users (with min_ratings)
    print(f"Step 2/4: Finding users with at least {min_ratings} ratings...")
    try:
        eligible_users_query = f"""
        SELECT user_id, COUNT(*) as rating_count
        FROM reviews
        WHERE rating IS NOT NULL
        GROUP BY user_id
        HAVING COUNT(*) >= {min_ratings}
        """
        eligible_users_df = pd.read_sql_query(eligible_users_query, conn)
        user_count = len(eligible_users_df)

        if user_count == 0:
            print("No users found with specified criteria.")
            conn.close()
            return None

        print(f"Found {user_count} eligible users")
    except Exception as e:
        print(f"Error finding eligible users: {e}")
        conn.close()
        return None

    # Step 3: Calculate user bias metrics in chunks
    # Process in chunks to avoid memory issues
    print("Step 3/4: Calculating bias metrics...")

    # Optimize by using a single efficient query for all metrics
    try:
        # This single query calculates all necessary metrics in one go
        # significantly reducing the processing time
        bias_query = f"""
        WITH user_ratings AS (
            SELECT 
                user_id,
                AVG(rating) as avg_rating,
                COUNT(*) as rating_count,
                MIN(rating) as min_rating,
                MAX(rating) as max_rating
            FROM reviews
            WHERE rating IS NOT NULL
            GROUP BY user_id
            HAVING COUNT(*) >= {min_ratings}
        )
        SELECT 
            u.*,
            (u.avg_rating - {global_avg}) as rating_bias
        FROM user_ratings u
        """

        # Stream results to avoid memory issues
        print("Running optimized query (this may take a minute)...")
        bias_profiles = pd.read_sql_query(bias_query, conn)
        print(f"Retrieved bias data for {len(bias_profiles)} users")
    except Exception as e:
        print(f"Error calculating user bias metrics: {e}")
        conn.close()
        return None

    # Close connection
    conn.close()

    # Step 4: Post-processing and visualizations
    print("Step 4/4: Post-processing and creating visualizations...")

    # Rename columns for clarity
    bias_profiles = bias_profiles.rename(
        columns={"avg_rating": "user_avg_rating", "rating_bias": "avg_bias"}
    )

    # Add bias categories
    bias_profiles["bias_category"] = pd.cut(
        bias_profiles["avg_bias"],
        bins=[-float("inf"), -1.0, -0.5, 0.5, 1.0, float("inf")],
        labels=["Very Critical", "Critical", "Neutral", "Generous", "Very Generous"],
    )

    # Calculate additional derived metrics
    bias_profiles["rating_diversity"] = (
        bias_profiles["max_rating"] - bias_profiles["min_rating"]
    )

    # Generate visualizations

    # Distribution of average bias
    plt.figure(figsize=(12, 6))
    plt.hist(bias_profiles["avg_bias"], bins=50, color="skyblue", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", label="Global Average")
    plt.axvline(
        bias_profiles["avg_bias"].mean(),
        color="green",
        linestyle="-",
        label=f"Mean Bias: {bias_profiles['avg_bias'].mean():.2f}",
    )
    plt.xlabel("Rating Bias")
    plt.ylabel("Number of Users")
    plt.title("Distribution of User Rating Bias")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Distribution of bias categories
    plt.figure(figsize=(12, 6))
    sns.countplot(x="bias_category", data=bias_profiles, palette="viridis")
    plt.xlabel("Bias Category")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Users by Bias Category")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Relationship between number of ratings and bias
    plt.figure(figsize=(12, 6))
    plt.scatter(
        bias_profiles["rating_count"],
        bias_profiles["avg_bias"],
        alpha=0.3,
        s=5,
        color="blue",
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Number of Ratings (log scale)")
    plt.ylabel("Rating Bias")
    plt.title("Relationship Between Number of Ratings and Rating Bias")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Relationship between average rating and bias
    plt.figure(figsize=(12, 6))
    plt.scatter(
        bias_profiles["user_avg_rating"],
        bias_profiles["avg_bias"],
        alpha=0.3,
        s=5,
        color="green",
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("User Average Rating")
    plt.ylabel("Rating Bias")
    plt.title("Relationship Between Average Rating and Rating Bias")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Rating diversity distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(x="rating_diversity", data=bias_profiles, palette="plasma")
    plt.xlabel("Rating Diversity (max - min rating)")
    plt.ylabel("Number of Users")
    plt.title("Distribution of Rating Diversity")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    print(f"Total number of users analyzed: {len(bias_profiles)}")

    return bias_profiles


# Helper function to connect to the SQLite database
def connect_to_db():
    import sqlite3

    db_path = "data/goodreads.db"
    return sqlite3.connect(db_path)


# Helper function to execute SQL queries and return results as DataFrame
def sql_query(query):
    import pandas as pd

    conn = connect_to_db()
    try:
        result = pd.read_sql_query(query, conn)
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
