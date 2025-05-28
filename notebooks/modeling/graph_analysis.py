# File: user_archetype_analysis.py
"""
Comprehensive User Archetype Analysis for Book Recommendations
Builds user archetypes based on reading patterns, rating behavior, and book metadata
Provides visualizations to identify clusters and common traits
"""

import sqlite3
import time
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pyo.init_notebook_mode(connected=True)


def connect_to_db():
    """Connect to the SQLite database"""
    db_path = "data/goodreads.db"
    return sqlite3.connect(db_path)


def build_user_book_graph(
    min_ratings_per_user=10, min_readers_per_book=5, max_users=10000
):
    """
    Build a bipartite graph of users and books for network analysis

    Args:
        min_ratings_per_user: Minimum ratings a user must have to be included
        min_readers_per_book: Minimum readers a book must have to be included
        max_users: Maximum number of users to include (for performance)

    Returns:
        G: NetworkX graph object
        user_df: DataFrame with user metadata
        book_df: DataFrame with book metadata
    """
    print(
        f"Building user-book graph (min_ratings={min_ratings_per_user}, min_readers={min_readers_per_book})..."
    )
    start_time = time.time()

    conn = connect_to_db()

    # Get active users with sufficient ratings
    user_query = f"""
    SELECT user_id, COUNT(*) as rating_count
    FROM reviews
    WHERE rating IS NOT NULL
    GROUP BY user_id
    HAVING COUNT(*) >= {min_ratings_per_user}
    ORDER BY rating_count DESC
    LIMIT {max_users}
    """
    users_df = pd.read_sql_query(user_query, conn)
    user_list = tuple(users_df["user_id"].tolist())

    # Get books with sufficient readers
    book_query = f"""
    SELECT book_id, COUNT(DISTINCT user_id) as reader_count
    FROM reviews
    WHERE user_id IN {user_list}
    AND rating IS NOT NULL
    GROUP BY book_id
    HAVING COUNT(DISTINCT user_id) >= {min_readers_per_book}
    """
    books_df = pd.read_sql_query(book_query, conn)
    book_list = tuple(books_df["book_id"].tolist())

    # Get ratings for the filtered users and books
    rating_query = f"""
    SELECT r.user_id, r.book_id, r.rating, r.text
    FROM reviews r
    WHERE r.user_id IN {user_list}
    AND r.book_id IN {book_list}
    AND r.rating IS NOT NULL
    """
    ratings_df = pd.read_sql_query(rating_query, conn)

    # Get book metadata
    book_meta_query = f"""
    SELECT b.book_id, b.title, b.author, b.average_rating, 
           b.ratings_count, b.language, b.num_pages
    FROM books b
    WHERE b.book_id IN {book_list}
    """
    book_meta_df = pd.read_sql_query(book_meta_query, conn)

    # Join book metadata with books_df
    book_df = pd.merge(books_df, book_meta_df, on="book_id")

    # Calculate user rating fingerprints
    user_fingerprints = calculate_user_fingerprints(ratings_df)

    # Merge fingerprints with users_df
    user_df = pd.merge(users_df, user_fingerprints, on="user_id")

    # Build the graph
    G = nx.Graph()

    # Add user nodes
    for _, user in user_df.iterrows():
        G.add_node(
            f"u_{user['user_id']}",
            type="user",
            user_id=user["user_id"],
            rating_count=user["rating_count"],
            avg_rating=user.get("avg_rating", 0),
            rating_entropy=user.get("rating_entropy", 0),
            rating_skew=user.get("rating_skew", 0),
        )

    # Add book nodes
    for _, book in book_df.iterrows():
        G.add_node(
            f"b_{book['book_id']}",
            type="book",
            book_id=book["book_id"],
            title=book["title"],
            author=book["author"],
            reader_count=book["reader_count"],
            avg_rating=book["average_rating"],
        )

    # Add edges (ratings)
    for _, rating in ratings_df.iterrows():
        G.add_edge(
            f"u_{rating['user_id']}",
            f"b_{rating['book_id']}",
            rating=rating["rating"],
            has_review=not pd.isna(rating["text"]) and len(str(rating["text"])) > 0,
        )

    print(f"Graph built in {time.time() - start_time:.2f} seconds:")
    print(f"  {len(user_df)} users, {len(book_df)} books, {len(ratings_df)} ratings")
    print(f"  {nx.number_of_nodes(G)} nodes, {nx.number_of_edges(G)} edges")

    conn.close()
    return G, user_df, book_df


def calculate_user_fingerprints(ratings_df):
    """Calculate rating fingerprints for users from ratings dataframe"""
    # Create rating pattern matrix
    pattern_df = ratings_df.groupby(["user_id", "rating"]).size().unstack(fill_value=0)

    # Ensure all ratings 1-5 exist
    for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
        if rating not in pattern_df.columns:
            pattern_df[rating] = 0

    # Normalize to percentages
    total_ratings = pattern_df.sum(axis=1)
    pattern_norm = pattern_df.div(total_ratings, axis=0)

    # Calculate metrics
    metrics = pd.DataFrame(index=pattern_df.index)

    # Average rating
    metrics["avg_rating"] = (
        pattern_df[1.0] * 1
        + pattern_df[2.0] * 2
        + pattern_df[3.0] * 3
        + pattern_df[4.0] * 4
        + pattern_df[5.0] * 5
    ) / total_ratings

    # Primary rating
    metrics["primary_rating"] = pattern_norm.idxmax(axis=1)
    metrics["primary_rating_pct"] = pattern_norm.max(axis=1)

    # Rating entropy (diversity)
    epsilon = 1e-10
    log_vals = np.log(pattern_norm.clip(lower=epsilon))
    metrics["rating_entropy"] = -(pattern_norm * log_vals).sum(axis=1)

    # Rating skew (positive vs negative bias)
    metrics["high_rating_bias"] = pattern_norm[4.0] + pattern_norm[5.0]
    metrics["low_rating_bias"] = pattern_norm[1.0] + pattern_norm[2.0]
    metrics["rating_skew"] = metrics["high_rating_bias"] - metrics["low_rating_bias"]

    # Rating extremity (preference for extreme vs middle ratings)
    metrics["rating_extremity"] = (
        pattern_norm[1.0] + pattern_norm[5.0]
    ) - pattern_norm[3.0]

    # Add total rating count
    metrics["total_ratings"] = total_ratings

    return metrics.reset_index()


def identify_user_archetypes(user_df, n_clusters=5):
    """
    Identify user archetypes using clustering on rating behavior

    Args:
        user_df: DataFrame with user fingerprints
        n_clusters: Number of clusters to create

    Returns:
        user_df: Updated with cluster assignments
        cluster_profiles: Archetype descriptions
    """
    print(f"Identifying {n_clusters} user archetypes...")

    # Select features for clustering
    features = [
        "avg_rating",
        "rating_entropy",
        "rating_skew",
        "rating_extremity",
        "primary_rating_pct",
    ]

    # Prepare data
    X = user_df[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster to user dataframe
    user_df["archetype"] = clusters

    # Create archetype profiles
    cluster_profiles = []

    # Get cluster centers in original scale
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    # For each cluster, create a profile
    for i in range(n_clusters):
        cluster_size = sum(user_df["archetype"] == i)
        cluster_pct = cluster_size / len(user_df) * 100

        # Get center values
        center = {feature: value for feature, value in zip(features, centers[i])}

        # Determine descriptive names based on center values
        if center["rating_skew"] > 0.5:
            if center["rating_entropy"] < 0.8:
                name = "Positive Enthusiast"
                desc = "Rates most books highly with little variation"
            else:
                name = "Generous Discriminator"
                desc = "Generally positive but shows meaningful distinctions"
        elif center["rating_skew"] < -0.3:
            if center["rating_extremity"] > 0.3:
                name = "Harsh Critic"
                desc = "Leans negative with occasional strong reactions"
            else:
                name = "Selective Reader"
                desc = "Consistently critical with narrow preferences"
        elif center["rating_entropy"] > 1.2:
            name = "Balanced Evaluator"
            desc = "Uses the full rating scale with minimal bias"
        elif center["rating_extremity"] > 0.4:
            name = "Polarized Reader"
            desc = "Tends toward love-it-or-hate-it ratings"
        else:
            name = "Middle-ground Reader"
            desc = "Avoids extremes, prefers moderate ratings"

        # Create profile
        profile = {
            "archetype_id": i,
            "name": name,
            "description": desc,
            "size": cluster_size,
            "percentage": cluster_pct,
            "avg_rating": center["avg_rating"],
            "rating_entropy": center["rating_entropy"],
            "rating_skew": center["rating_skew"],
            "rating_extremity": center["rating_extremity"],
            "primary_rating_pct": center["primary_rating_pct"],
        }

        cluster_profiles.append(profile)

    # Convert to DataFrame
    cluster_profiles = pd.DataFrame(cluster_profiles)

    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, clusters)
    print(f"Silhouette score: {silhouette:.3f}")

    return user_df, cluster_profiles


def find_similar_users(G, user_id, similarity_metric="jaccard", n=10):
    """
    Find users with similar reading patterns

    Args:
        G: NetworkX graph
        user_id: User ID to find similar users for
        similarity_metric: 'jaccard' or 'overlap'
        n: Number of similar users to return

    Returns:
        similar_users: DataFrame with similar users and similarity scores
    """
    user_node = f"u_{user_id}"

    if user_node not in G:
        return pd.DataFrame(columns=["user_id", "similarity", "common_books"])

    # Get books read by target user
    target_books = set(
        [neighbor for neighbor in G.neighbors(user_node) if neighbor.startswith("b_")]
    )

    similarities = []

    # Get all user nodes
    user_nodes = [
        node for node in G.nodes() if node.startswith("u_") and node != user_node
    ]

    for user in user_nodes:
        # Get books read by this user
        user_books = set(
            [neighbor for neighbor in G.neighbors(user) if neighbor.startswith("b_")]
        )

        # Calculate similarity
        common = len(target_books.intersection(user_books))

        if common > 0:
            if similarity_metric == "jaccard":
                sim = common / len(target_books.union(user_books))
            else:  # overlap
                sim = common / min(len(target_books), len(user_books))

            similarities.append(
                {
                    "user_id": user.replace("u_", ""),
                    "similarity": sim,
                    "common_books": common,
                }
            )

    # Convert to DataFrame and sort
    similar_df = pd.DataFrame(similarities)
    if len(similar_df) > 0:
        similar_df = similar_df.sort_values("similarity", ascending=False).head(n)

    return similar_df


def evaluate_tsne_embedding(X_high, X_low, y=None):
    """
    Evaluate t-SNE embedding quality using multiple metrics

    Args:
        X_high: Original high-dimensional data
        X_low: Low-dimensional embedding
        y: Optional class labels for supervised metrics

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {}

    # Calculate distances
    D_high = squareform(pdist(X_high))
    D_low = squareform(pdist(X_low))

    # Trustworthiness (local structure preservation)
    k = min(20, len(X_high) - 1)  # Number of neighbors to consider
    trust = 0
    for i in range(len(X_high)):
        # Get k nearest neighbors in high and low dimensions
        high_neighbors = np.argsort(D_high[i])[1 : k + 1]
        low_neighbors = np.argsort(D_low[i])[1 : k + 1]

        # Count preserved neighbors
        preserved = len(set(high_neighbors) & set(low_neighbors))
        trust += preserved / k
    metrics["trustworthiness"] = trust / len(X_high)

    # Continuity (global structure preservation)
    cont = 0
    for i in range(len(X_high)):
        high_neighbors = np.argsort(D_high[i])[1 : k + 1]
        low_neighbors = np.argsort(D_low[i])[1 : k + 1]
        preserved = len(set(high_neighbors) & set(low_neighbors))
        cont += preserved / k
    metrics["continuity"] = cont / len(X_high)

    # Stress (distance preservation)
    metrics["stress"] = np.sum((D_high - D_low) ** 2) / np.sum(D_high**2)

    # KNN accuracy if labels are available
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_low, y, test_size=0.2, random_state=42
        )
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        metrics["knn_accuracy"] = accuracy_score(y_test, y_pred)

    return metrics


def tune_tsne_parameters(
    X_scaled,
    y=None,
    perplexity_range=[5, 10, 30, 50, 100],
    learning_rate_range=[10, 50, 100, 200, 500],
    n_iter_range=[250, 500, 1000, 2000],
):
    """
    Tune t-SNE parameters using proper train-validation-test splits

    Args:
        X_scaled: Scaled feature matrix
        y: Optional class labels
        perplexity_range: Range of perplexity values to try
        learning_rate_range: Range of learning rates to try
        n_iter_range: Range of iteration counts to try

    Returns:
        best_params: Dictionary of best parameters
        results_df: DataFrame with all results
    """
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    results = []

    # Grid search over parameter combinations
    for perplexity, learning_rate, n_iter in product(
        perplexity_range, learning_rate_range, n_iter_range
    ):
        try:
            # Run t-SNE on training data
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42,
            )
            X_train_tsne = tsne.fit_transform(X_train)

            # Transform validation data
            X_val_tsne = tsne.fit_transform(X_val)

            # Evaluate on validation set
            metrics = evaluate_tsne_embedding(X_val, X_val_tsne, y_val)

            results.append(
                {
                    "perplexity": perplexity,
                    "learning_rate": learning_rate,
                    "n_iter": n_iter,
                    **metrics,
                }
            )

        except Exception as e:
            print(
                f"Failed for params {perplexity}, {learning_rate}, {n_iter}: {str(e)}"
            )
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find best parameters based on weighted combination of metrics
    weights = {
        "trustworthiness": 0.3,
        "continuity": 0.3,
        "stress": -0.2,  # Negative because lower is better
        "knn_accuracy": 0.2 if y is not None else 0,
    }

    # Normalize metrics
    for metric in weights.keys():
        if metric in results_df.columns:
            results_df[f"{metric}_norm"] = (
                results_df[metric] - results_df[metric].min()
            ) / (results_df[metric].max() - results_df[metric].min())

    # Calculate weighted score
    results_df["score"] = sum(
        results_df[f"{metric}_norm"] * weight
        for metric, weight in weights.items()
        if f"{metric}_norm" in results_df.columns
    )

    # Find best parameters
    best_idx = results_df["score"].idxmax()
    best_params = results_df.loc[best_idx].to_dict()

    # Final evaluation on test set
    tsne = TSNE(
        n_components=2,
        perplexity=best_params["perplexity"],
        learning_rate=best_params["learning_rate"],
        n_iter=best_params["n_iter"],
        random_state=42,
    )
    X_test_tsne = tsne.fit_transform(X_test)
    test_metrics = evaluate_tsne_embedding(X_test, X_test_tsne, y_test)

    print("\nFinal Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    return best_params, results_df


def create_archetype_visualization(user_df, cluster_profiles):
    """Create interactive visualization of user archetypes"""
    # Prepare data for dimensionality reduction
    features = [
        "avg_rating",
        "rating_entropy",
        "rating_skew",
        "rating_extremity",
        "primary_rating_pct",
    ]

    X = user_df[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tune t-SNE parameters
    best_params, tuning_results = tune_tsne_parameters(X_scaled)
    print("Best t-SNE parameters:", best_params)

    # Run t-SNE with best parameters
    tsne = TSNE(
        n_components=2,
        perplexity=best_params["perplexity"],
        learning_rate=best_params["learning_rate"],
        n_iter=best_params["n_iter"],
        random_state=42,
    )
    tsne_result = tsne.fit_transform(X_scaled)

    # Create DataFrames for plotting
    tsne_df = pd.DataFrame(data=tsne_result, columns=["TSNE1", "TSNE2"])
    tsne_df["user_id"] = user_df["user_id"].values
    tsne_df["archetype"] = user_df["archetype"].values
    tsne_df["avg_rating"] = user_df["avg_rating"].values
    tsne_df["rating_count"] = user_df["rating_count"].values

    # Add archetype names
    archetype_map = {
        row["archetype_id"]: row["name"] for _, row in cluster_profiles.iterrows()
    }
    tsne_df["archetype_name"] = tsne_df["archetype"].map(archetype_map)

    # Create t-SNE plot
    fig_tsne = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="archetype_name",
        hover_data=["user_id", "avg_rating", "rating_count"],
        title="User Archetypes - t-SNE",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    # Display variance explained by t-SNE
    print(
        f"t-SNE variance explained: {tsne.explained_variance_ratio_[0]:.2%} (TSNE1), "
        f"{tsne.explained_variance_ratio_[1]:.2%} (TSNE2), "
        f"Total: {sum(tsne.explained_variance_ratio_):.2%}"
    )

    # Display t-SNE components matrix
    tsne_components = pd.DataFrame(
        data=tsne.components_.T,  # Transpose to get features as rows
        columns=["TSNE1", "TSNE2"],
        index=features,
    )

    # Display the matrix with feature contributions to each component
    print("\nt-SNE Components Matrix:")
    print(tsne_components)

    # Create a heatmap visualization of the components
    plt.figure(figsize=(10, 6))
    sns.heatmap(tsne_components, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Contributions to t-SNE Components")
    plt.tight_layout()

    # Create archetype profiles radar chart
    fig_radar = go.Figure()

    # Standardize metrics for radar
    metrics = [
        "avg_rating",
        "rating_entropy",
        "rating_skew",
        "rating_extremity",
        "primary_rating_pct",
    ]

    # Scale metrics from 0-1 for radar chart
    min_vals = cluster_profiles[metrics].min()
    max_vals = cluster_profiles[metrics].max()

    # Create radar data
    for _, profile in cluster_profiles.iterrows():
        scaled_metrics = [
            (profile[m] - min_vals[m]) / (max_vals[m] - min_vals[m]) for m in metrics
        ]

        fig_radar.add_trace(
            go.Scatterpolar(
                r=scaled_metrics + [scaled_metrics[0]],  # Close the polygon
                theta=metrics + [metrics[0]],  # Close the polygon
                fill="toself",
                name=profile["name"],
            )
        )

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Archetype Profiles",
    )

    return fig_tsne, fig_radar, tsne_df


def create_user_book_network_visualization(G, user_df, book_df, max_nodes=300):
    """Create visualization of user-book network focusing on archetypes"""
    # Create a smaller subgraph for visualization
    # Select popular books and their readers
    popular_books = sorted(
        [n for n in G.nodes() if n.startswith("b_")],
        key=lambda x: G.degree(x),
        reverse=True,
    )[:50]

    # Get users who read these books
    users = set()
    for book in popular_books:
        readers = [n for n in G.neighbors(book) if n.startswith("u_")]
        users.update(readers[:10])  # Take top 10 readers per book

    # Limit number of users to make viz manageable
    users = list(users)[: max_nodes - len(popular_books)]

    # Create subgraph
    nodes = users + popular_books
    sub_G = G.subgraph(nodes)

    # Prepare for visualization
    pos = nx.spring_layout(sub_G, k=0.3, iterations=50)

    # Node lists by type
    user_nodes = [n for n in sub_G.nodes() if n.startswith("u_")]
    book_nodes = [n for n in sub_G.nodes() if n.startswith("b_")]

    # Get archetype info
    user_id_to_archetype = {
        f"u_{row['user_id']}": row["archetype"] for _, row in user_df.iterrows()
    }

    # Node colors by archetype
    node_colors = []
    for node in user_nodes:
        archetype = user_id_to_archetype.get(node, 0)
        node_colors.append(f"C{archetype}")

    # Create figure
    plt.figure(figsize=(12, 12))

    # Draw book nodes
    nx.draw_networkx_nodes(
        sub_G,
        pos,
        nodelist=book_nodes,
        node_color="lightgray",
        node_shape="s",
        node_size=100,
        alpha=0.8,
    )

    # Draw user nodes colored by archetype
    nx.draw_networkx_nodes(
        sub_G,
        pos,
        nodelist=user_nodes,
        node_color=node_colors,
        node_shape="o",
        node_size=80,
        alpha=0.7,
    )

    # Draw edges
    nx.draw_networkx_edges(sub_G, pos, width=0.5, alpha=0.3)

    # Add labels for top books
    top_book_nodes = sorted(book_nodes, key=lambda x: G.degree(x), reverse=True)[:10]
    book_labels = {n: G.nodes[n]["title"][:15] + "..." for n in top_book_nodes}
    nx.draw_networkx_labels(sub_G, pos, labels=book_labels, font_size=8)

    plt.title("User-Book Network: Users colored by archetype, squares are books")
    plt.axis("off")

    return plt.gcf()  # Return the figure


def analyze_archetype_preferences(G, user_df, book_df):
    """Analyze which books are preferred by different archetypes"""
    # Map each user to their archetype
    user_archetypes = {
        row["user_id"]: row["archetype"] for _, row in user_df.iterrows()
    }

    # Collect ratings by archetype for each book
    book_archetype_ratings = {}

    for u, b, data in G.edges(data=True):
        if u.startswith("u_") and b.startswith("b_"):
            user_id = u.replace("u_", "")
            book_id = b.replace("b_", "")
            rating = data.get("rating", 0)

            if user_id in user_archetypes:
                archetype = user_archetypes[user_id]

                if book_id not in book_archetype_ratings:
                    book_archetype_ratings[book_id] = {
                        i: [] for i in range(len(set(user_archetypes.values())))
                    }

                book_archetype_ratings[book_id][archetype].append(rating)

    # Calculate average rating per archetype for each book
    results = []

    for book_id, archetype_ratings in book_archetype_ratings.items():
        row = {"book_id": book_id}

        # Add book title
        book_data = book_df[book_df["book_id"] == book_id]
        if len(book_data) > 0:
            row["title"] = book_data.iloc[0]["title"]
            row["author"] = book_data.iloc[0]["author"]
            row["reader_count"] = book_data.iloc[0]["reader_count"]

        # Calculate average rating per archetype
        for archetype, ratings in archetype_ratings.items():
            if ratings:
                row[f"archetype_{archetype}_avg"] = np.mean(ratings)
                row[f"archetype_{archetype}_count"] = len(ratings)
            else:
                row[f"archetype_{archetype}_avg"] = None
                row[f"archetype_{archetype}_count"] = 0

        results.append(row)

    preference_df = pd.DataFrame(results)

    return preference_df


def find_cold_start_recommendations(
    user_df, preference_df, cluster_profiles, user_properties
):
    """
    Predict archetype for a new user and find recommended books

    Args:
        user_df: DataFrame with all user data
        preference_df: DataFrame with book preferences by archetype
        cluster_profiles: DataFrame with archetype profiles
        user_properties: Dict with what we know about a new user

    Returns:
        recommended_books: DataFrame with recommended books
        assigned_archetype: Dict with archetype info
    """
    # Determine which archetype the user is closest to
    archetypes = []

    for _, profile in cluster_profiles.iterrows():
        # Initialize match score
        score = 0
        matches = []

        # Check each property we have
        if "age" in user_properties:
            # This is a placeholder - we'd need age data in our database
            pass

        if "favorite_genres" in user_properties:
            # This is a placeholder - we'd need genre data in our database
            pass

        if "rating_style" in user_properties:
            style = user_properties["rating_style"]
            if style == "critical" and profile["rating_skew"] < 0:
                score += 2
                matches.append("critical rating style")
            elif style == "generous" and profile["rating_skew"] > 0.3:
                score += 2
                matches.append("generous rating style")
            elif style == "neutral" and abs(profile["rating_skew"]) < 0.3:
                score += 2
                matches.append("neutral rating style")

        if "reading_frequency" in user_properties:
            # This would need additional data
            pass

        archetypes.append(
            {
                "archetype_id": profile["archetype_id"],
                "name": profile["name"],
                "match_score": score,
                "matches": matches,
            }
        )

    # Find best archetype match
    archetypes.sort(key=lambda x: x["match_score"], reverse=True)
    best_match = archetypes[0]

    # Find popular books with high ratings from this archetype
    archetype_id = best_match["archetype_id"]

    # Create columns to look at
    rating_col = f"archetype_{archetype_id}_avg"
    count_col = f"archetype_{archetype_id}_count"

    # Filter for books with sufficient ratings from this archetype
    min_ratings = 3
    recommended = preference_df[preference_df[count_col] >= min_ratings].copy()

    if len(recommended) > 0:
        # Sort by average rating
        recommended = recommended.sort_values(rating_col, ascending=False)

        # Take top recommendations
        top_recommendations = recommended.head(10)[
            ["book_id", "title", "author", rating_col, count_col]
        ]

        return top_recommendations, best_match
    else:
        # Fallback to overall popular books
        popular_books = preference_df.sort_values("reader_count", ascending=False).head(
            10
        )
        return popular_books[["book_id", "title", "author", "reader_count"]], best_match


def main():
    """Main function to run the user archetype analysis"""
    print("Starting User Archetype Analysis...")

    # Build user-book graph
    G, user_df, book_df = build_user_book_graph(
        min_ratings_per_user=10, min_readers_per_book=5, max_users=5000
    )

    # Identify user archetypes
    user_df, cluster_profiles = identify_user_archetypes(user_df, n_clusters=5)

    print("\nUser Archetypes:")
    for _, profile in cluster_profiles.iterrows():
        print(f"Archetype {profile['archetype_id']}: {profile['name']}")
        print(f"  {profile['description']}")
        print(f"  Size: {profile['size']} users ({profile['percentage']:.1f}%)")
        print(
            f"  Avg Rating: {profile['avg_rating']:.2f}, Entropy: {profile['rating_entropy']:.2f}"
        )
        print(
            f"  Skew: {profile['rating_skew']:.2f}, Extremity: {profile['rating_extremity']:.2f}"
        )
        print()

    # Create visualizations
    print("Creating visualizations...")
    fig_pca, fig_tsne, fig_radar, pca_df, tsne_df = create_archetype_visualization(
        user_df, cluster_profiles
    )

    # For interactive use in a notebook:
    # fig_pca.show()
    # fig_tsne.show()
    # fig_radar.show()

    # Network visualization - this might be slow for large graphs
    network_fig = create_user_book_network_visualization(G, user_df, book_df)

    # Analyze archetype preferences
    print("Analyzing archetype book preferences...")
    preference_df = analyze_archetype_preferences(G, user_df, book_df)

    # Example of cold start recommendation
    print("\nExample Cold Start Recommendation:")
    # Assuming we know the user tends to be critical
    user_properties = {"rating_style": "critical"}
    recommendations, assigned_archetype = find_cold_start_recommendations(
        user_df, preference_df, cluster_profiles, user_properties
    )

    print(f"User matched to archetype: {assigned_archetype['name']}")
    print(f"Matches: {', '.join(assigned_archetype['matches'])}")
    print("\nTop Book Recommendations:")
    if "reader_count" in recommendations.columns:
        for _, book in recommendations.iterrows():
            print(
                f"- {book['title']} by {book['author']} ({book['reader_count']} readers)"
            )
    else:
        rating_col = f"archetype_{assigned_archetype['archetype_id']}_avg"
        count_col = f"archetype_{assigned_archetype['archetype_id']}_count"
        for _, book in recommendations.iterrows():
            print(
                f"- {book['title']} by {book['author']} "
                f"(Rating: {book[rating_col]:.2f} from {book[count_col]} readers)"
            )

    print("\nAnalysis complete!")
    return {
        "graph": G,
        "user_df": user_df,
        "book_df": book_df,
        "cluster_profiles": cluster_profiles,
        "visualizations": {
            "pca": fig_pca,
            "tsne": fig_tsne,
            "radar": fig_radar,
            "network": network_fig,
        },
        "preference_df": preference_df,
    }


if __name__ == "__main__":
    main()
