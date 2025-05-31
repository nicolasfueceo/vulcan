import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Import enhanced visualizations
from enhanced_visualizations import (
    create_feature_evolution_chart,
    create_llm_interactions_timeline,
    create_mcts_tree_visualization,
    render_real_time_monitoring,
)
from plotly.subplots import make_subplots

# Constants
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds

# Page config
st.set_page_config(
    page_title="VULCAN 2.0 Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .experiment-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# Helper functions
def fetch_data(endpoint: str) -> Dict:
    """Fetch data from the API endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {endpoint}: {str(e)}")
        return {}


def create_data_stats_visualizations(stats: Dict) -> Dict[str, go.Figure]:
    """Create visualizations for data statistics."""
    figures = {}

    if not stats:
        return figures

    # Rating distribution
    if "ratingDistribution" in stats:
        rating_data = stats["ratingDistribution"]
        df_ratings = pd.DataFrame(rating_data)

        figures["rating_dist"] = px.bar(
            df_ratings,
            x="rating",
            y="count",
            title="Rating Distribution",
            labels={"rating": "Rating", "count": "Number of Ratings"},
            color="count",
            color_continuous_scale="viridis",
        )
        figures["rating_dist"].update_layout(showlegend=False)

    # Year distribution
    if "yearDistribution" in stats:
        year_data = stats["yearDistribution"]
        df_years = pd.DataFrame(year_data)

        figures["year_dist"] = px.line(
            df_years,
            x="year",
            y="count",
            title="Books by Publication Decade",
            labels={"year": "Publication Decade", "count": "Number of Books"},
            markers=True,
        )

    # Fold statistics
    if "foldStatistics" in stats and stats["foldStatistics"]["outer"]:
        outer_folds = stats["foldStatistics"]["outer"]
        df_folds = pd.DataFrame(outer_folds)

        figures["fold_stats"] = px.bar(
            df_folds,
            x="fold",
            y=["trainUsers", "valUsers"],
            title="Cross-Validation Fold Statistics",
            labels={"value": "Number of Users", "variable": "Split Type"},
            barmode="group",
        )

    return figures


def create_experiment_performance_chart(experiments: List[Dict]) -> go.Figure:
    """Create performance chart for experiments."""
    if not experiments:
        return go.Figure()

    df = pd.DataFrame(experiments)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Overall Score Trend",
            "Feature Count vs Score",
            "Evaluation Time",
            "Score Distribution",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Score trend
    fig.add_trace(
        go.Scatter(
            x=df["iteration"],
            y=df["overall_score"],
            mode="lines+markers",
            name="Score Trend",
        ),
        row=1,
        col=1,
    )

    # Feature count vs score
    fig.add_trace(
        go.Scatter(
            x=df["feature_count"],
            y=df["overall_score"],
            mode="markers",
            name="Feature Count vs Score",
        ),
        row=1,
        col=2,
    )

    # Evaluation time
    fig.add_trace(
        go.Bar(x=df["iteration"], y=df["evaluation_time"], name="Eval Time"),
        row=2,
        col=1,
    )

    # Score distribution
    fig.add_trace(
        go.Histogram(x=df["overall_score"], name="Score Distribution"), row=2, col=2
    )

    fig.update_layout(
        height=600, showlegend=False, title_text="Experiment Performance Analysis"
    )
    return fig


def render_system_status():
    """Render system status section."""
    st.subheader("🏥 System Health")

    col1, col2 = st.columns(2)

    with col1:
        # Health check
        health_data = fetch_data("/api/health")
        if health_data:
            status = health_data.get("status", "unknown")
            if status == "healthy":
                st.markdown(
                    f'<div class="status-healthy">✅ System Status: {status.upper()}</div>',
                    unsafe_allow_html=True,
                )
                st.success(f"Version: {health_data.get('version', 'unknown')}")
            else:
                st.markdown(
                    f'<div class="status-error">❌ System Status: {status.upper()}</div>',
                    unsafe_allow_html=True,
                )

    with col2:
        # System status
        status_data = fetch_data("/api/status")
        if status_data:
            st.metric("Experiments Count", status_data.get("experiments_count", 0))

            components = status_data.get("components", {})
            st.write("**Component Status:**")
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                st.write(f"{status_icon} {component}")


def render_data_insights():
    """Render data insights section."""
    st.subheader("📊 Dataset Insights")

    stats_response = fetch_data("/api/data/stats")
    if not stats_response or stats_response.get("status") != "success":
        st.warning("Unable to load dataset statistics")
        return

    stats = stats_response.get("data", {})

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", f"{stats.get('totalUsers', 0):,}")
    with col2:
        st.metric("Total Books", f"{stats.get('totalBooks', 0):,}")
    with col3:
        st.metric("Total Ratings", f"{stats.get('totalRatings', 0):,}")
    with col4:
        sparsity = stats.get("sparsity", 0)
        st.metric("Sparsity", f"{sparsity:.4%}")

    # Additional metrics
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Avg Ratings/User", f"{stats.get('avgRatingsPerUser', 0):.1f}")
    with col6:
        st.metric("Avg Ratings/Book", f"{stats.get('avgRatingsPerBook', 0):.1f}")

    # Visualizations
    figures = create_data_stats_visualizations(stats)

    if figures:
        col1, col2 = st.columns(2)

        with col1:
            if "rating_dist" in figures:
                st.plotly_chart(figures["rating_dist"], use_container_width=True)
            if "fold_stats" in figures:
                st.plotly_chart(figures["fold_stats"], use_container_width=True)

        with col2:
            if "year_dist" in figures:
                st.plotly_chart(figures["year_dist"], use_container_width=True)


def render_experiments():
    """Render experiments section."""
    st.subheader("🧪 Experiment History")

    experiments_data = fetch_data("/api/experiments")

    if not experiments_data:
        st.warning("No experiment data available")
        return

    if not experiments_data:
        st.info(
            "No experiments have been run yet. Start a new experiment to see results here."
        )
        return

    # Performance chart
    if len(experiments_data) > 1:
        st.subheader("Performance Analysis")
        perf_fig = create_experiment_performance_chart(experiments_data)
        st.plotly_chart(perf_fig, use_container_width=True)

    # Feature evolution chart
    if len(experiments_data) > 1:
        st.subheader("Feature Evolution")
        evolution_fig = create_feature_evolution_chart(experiments_data)
        st.plotly_chart(evolution_fig, use_container_width=True)

    # Experiments table
    st.subheader("Experiment Results")

    # Create a more detailed dataframe
    exp_df = pd.DataFrame(experiments_data)
    display_df = exp_df[
        ["iteration", "fold_id", "overall_score", "feature_count", "evaluation_time"]
    ].copy()
    display_df.columns = ["Iteration", "Fold ID", "Score", "Features", "Eval Time (s)"]
    display_df["Score"] = display_df["Score"].round(4)
    display_df["Eval Time (s)"] = display_df["Eval Time (s)"].round(2)

    st.dataframe(display_df, use_container_width=True)

    # Best performing experiment
    if experiments_data:
        best_exp = max(experiments_data, key=lambda x: x["overall_score"])
        st.success(
            f"🏆 Best Score: {best_exp['overall_score']:.4f} (Iteration {best_exp['iteration']})"
        )


def render_mcts_visualization():
    """Render MCTS tree visualization section."""
    st.subheader("🌳 MCTS Tree Exploration")

    # Try to fetch MCTS tree data
    tree_data = fetch_data("/api/tree")

    # For now, create mock data since the endpoint isn't implemented
    mock_tree_data = {
        "nodes": {
            "root": {
                "node_id": "root",
                "visits": 50,
                "evaluation_score": 0.75,
                "new_feature": "root",
                "depth": 0,
                "children": ["node_1", "node_2"],
            },
            "node_1": {
                "node_id": "node_1",
                "visits": 25,
                "evaluation_score": 0.78,
                "new_feature": "user_behavior_pattern_1",
                "depth": 1,
                "children": ["node_3"],
            },
            "node_2": {
                "node_id": "node_2",
                "visits": 20,
                "evaluation_score": 0.72,
                "new_feature": "item_popularity_score",
                "depth": 1,
                "children": [],
            },
            "node_3": {
                "node_id": "node_3",
                "visits": 15,
                "evaluation_score": 0.81,
                "new_feature": "user_rating_variance",
                "depth": 2,
                "children": [],
            },
        },
        "edges": [
            {"parent_id": "root", "child_id": "node_1", "score": 0.78},
            {"parent_id": "root", "child_id": "node_2", "score": 0.72},
            {"parent_id": "node_1", "child_id": "node_3", "score": 0.81},
        ],
    }

    # Show tree visualization
    fig = create_mcts_tree_visualization(mock_tree_data)
    st.plotly_chart(fig, use_container_width=True)

    # Tree statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", len(mock_tree_data["nodes"]))
    with col2:
        st.metric(
            "Max Depth", max(node["depth"] for node in mock_tree_data["nodes"].values())
        )
    with col3:
        best_score = max(
            node["evaluation_score"] for node in mock_tree_data["nodes"].values()
        )
        st.metric("Best Score", f"{best_score:.4f}")


def render_llm_interactions():
    """Render LLM interactions section."""
    st.subheader("🤖 LLM Interactions")

    # Mock LLM data since the endpoint doesn't exist yet
    mock_llm_data = [
        {
            "timestamp": "2024-05-30T14:30:00Z",
            "prompt": "Generate a new feature based on user behavior patterns...",
            "response": "user_behavior_pattern_1 = user_ratings.groupby('user_id').rating.std()",
            "metadata": {"tokens": 1245, "model": "gpt-4"},
        },
        {
            "timestamp": "2024-05-30T14:25:00Z",
            "prompt": "Mutate the existing feature to improve performance...",
            "response": "user_rating_variance = user_ratings.groupby('user_id').rating.var() * 0.8",
            "metadata": {"tokens": 892, "model": "gpt-4"},
        },
        {
            "timestamp": "2024-05-30T14:20:00Z",
            "prompt": "Analyze the performance of current features...",
            "response": "The user_behavior_pattern_1 shows good correlation with clustering performance...",
            "metadata": {"tokens": 1534, "model": "gpt-4"},
        },
    ]

    # Timeline visualization
    fig = create_llm_interactions_timeline(mock_llm_data)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed interactions
    st.subheader("Recent Interactions")
    for i, interaction in enumerate(mock_llm_data):
        with st.expander(f"Interaction {i + 1} - {interaction['timestamp']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Prompt:**")
                st.code(interaction["prompt"], language="text")
                st.markdown("**Response:**")
                st.code(interaction["response"], language="python")
            with col2:
                st.metric("Tokens", interaction["metadata"]["tokens"])
                st.metric("Model", interaction["metadata"]["model"])


def render_performance_tracking():
    """Render performance tracking section."""
    st.subheader("📈 Performance Tracking")

    col1, col2 = st.columns(2)

    with col1:
        # Performance summary
        summary_data = fetch_data("/api/performance/summary")
        if summary_data and summary_data.get("status") == "success":
            summary = summary_data.get("data", {})
            st.write("**Performance Summary:**")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    st.metric(
                        key.replace("_", " ").title(),
                        f"{value:.4f}" if isinstance(value, float) else value,
                    )
        else:
            st.info("Performance tracking data not available")

    with col2:
        # Best features
        features_data = fetch_data("/api/performance/features?top_k=5")
        if features_data and features_data.get("status") == "success":
            best_features = features_data.get("data", {}).get("best_features", [])
            if best_features:
                st.write("**Top Performing Features:**")
                for i, feature in enumerate(best_features, 1):
                    st.write(
                        f"{i}. {feature.get('name', 'Unknown')} (Score: {feature.get('score', 0):.4f})"
                    )
        else:
            st.info("Feature performance data not available")


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("🔥 VULCAN 2.0")
    st.sidebar.markdown("---")

    # Auto-refresh controls
    st.sidebar.subheader("🔄 Refresh Settings")
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh", value=st.session_state.auto_refresh
    )
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)", 1, 60, REFRESH_INTERVAL
    )

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.subheader("📍 Navigation")
    sections = {
        "🏥 System Status": "status",
        "📊 Dataset": "data",
        "🧪 Experiments": "experiments",
        "🌳 MCTS Tree": "mcts",
        "🤖 LLM Interactions": "llm",
        "📈 Performance": "performance",
        "🔴 Live Monitor": "live",
    }

    selected_section = st.sidebar.radio("Go to section:", list(sections.keys()))

    st.sidebar.markdown("---")
    st.sidebar.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    return auto_refresh, refresh_interval, sections[selected_section]


# Main app
def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🔥 VULCAN 2.0 Dashboard</h1>', unsafe_allow_html=True
    )
    st.markdown("**Autonomous Feature Engineering for Recommender Systems**")
    st.markdown("---")

    # Sidebar
    auto_refresh, refresh_interval, selected_section = render_sidebar()

    # Update session state
    st.session_state.auto_refresh = auto_refresh

    # Main content based on selection
    if selected_section == "status":
        render_system_status()
    elif selected_section == "data":
        render_data_insights()
    elif selected_section == "experiments":
        render_experiments()
    elif selected_section == "mcts":
        render_mcts_visualization()
    elif selected_section == "llm":
        render_llm_interactions()
    elif selected_section == "performance":
        render_performance_tracking()
    elif selected_section == "live":
        render_real_time_monitoring()
    else:
        # Default: show all sections
        render_system_status()
        st.markdown("---")
        render_data_insights()
        st.markdown("---")
        render_experiments()
        st.markdown("---")
        render_mcts_visualization()
        st.markdown("---")
        render_llm_interactions()
        st.markdown("---")
        render_performance_tracking()

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
