from typing import Dict, List

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def create_mcts_tree_visualization(tree_data: Dict) -> go.Figure:
    """Create an interactive MCTS tree visualization."""
    if not tree_data or "nodes" not in tree_data:
        fig = go.Figure()
        fig.add_annotation(
            text="MCTS Tree data not available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        return fig

    nodes = tree_data["nodes"]
    edges = tree_data.get("edges", [])

    # Create networkx graph for layout calculation
    G = nx.DiGraph()

    # Add nodes
    for node_id, node_data in nodes.items():
        G.add_node(node_id, **node_data)

    # Add edges
    for edge in edges:
        G.add_edge(edge["parent_id"], edge["child_id"], weight=edge.get("score", 1))

    # Calculate layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        # Fallback to random positions if spring layout fails
        import random

        pos = {node: (random.random(), random.random()) for node in G.nodes()}

    # Create traces
    edge_x = []
    edge_y = []

    for edge in edges:
        parent_id = edge["parent_id"]
        child_id = edge["child_id"]
        if parent_id in pos and child_id in pos:
            x0, y0 = pos[parent_id]
            x1, y1 = pos[child_id]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors = []
    node_sizes = []

    for node_id, node_data in nodes.items():
        if node_id in pos:
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            visits = node_data.get("visits", 0)
            score = node_data.get("evaluation_score", 0)
            feature = node_data.get("new_feature", "root")
            depth = node_data.get("depth", 0)

            node_text.append(
                f"Node: {node_id[:8]}<br>Feature: {feature}<br>Visits: {visits}"
            )
            node_info.append(
                f"ID: {node_id}<br>Score: {score:.4f}<br>Depth: {depth}<br>Visits: {visits}"
            )
            node_colors.append(score)
            node_sizes.append(max(10, min(30, 10 + visits * 2)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        hovertext=node_info,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Evaluation Score"),
            line=dict(width=2, color="white"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="MCTS Tree Visualization",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size = visits, Color = evaluation score",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="gray", size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def create_llm_interactions_timeline(llm_data: List[Dict]) -> go.Figure:
    """Create a timeline visualization of LLM interactions."""
    if not llm_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No LLM interaction data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        return fig

    df = pd.DataFrame(llm_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["tokens"] = df.apply(lambda x: x.get("metadata", {}).get("tokens", 0), axis=1)
    df["model"] = df.apply(
        lambda x: x.get("metadata", {}).get("model", "unknown"), axis=1
    )
    df["response_length"] = df["response"].str.len()

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Token Usage Over Time",
            "Response Length",
            "Interaction Frequency",
        ],
        vertical_spacing=0.08,
    )

    # Token usage
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["tokens"],
            mode="lines+markers",
            name="Tokens",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Response length
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["response_length"],
            mode="lines+markers",
            name="Response Length",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    # Interaction frequency (histogram by hour)
    df["hour"] = df["timestamp"].dt.hour
    hourly_counts = df.groupby("hour").size().reset_index(name="count")
    fig.add_trace(
        go.Bar(
            x=hourly_counts["hour"],
            y=hourly_counts["count"],
            name="Interactions per Hour",
            marker_color="orange",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=800, title_text="LLM Interaction Analysis", showlegend=False
    )
    return fig


def create_feature_evolution_chart(experiments: List[Dict]) -> go.Figure:
    """Create a chart showing feature evolution over experiments."""
    if not experiments:
        return go.Figure()

    df = pd.DataFrame(experiments)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Score Evolution",
            "Feature Count Growth",
            "Evaluation Time Trends",
            "Feature Performance Heatmap",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "heatmap"}],
        ],
    )

    # Score evolution with trend line
    fig.add_trace(
        go.Scatter(
            x=df["iteration"],
            y=df["overall_score"],
            mode="lines+markers",
            name="Score",
            line=dict(color="blue", width=3),
        ),
        row=1,
        col=1,
    )

    # Add trend line
    z = np.polyfit(df["iteration"], df["overall_score"], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df["iteration"],
            y=p(df["iteration"]),
            mode="lines",
            name="Trend",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Feature count growth
    fig.add_trace(
        go.Bar(
            x=df["iteration"],
            y=df["feature_count"],
            name="Features",
            marker_color="green",
        ),
        row=1,
        col=2,
    )

    # Evaluation time trends
    fig.add_trace(
        go.Scatter(
            x=df["iteration"],
            y=df["evaluation_time"],
            mode="lines+markers",
            name="Eval Time",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    # Feature performance heatmap (if we have feature details)
    if "features" in df.columns and df["features"].iloc[0]:
        # Create a simple heatmap of scores vs iterations
        scores_matrix = []
        for i, exp in df.iterrows():
            scores_matrix.append([exp["overall_score"]] * 5)  # Simplified

        fig.add_trace(
            go.Heatmap(z=scores_matrix, colorscale="Viridis", showscale=False),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=600, title_text="Feature Evolution Analysis", showlegend=False
    )
    return fig


def render_real_time_monitoring():
    """Render real-time monitoring section with live updates."""
    st.subheader("🔴 Live System Monitoring")

    # Create columns for real-time metrics
    col1, col2, col3, col4 = st.columns(4)

    # Placeholder for live metrics (would be updated via WebSocket in production)
    with col1:
        st.metric("Active Experiments", "1", delta="0")

    with col2:
        st.metric("Nodes Explored", "156", delta="+12")

    with col3:
        st.metric("Current Score", "0.7814", delta="+0.0023")

    with col4:
        st.metric("LLM Calls", "23", delta="+2")

    # Real-time activity feed
    st.subheader("Activity Feed")
    activity_data = [
        {
            "time": "14:32:15",
            "event": "New MCTS node explored",
            "details": "Feature: user_behavior_pattern_5",
        },
        {
            "time": "14:32:10",
            "event": "LLM generated feature",
            "details": "Token usage: 1,245",
        },
        {
            "time": "14:32:05",
            "event": "Evaluation completed",
            "details": "Score: 0.7814",
        },
        {
            "time": "14:32:00",
            "event": "Feature mutation applied",
            "details": "Parent: user_rating_variance",
        },
    ]

    for activity in activity_data:
        st.write(f"**{activity['time']}** - {activity['event']}")
        st.caption(activity["details"])
        st.divider()


# Import numpy for trend calculations
import numpy as np
