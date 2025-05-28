"""
Pipeline visualizer for VULCAN feature engineering system.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PipelineVisualizer:
    """
    Real-time visualizer for VULCAN pipeline progress.

    Provides live updates on agent activities, MCTS progress,
    feature performance, and baseline comparisons.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_time = time.time()

        # Data storage for visualization
        self.mcts_history = []
        self.agent_activities = []
        self.feature_scores = []
        self.baseline_scores = {}
        self.reflection_events = []
        self.mcts_tree_snapshots = []  # Store MCTS tree snapshots for graph visualization

        # Visualization settings
        self.update_interval = config.get("visualization", {}).get("update_interval", 5)
        self.max_history_points = config.get("visualization", {}).get(
            "max_history", 100
        )

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def log_mcts_iteration(
        self,
        iteration: int,
        best_score: float,
        nodes_explored: int,
        current_features: List[str],
    ) -> None:
        """
        Log MCTS iteration data.

        Args:
            iteration: Current iteration number
            best_score: Best score achieved so far
            nodes_explored: Number of nodes explored
            current_features: List of current feature names
        """
        timestamp = time.time() - self.start_time

        self.mcts_history.append(
            {
                "timestamp": timestamp,
                "iteration": iteration,
                "best_score": best_score,
                "nodes_explored": nodes_explored,
                "num_features": len(current_features),
                "features": current_features.copy(),
            }
        )

        # Keep only recent history
        if len(self.mcts_history) > self.max_history_points:
            self.mcts_history = self.mcts_history[-self.max_history_points :]

    def log_agent_activity(
        self, agent_name: str, activity: str, details: Dict[str, Any]
    ) -> None:
        """
        Log agent activity.

        Args:
            agent_name: Name of the agent
            activity: Type of activity
            details: Additional details about the activity
        """
        timestamp = time.time() - self.start_time

        self.agent_activities.append(
            {
                "timestamp": timestamp,
                "agent": agent_name,
                "activity": activity,
                "details": details,
            }
        )

        # Keep only recent activities
        if len(self.agent_activities) > self.max_history_points:
            self.agent_activities = self.agent_activities[-self.max_history_points :]

    def log_feature_evaluation(
        self, feature_name: str, score: float, evaluation_time: float
    ) -> None:
        """
        Log feature evaluation result.

        Args:
            feature_name: Name of the feature
            score: Evaluation score
            evaluation_time: Time taken for evaluation
        """
        timestamp = time.time() - self.start_time

        self.feature_scores.append(
            {
                "timestamp": timestamp,
                "feature_name": feature_name,
                "score": score,
                "evaluation_time": evaluation_time,
            }
        )

    def log_reflection_event(
        self, reflection_type: str, content: str, insights: List[str]
    ) -> None:
        """
        Log reflection event.

        Args:
            reflection_type: Type of reflection
            content: Reflection content
            insights: Key insights from reflection
        """
        timestamp = time.time() - self.start_time

        self.reflection_events.append(
            {
                "timestamp": timestamp,
                "type": reflection_type,
                "content": content,
                "insights": insights,
            }
        )

    def update_baseline_scores(self, baseline_scores: Dict[str, float]) -> None:
        """
        Update baseline scores for comparison.

        Args:
            baseline_scores: Dictionary of baseline model scores
        """
        self.baseline_scores = baseline_scores.copy()

    def log_mcts_tree_snapshot(self, root_node, iteration: int) -> None:
        """
        Log a snapshot of the MCTS tree for graph visualization.

        Args:
            root_node: Root node of the MCTS tree
            iteration: Current iteration number
        """
        timestamp = time.time() - self.start_time

        # Extract graph data from the tree
        graph_data = self._extract_mcts_graph_data(root_node)

        self.mcts_tree_snapshots.append(
            {"timestamp": timestamp, "iteration": iteration, "graph_data": graph_data}
        )

        # Keep only recent snapshots
        if len(self.mcts_tree_snapshots) > 20:  # Keep last 20 snapshots
            self.mcts_tree_snapshots = self.mcts_tree_snapshots[-20:]

    def _extract_mcts_graph_data(self, root_node) -> Dict[str, Any]:
        """
        Extract graph data from MCTS tree for visualization.

        Args:
            root_node: Root node of the MCTS tree

        Returns:
            Dictionary containing nodes and edges data
        """
        nodes = []
        edges = []

        def traverse_tree(node, visited=None):
            if visited is None:
                visited = set()

            if node.node_id in visited:
                return
            visited.add(node.node_id)

            # Add node data
            node_data = {
                "id": node.node_id,
                "label": f"Node {node.node_id[:8]}",
                "visits": node.visits,
                "value": node.value,
                "score": node.score,
                "depth": node.depth,
                "feature_count": len(node.state_features),
                "feature_names": [f.name for f in node.state_features],
                "feature_that_led_here": node.feature_that_led_here.name
                if node.feature_that_led_here
                else "ROOT",
                "avg_value": node.value / max(node.visits, 1),
                "is_terminal": node.is_terminal,
            }
            nodes.append(node_data)

            # Add edges to children
            for child in node.children:
                edge_data = {
                    "source": node.node_id,
                    "target": child.node_id,
                    "feature": child.feature_that_led_here.name
                    if child.feature_that_led_here
                    else "",
                    "weight": child.visits,
                }
                edges.append(edge_data)

                # Recursively traverse children
                traverse_tree(child, visited)

        traverse_tree(root_node)

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "max_depth": max([n["depth"] for n in nodes]) if nodes else 0,
        }

    def create_mcts_graph_visualization(
        self, save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive MCTS graph visualization.

        Args:
            save_path: Optional path to save the visualization

        Returns:
            Plotly figure object
        """
        if not self.mcts_tree_snapshots:
            logger.warning("No MCTS tree snapshots available for visualization")
            return go.Figure()

        # Use the latest snapshot
        latest_snapshot = self.mcts_tree_snapshots[-1]
        graph_data = latest_snapshot["graph_data"]

        # Create NetworkX graph for layout calculation
        G = nx.DiGraph()

        # Add nodes
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **node)

        # Add edges
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], **edge)

        # Calculate layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            # Fallback to random layout if spring layout fails
            pos = {node: (np.random.random(), np.random.random()) for node in G.nodes()}

        # Prepare data for Plotly
        node_trace = self._create_node_trace(graph_data["nodes"], pos)
        edge_trace = self._create_edge_trace(graph_data["edges"], pos)

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"MCTS Tree Visualization (Iteration {latest_snapshot['iteration']})",
                    font=dict(size=16),
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Nodes: {graph_data['total_nodes']}, Edges: {graph_data['total_edges']}, Max Depth: {graph_data['max_depth']}",
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
                plot_bgcolor="white",
            ),
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"MCTS graph visualization saved to {save_path}")

        return fig

    def _create_node_trace(self, nodes: List[Dict], pos: Dict) -> go.Scatter:
        """Create node trace for MCTS graph visualization."""
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in nodes:
            x, y = pos.get(node["id"], (0, 0))
            node_x.append(x)
            node_y.append(y)

            # Create hover text
            hover_text = (
                f"Node: {node['id'][:8]}<br>"
                f"Feature: {node['feature_that_led_here']}<br>"
                f"Visits: {node['visits']}<br>"
                f"Score: {node['score']:.4f}<br>"
                f"Avg Value: {node['avg_value']:.4f}<br>"
                f"Depth: {node['depth']}<br>"
                f"Features: {', '.join(node['feature_names'][:3])}"
            )
            if len(node["feature_names"]) > 3:
                hover_text += f"... (+{len(node['feature_names']) - 3} more)"

            node_text.append(hover_text)

            # Color by score (normalized)
            node_color.append(node["score"])

            # Size by visits (with minimum size)
            node_size.append(max(10, min(50, 10 + node["visits"] * 2)))

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Score", thickness=15, len=0.5),
                line=dict(width=2, color="black"),
            ),
        )

    def _create_edge_trace(self, edges: List[Dict], pos: Dict) -> go.Scatter:
        """Create edge trace for MCTS graph visualization."""
        edge_x = []
        edge_y = []

        for edge in edges:
            x0, y0 = pos.get(edge["source"], (0, 0))
            x1, y1 = pos.get(edge["target"], (0, 0))

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="gray"),
            hoverinfo="none",
            mode="lines",
        )

    def create_mcts_evolution_animation(
        self, save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create an animated visualization showing MCTS tree evolution over time.

        Args:
            save_path: Optional path to save the animation

        Returns:
            Plotly figure with animation
        """
        if len(self.mcts_tree_snapshots) < 2:
            logger.warning("Need at least 2 snapshots for animation")
            return go.Figure()

        frames = []

        for i, snapshot in enumerate(self.mcts_tree_snapshots):
            graph_data = snapshot["graph_data"]

            # Create NetworkX graph for this frame
            G = nx.DiGraph()
            for node in graph_data["nodes"]:
                G.add_node(node["id"], **node)
            for edge in graph_data["edges"]:
                G.add_edge(edge["source"], edge["target"], **edge)

            # Calculate layout (use consistent seed for stability)
            try:
                pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
            except:
                pos = {
                    node: (np.random.random(), np.random.random()) for node in G.nodes()
                }

            # Create traces for this frame
            node_trace = self._create_node_trace(graph_data["nodes"], pos)
            edge_trace = self._create_edge_trace(graph_data["edges"], pos)

            frame = go.Frame(
                data=[edge_trace, node_trace],
                name=str(i),
                layout=go.Layout(
                    title=f"MCTS Tree Evolution - Iteration {snapshot['iteration']}"
                ),
            )
            frames.append(frame)

        # Create initial figure with first frame
        initial_graph = self.mcts_tree_snapshots[0]["graph_data"]
        G_initial = nx.DiGraph()
        for node in initial_graph["nodes"]:
            G_initial.add_node(node["id"], **node)
        for edge in initial_graph["edges"]:
            G_initial.add_edge(edge["source"], edge["target"], **edge)

        pos_initial = nx.spring_layout(G_initial, k=2, iterations=30, seed=42)
        initial_node_trace = self._create_node_trace(
            initial_graph["nodes"], pos_initial
        )
        initial_edge_trace = self._create_edge_trace(
            initial_graph["edges"], pos_initial
        )

        fig = go.Figure(
            data=[initial_edge_trace, initial_node_trace],
            frames=frames,
            layout=go.Layout(
                title="MCTS Tree Evolution Animation",
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 1000, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            },
                        ],
                    }
                ],
                sliders=[
                    {
                        "steps": [
                            {
                                "args": [
                                    [str(i)],
                                    {
                                        "frame": {"duration": 300, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 300},
                                    },
                                ],
                                "label": f"Iter {snapshot['iteration']}",
                                "method": "animate",
                            }
                            for i, snapshot in enumerate(self.mcts_tree_snapshots)
                        ],
                        "active": 0,
                        "currentvalue": {"prefix": "Iteration: "},
                        "len": 0.9,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 0,
                        "yanchor": "top",
                    }
                ],
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            ),
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"MCTS evolution animation saved to {save_path}")

        return fig

    def create_live_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create a live dashboard showing pipeline progress.

        Args:
            save_path: Optional path to save the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "MCTS Progress",
                "Agent Activities",
                "Feature Performance",
                "Baseline Comparison",
                "Exploration Timeline",
                "Recent Reflections",
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"colspan": 2}, None],
            ],
        )

        # 1. MCTS Progress
        if self.mcts_history:
            df_mcts = pd.DataFrame(self.mcts_history)

            # Best score over time
            fig.add_trace(
                go.Scatter(
                    x=df_mcts["timestamp"],
                    y=df_mcts["best_score"],
                    mode="lines+markers",
                    name="Best Score",
                    line=dict(color="blue", width=3),
                ),
                row=1,
                col=1,
            )

            # Nodes explored (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=df_mcts["timestamp"],
                    y=df_mcts["nodes_explored"],
                    mode="lines",
                    name="Nodes Explored",
                    line=dict(color="red", dash="dash"),
                    yaxis="y2",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # 2. Agent Activities
        if self.agent_activities:
            df_agents = pd.DataFrame(self.agent_activities)
            activity_counts = df_agents["agent"].value_counts()

            fig.add_trace(
                go.Bar(
                    x=activity_counts.index,
                    y=activity_counts.values,
                    name="Agent Activities",
                    marker_color="lightblue",
                ),
                row=1,
                col=2,
            )

        # 3. Feature Performance
        if self.feature_scores:
            df_features = pd.DataFrame(self.feature_scores)
            recent_features = df_features.tail(10)

            fig.add_trace(
                go.Scatter(
                    x=recent_features["timestamp"],
                    y=recent_features["score"],
                    mode="markers",
                    marker=dict(
                        size=recent_features["evaluation_time"] * 10,
                        color=recent_features["score"],
                        colorscale="Viridis",
                        showscale=True,
                    ),
                    text=recent_features["feature_name"],
                    name="Feature Scores",
                ),
                row=2,
                col=1,
            )

        # 4. Baseline Comparison
        if self.baseline_scores and self.mcts_history:
            current_vulcan_score = self.mcts_history[-1]["best_score"]

            models = list(self.baseline_scores.keys()) + ["VULCAN"]
            scores = list(self.baseline_scores.values()) + [current_vulcan_score]
            colors = ["lightcoral"] * len(self.baseline_scores) + ["darkgreen"]

            fig.add_trace(
                go.Bar(
                    x=models, y=scores, name="Model Comparison", marker_color=colors
                ),
                row=2,
                col=2,
            )

        # 5. Exploration Timeline
        if self.mcts_history:
            df_mcts = pd.DataFrame(self.mcts_history)

            fig.add_trace(
                go.Scatter(
                    x=df_mcts["timestamp"],
                    y=df_mcts["num_features"],
                    mode="lines+markers",
                    name="Number of Features",
                    line=dict(color="green", width=2),
                    fill="tonexty",
                ),
                row=3,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"VULCAN Pipeline Dashboard - Runtime: {time.time() - self.start_time:.1f}s",
            height=900,
            showlegend=True,
            template="plotly_white",
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Nodes", row=1, col=1, secondary_y=True)

        fig.update_xaxes(title_text="Agent", row=1, col=2)
        fig.update_yaxes(title_text="Activities", row=1, col=2)

        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Feature Score", row=2, col=1)

        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=2)

        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Features", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")

        fig.show()

    def create_static_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create a static summary plot of the pipeline run.

        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("VULCAN Pipeline Summary", fontsize=16, fontweight="bold")

        # 1. MCTS Progress
        if self.mcts_history:
            df_mcts = pd.DataFrame(self.mcts_history)

            ax1 = axes[0, 0]
            ax1.plot(
                df_mcts["timestamp"],
                df_mcts["best_score"],
                "b-",
                linewidth=2,
                label="Best Score",
            )
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Score", color="b")
            ax1.tick_params(axis="y", labelcolor="b")

            ax1_twin = ax1.twinx()
            ax1_twin.plot(
                df_mcts["timestamp"],
                df_mcts["nodes_explored"],
                "r--",
                alpha=0.7,
                label="Nodes Explored",
            )
            ax1_twin.set_ylabel("Nodes Explored", color="r")
            ax1_twin.tick_params(axis="y", labelcolor="r")

            ax1.set_title("MCTS Progress Over Time")
            ax1.grid(True, alpha=0.3)

        # 2. Agent Activity Distribution
        if self.agent_activities:
            df_agents = pd.DataFrame(self.agent_activities)
            activity_counts = df_agents["agent"].value_counts()

            axes[0, 1].bar(
                activity_counts.index,
                activity_counts.values,
                color="lightblue",
                alpha=0.7,
            )
            axes[0, 1].set_title("Agent Activity Distribution")
            axes[0, 1].set_xlabel("Agent")
            axes[0, 1].set_ylabel("Number of Activities")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Feature Score Distribution
        if self.feature_scores:
            df_features = pd.DataFrame(self.feature_scores)

            axes[1, 0].hist(
                df_features["score"],
                bins=20,
                alpha=0.7,
                color="green",
                edgecolor="black",
            )
            axes[1, 0].set_title("Feature Score Distribution")
            axes[1, 0].set_xlabel("Score")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].axvline(
                df_features["score"].mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {df_features['score'].mean():.3f}",
            )
            axes[1, 0].legend()

        # 4. Baseline Comparison
        if self.baseline_scores and self.mcts_history:
            current_vulcan_score = self.mcts_history[-1]["best_score"]

            models = list(self.baseline_scores.keys()) + ["VULCAN"]
            scores = list(self.baseline_scores.values()) + [current_vulcan_score]
            colors = ["lightcoral"] * len(self.baseline_scores) + ["darkgreen"]

            bars = axes[1, 1].bar(models, scores, color=colors, alpha=0.7)
            axes[1, 1].set_title("Final Model Comparison")
            axes[1, 1].set_xlabel("Model")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        plt.show()

    def print_live_status(self) -> None:
        """Print current pipeline status to console."""
        runtime = time.time() - self.start_time

        print(f"\n{'=' * 60}")
        print(f"VULCAN PIPELINE STATUS - Runtime: {runtime:.1f}s")
        print(f"{'=' * 60}")

        # MCTS Status
        if self.mcts_history:
            latest = self.mcts_history[-1]
            print("MCTS Progress:")
            print(f"  Iteration: {latest['iteration']}")
            print(f"  Best Score: {latest['best_score']:.4f}")
            print(f"  Nodes Explored: {latest['nodes_explored']}")
            print(f"  Current Features: {latest['num_features']}")

        # Recent Agent Activities
        if self.agent_activities:
            recent_activities = self.agent_activities[-3:]
            print("\nRecent Agent Activities:")
            for activity in recent_activities:
                print(f"  {activity['agent']}: {activity['activity']}")

        # Feature Performance
        if self.feature_scores:
            recent_scores = [f["score"] for f in self.feature_scores[-5:]]
            avg_recent = np.mean(recent_scores)
            print("\nRecent Feature Performance:")
            print(f"  Average Score (last 5): {avg_recent:.4f}")
            print(
                f"  Best Score Overall: {max(f['score'] for f in self.feature_scores):.4f}"
            )

        # Baseline Comparison
        if self.baseline_scores and self.mcts_history:
            current_score = self.mcts_history[-1]["best_score"]
            best_baseline = max(self.baseline_scores.values())
            best_baseline_name = max(self.baseline_scores.items(), key=lambda x: x[1])[
                0
            ]

            print("\nBaseline Comparison:")
            print(f"  VULCAN: {current_score:.4f}")
            print(f"  Best Baseline ({best_baseline_name}): {best_baseline:.4f}")
            if best_baseline > 0:
                improvement = ((current_score - best_baseline) / best_baseline) * 100
                print(f"  Improvement: {improvement:+.1f}%")

        print(f"{'=' * 60}\n")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the pipeline run.

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            "runtime": time.time() - self.start_time,
            "total_iterations": len(self.mcts_history),
            "total_features_evaluated": len(self.feature_scores),
            "total_agent_activities": len(self.agent_activities),
            "total_reflections": len(self.reflection_events),
        }

        if self.mcts_history:
            scores = [h["best_score"] for h in self.mcts_history]
            stats.update(
                {
                    "final_score": scores[-1],
                    "best_score": max(scores),
                    "score_improvement": scores[-1] - scores[0]
                    if len(scores) > 1
                    else 0,
                    "final_nodes_explored": self.mcts_history[-1]["nodes_explored"],
                }
            )

        if self.feature_scores:
            feature_scores = [f["score"] for f in self.feature_scores]
            stats.update(
                {
                    "avg_feature_score": np.mean(feature_scores),
                    "best_feature_score": max(feature_scores),
                    "feature_score_std": np.std(feature_scores),
                }
            )

        return stats
