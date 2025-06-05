import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx


class VisualizationManager:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_output_path(self, name: str, extension: str = "png") -> Path:
        """Generate a unique output path for a visualization."""
        return self.output_dir / f"{name}_{self.timestamp}.{extension}"

    def visualize_llm_output(
        self, llm_output: Dict[str, Any], title: str = "LLM Output"
    ):
        """Visualize LLM output as a structured tree."""
        plt.figure(figsize=(12, 8))

        # Create a tree structure
        G = nx.DiGraph()

        # Add root node
        G.add_node("root", label="LLM Output")

        # Add nodes for each key in the output
        for key, value in llm_output.items():
            if isinstance(value, (dict, list)):
                G.add_node(key, label=f"{key}")
                G.add_edge("root", key)

                # Handle nested structures
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        node_id = f"{key}_{subkey}"
                        G.add_node(node_id, label=f"{subkey}")
                        G.add_edge(key, node_id)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        node_id = f"{key}_{i}"
                        G.add_node(node_id, label=f"Item {i}")
                        G.add_edge(key, node_id)
            else:
                node_id = f"{key}"
                G.add_node(node_id, label=f"{key}: {value}")
                G.add_edge("root", node_id)

        # Draw the tree
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            font_size=8,
            font_weight="bold",
        )

        plt.title(title)
        plt.savefig(self._get_output_path("llm_output"))
        plt.close()

    def visualize_exploration_graph(
        self, graph: nx.Graph, title: str = "Exploration Graph"
    ):
        """Visualize the current exploration graph."""
        plt.figure(figsize=(12, 8))

        # Draw the graph
        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="lightcoral",
            node_size=2000,
            font_size=8,
            font_weight="bold",
        )

        plt.title(title)
        plt.savefig(self._get_output_path("exploration_graph"))
        plt.close()

    def visualize_clusters(
        self, clusters: Dict[str, List[str]], title: str = "Clusters"
    ):
        """Visualize the clustering results."""
        plt.figure(figsize=(12, 8))

        G = nx.Graph()

        # Add nodes for each cluster
        for cluster_id, nodes in clusters.items():
            G.add_node(cluster_id, label=f"Cluster {cluster_id}")
            for node in nodes:
                G.add_node(node, label=node)
                G.add_edge(cluster_id, node)

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightyellow",
            node_size=2000,
            font_size=8,
            font_weight="bold",
        )

        plt.title(title)
        plt.savefig(self._get_output_path("clusters"))
        plt.close()

    def save_visualization_data(self, data: Dict[str, Any], name: str):
        """Save visualization data to a JSON file."""
        output_path = self._get_output_path(name, "json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
