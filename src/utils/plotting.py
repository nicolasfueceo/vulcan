from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns


class PlotManager:
    def __init__(self, base_dir: str = "outputs/plots"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Set up default plotting style"""
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def _generate_filename(self, base_name: str, plot_type: str) -> str:
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{plot_type}_{timestamp}.png"

    def save_plot(
        self,
        plot_type: str,
        base_name: str,
        fig: Optional[plt.Figure] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dpi: int = 300,
    ) -> str:
        """Save the current plot with metadata"""
        if fig is None:
            fig = plt.gcf()

        filename = self._generate_filename(base_name, plot_type)
        filepath = self.base_dir / filename

        # Add metadata as text in the figure if provided
        if metadata:
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            fig.text(0.02, 0.02, metadata_str, fontsize=8, alpha=0.7)

        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return str(filepath)

    def create_subplot_grid(self, n_plots: int) -> tuple:
        """Calculate optimal subplot grid dimensions"""
        n_rows = int(n_plots**0.5)
        n_cols = (n_plots + n_rows - 1) // n_rows
        return plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))


plot_manager = PlotManager()
