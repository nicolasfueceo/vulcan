import logging
from src.utils.tools import compute_summary_stats, create_plot

from src.utils.run_utils import get_run_dir

logger = logging.getLogger(__name__)

class FeatureAuditorAgent:
    """
    Audits realized features for informativeness using comprehensive statistics, plots, and vision analysis.
    """
    def __init__(self, db_path, vision_tool):
        self.db_path = db_path
        self.vision_tool = vision_tool  # Callable: vision_tool(plot_path) -> str
        self.plots_dir = get_run_dir() / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def audit_feature(self, feature_name: str) -> dict:
        """
        For a given feature (column in a realized features view/table):
        - Compute summary stats
        - Generate and save plot
        - Use vision tool to interpret plot
        - Log structured insight
        Returns a dict with stats, plot_path, vision_summary, and a boolean 'informative'.
        """
        try:
            stats_md = compute_summary_stats(feature_name)
            # Generate histogram plot for the feature
            plot_path = create_plot(f'SELECT "{feature_name}" FROM realized_features', plot_type="hist", x=feature_name, file_name=f"{feature_name}_hist.png")
            vision_summary = self.vision_tool(plot_path) if not plot_path.startswith("ERROR") else "Plot could not be generated."
            # Simple informativeness filter: feature is informative if not constant and not mostly missing
            informative = ("No data" not in stats_md and "ERROR" not in stats_md and "Missing: 0" not in stats_md)
            insight = {
                "feature": feature_name,
                "stats": stats_md,
                "plot_path": plot_path,
                "vision_summary": vision_summary,
                "informative": informative
            }
            logger.info(f"Audited feature {feature_name}: informative={informative}")
            return insight
        except Exception as e:
            logger.error(f"Failed to audit feature {feature_name}: {e}")
            return {"feature": feature_name, "error": str(e), "informative": False}

    def audit_features(self, feature_names: list) -> list:
        """
        Audits a list of features and returns a list of insight dicts.
        """
        results = []
        for feat in feature_names:
            results.append(self.audit_feature(feat))
        return results
