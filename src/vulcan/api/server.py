"""FastAPI server for VULCAN system."""

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vulcan.core import ConfigManager, VulcanOrchestrator
from vulcan.types import (
    ApiResponse,
    ErrorResponse,
    ExperimentRequest,
    HealthResponse,
    StatusResponse,
    VulcanConfig,
)
from vulcan.utils import setup_logging

# Global variables
config_manager: ConfigManager = None
orchestrator: VulcanOrchestrator = None
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global orchestrator

    # Startup
    print("✅ VULCAN system initialized successfully")
    if orchestrator:
        await orchestrator.initialize_components()

    yield

    # Shutdown
    if orchestrator:
        await orchestrator.cleanup()
    print("🔄 VULCAN system shutdown complete")


def create_app(config: VulcanConfig) -> FastAPI:
    """Create FastAPI application with configuration.

    Args:
        config: VULCAN configuration.

    Returns:
        Configured FastAPI application.
    """
    global config_manager, orchestrator

    # Create app with lifespan
    app = FastAPI(
        title="VULCAN 2.0 API",
        description="Autonomous Feature Engineering for Recommender Systems",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize components
    config_manager = ConfigManager()
    orchestrator = VulcanOrchestrator(config)

    # Add routes
    @app.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            message="VULCAN 2.0 API is running",
            version="2.0.0",
        )

    @app.websocket("/ws/exploration")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time exploration visualization."""
        try:
            # Get orchestrator instance
            if not orchestrator:
                await websocket.close(code=1011, reason="System not initialized")
                return

            # Add websocket to orchestrator
            await orchestrator.add_websocket(websocket)

            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for messages (can be used for interactive features later)
                    data = await websocket.receive_text()
                except WebSocketDisconnect:
                    break

        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
        finally:
            # Clean up connection
            if orchestrator:
                await orchestrator.remove_websocket(websocket)

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        """Get system status."""
        try:
            if not orchestrator:
                raise HTTPException(status_code=503, detail="System not initialized")

            # Get status from orchestrator
            status = orchestrator.get_status()

            return StatusResponse(
                status="running" if status.is_running else "partial",
                components=status.components_initialized,
                config_loaded=config_manager is not None,
                experiments_count=status.experiment_history_count,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/experiments", response_model=List[Dict[str, Any]])
    async def get_experiments() -> List[Dict[str, Any]]:
        """Get list of past experiments."""
        try:
            if not orchestrator:
                return []

            # Get experiment history from orchestrator
            experiment_history = orchestrator.get_experiment_history()
            experiments = []

            for i, result in enumerate(experiment_history):
                if not hasattr(result, "best_score") or result.best_score is None:
                    logger.warning(f"Incomplete experiment result at index {i}")
                    continue

                experiments.append(
                    {
                        "id": i,
                        "fold_id": result.fold_id
                        if hasattr(result, "fold_id")
                        else "unknown",
                        "iteration": i + 1,
                        "overall_score": result.best_score,
                        "feature_count": len(result.features)
                        if hasattr(result, "features") and result.features
                        else 0,
                        "evaluation_time": result.execution_time,
                        "action_taken": "experiment",
                        "features": [
                            {
                                "name": result.best_feature,
                                "type": "generated",
                                "description": f"Best feature from experiment {result.experiment_name}",
                            }
                        ]
                        if result.best_feature
                        else [],
                        "metrics": {
                            "silhouette_score": result.best_score,
                            "calinski_harabasz": result.best_score * 100,
                            "davies_bouldin": 1.0 - result.best_score,
                        },
                    }
                )

            return experiments

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/experiments/{experiment_id}", response_model=Dict[str, Any])
    async def get_experiment(experiment_id: int) -> Dict[str, Any]:
        """Get specific experiment details."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            experiment_history = orchestrator.get_experiment_history()
            if experiment_id >= len(experiment_history):
                raise HTTPException(status_code=404, detail="Experiment not found")

            result = experiment_history[experiment_id]

            if not hasattr(result, "best_score") or result.best_score is None:
                raise HTTPException(
                    status_code=404, detail="Experiment result incomplete"
                )

            return {
                "id": experiment_id,
                "fold_id": result.fold_id if hasattr(result, "fold_id") else "unknown",
                "iteration": experiment_id + 1,
                "overall_score": result.best_score,
                "improvement_over_parent": result.improvement_over_parent
                if hasattr(result, "improvement_over_parent")
                else None,
                "evaluation_time": result.execution_time,
                "feature_set": {
                    "action_taken": "experiment",
                    "parent_features": result.parent_features
                    if hasattr(result, "parent_features")
                    else [],
                    "total_cost": result.total_cost
                    if hasattr(result, "total_cost")
                    else None,
                    "features": [
                        {
                            "name": result.best_feature,
                            "type": "generated",
                            "description": f"Best feature from {result.experiment_name}",
                        }
                    ]
                    if result.best_feature
                    else [],
                },
                "metrics": {
                    "silhouette_score": result.best_score,
                    "calinski_harabasz": result.best_score * 100,
                    "davies_bouldin": 1.0 - result.best_score,
                },
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tree", response_model=Dict[str, Any])
    async def get_evolution_tree() -> Dict[str, Any]:
        """Get Progressive Evolution visualization data."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            # Check if we have an active Progressive Evolution orchestrator
            evo_orchestrator = getattr(orchestrator, "evo_orchestrator", None)
            if not evo_orchestrator:
                # Return empty evolution structure if no evolution is running
                return {
                    "population": [],
                    "generation_history": [],
                    "action_rewards": {"generate_new": [], "mutate_existing": []},
                    "best_candidate": None,
                    "stats": {
                        "current_generation": 0,
                        "population_size": 0,
                        "best_score": 0.0,
                        "total_features_generated": 0,
                    },
                }

            # Get evolution visualization data from the Progressive Evolution orchestrator
            evolution_data = await evo_orchestrator.get_evolution_visualization_data()

            return evolution_data

        except Exception as e:
            logger.error(f"Error getting Evolution data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/experiments/start")
    async def start_experiment(request: ExperimentRequest):
        """Start a new experiment with given configuration."""
        global orchestrator, config

        try:
            if not orchestrator:
                # Initialize orchestrator with proper config
                orchestrator = VulcanOrchestrator(config)
                await orchestrator.initialize_components()

            # Extract configuration overrides
            config_overrides = request.config_overrides or {}

            # Create data context based on configuration
            outer_fold = config_overrides.get("outer_fold", 1)
            inner_fold = config_overrides.get("inner_fold", 1)
            data_sample_size = config_overrides.get("data_sample_size", 5000)
            use_cache = config_overrides.get("use_cache", True)

            from vulcan.data.goodreads_loader import GoodreadsDataLoader

            loader = GoodreadsDataLoader(
                db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
                splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )

            # Load data context with specified sample size
            data_context = loader.get_data_context(sample_size=data_sample_size)

            # Start the experiment with the configuration
            experiment_id = await orchestrator.start_experiment(
                experiment_name=request.experiment_name,
                config_overrides=config_overrides,
                data_context=data_context,
            )

            return ApiResponse(status="success", data={"experiment_id": experiment_id})
        except Exception as e:
            logger.error(f"Error starting experiment: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.get("/api/data/stats")
    async def get_data_stats():
        """Get statistics about the Goodreads dataset."""
        try:
            import os
            import sqlite3

            import pandas as pd

            # Connect to the main database using absolute path
            db_path = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db"
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found: {db_path}")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get basic statistics
            stats = {}

            # Total users, books, ratings
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM reviews")
            stats["totalUsers"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT book_id) FROM books")
            stats["totalBooks"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM reviews WHERE rating IS NOT NULL")
            stats["totalRatings"] = cursor.fetchone()[0]

            # Calculate sparsity
            total_possible = stats["totalUsers"] * stats["totalBooks"]
            stats["sparsity"] = 1 - (stats["totalRatings"] / total_possible)

            # Average ratings per user and book
            stats["avgRatingsPerUser"] = stats["totalRatings"] / stats["totalUsers"]
            stats["avgRatingsPerBook"] = stats["totalRatings"] / stats["totalBooks"]

            # Rating distribution
            cursor.execute("""
                SELECT rating, COUNT(*) as count
                FROM reviews
                WHERE rating IS NOT NULL
                GROUP BY rating
                ORDER BY rating
            """)
            stats["ratingDistribution"] = [
                {"rating": row[0], "count": row[1]} for row in cursor.fetchall()
            ]

            # Get fold statistics from splits directory
            stats["foldStatistics"] = {"outer": [], "inner": []}

            # Read fold information from CSV files
            splits_dir = "/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits"

            for outer_fold in range(1, 6):
                # Read outer fold data
                train_fe_file = os.path.join(
                    splits_dir, f"outer_fold_{outer_fold}_train_fe_users.csv"
                )
                val_clusters_file = os.path.join(
                    splits_dir, f"outer_fold_{outer_fold}_val_clusters_users.csv"
                )

                if os.path.exists(train_fe_file) and os.path.exists(val_clusters_file):
                    train_fe_df = pd.read_csv(train_fe_file)
                    val_clusters_df = pd.read_csv(val_clusters_file)

                    stats["foldStatistics"]["outer"].append(
                        {
                            "fold": outer_fold,
                            "trainUsers": len(train_fe_df),
                            "valUsers": len(val_clusters_df),
                        }
                    )

                # Read inner fold data for this outer fold
                for inner_fold in range(1, 4):
                    feat_train_file = os.path.join(
                        splits_dir,
                        f"outer_fold_{outer_fold}_inner_fold_{inner_fold}_feat_train_users.csv",
                    )
                    feat_val_file = os.path.join(
                        splits_dir,
                        f"outer_fold_{outer_fold}_inner_fold_{inner_fold}_feat_val_users.csv",
                    )

                    if os.path.exists(feat_train_file) and os.path.exists(
                        feat_val_file
                    ):
                        feat_train_df = pd.read_csv(feat_train_file)
                        feat_val_df = pd.read_csv(feat_val_file)

                        stats["foldStatistics"]["inner"].append(
                            {
                                "fold": inner_fold,
                                "trainUsers": len(feat_train_df),
                                "valUsers": len(feat_val_df),
                            }
                        )

            # Year distribution
            cursor.execute("""
                SELECT publication_year, COUNT(*) as count
                FROM books 
                WHERE publication_year IS NOT NULL 
                AND publication_year > 1900 
                AND publication_year <= 2024
                GROUP BY publication_year
                ORDER BY publication_year
            """)
            year_data = cursor.fetchall()

            if year_data:
                # Group by decades for better visualization
                decade_stats = {}
                for year, count in year_data:
                    decade = int(year // 10) * 10
                    if decade not in decade_stats:
                        decade_stats[decade] = 0
                    decade_stats[decade] += count

                stats["yearDistribution"] = [
                    {"year": decade, "count": count}
                    for decade, count in sorted(decade_stats.items())
                ]
            else:
                stats["yearDistribution"] = []
                logger.warning("No valid publication year data found in database")

            conn.close()
            return ApiResponse(status="success", data=stats)

        except Exception as e:
            logger.error(f"Error getting data stats: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.get("/api/performance/summary")
    async def get_performance_summary():
        """Get comprehensive performance tracking summary."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            metrics = orchestrator.get_performance_metrics()
            return ApiResponse(status="success", data=metrics)

        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.get("/api/performance/features")
    async def get_feature_performance(
        feature_name: Optional[str] = None, top_k: int = 10, criteria: str = "avg_score"
    ):
        """Get feature performance metrics."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            if feature_name:
                # Get specific feature performance
                metrics = orchestrator.get_feature_performance(feature_name)
                if not metrics:
                    raise HTTPException(status_code=404, detail="Feature not found")
                return ApiResponse(status="success", data=metrics)
            else:
                # Get best performing features
                best_features = orchestrator.get_best_features(top_k, criteria)
                return ApiResponse(
                    status="success",
                    data={
                        "best_features": best_features,
                        "criteria": criteria,
                        "count": len(best_features),
                    },
                )

        except Exception as e:
            logger.error(f"Error getting feature performance: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.get("/api/performance/suggestions")
    async def get_performance_suggestions():
        """Get AI-powered feature performance suggestions."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            suggestions = orchestrator.get_feature_suggestions()
            return ApiResponse(status="success", data=suggestions)

        except Exception as e:
            logger.error(f"Error getting performance suggestions: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.get("/api/performance/export")
    async def export_performance_data():
        """Export complete performance tracking data for analysis."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            export_data = orchestrator.export_performance_data()
            return ApiResponse(status="success", data=export_data)

        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message=str(exc),
                details={"type": type(exc).__name__},
            ).dict(),
        )

    return app


# Create app instance at module level
# Load default configuration
try:
    config_manager = ConfigManager()
    config = config_manager.config
    setup_logging(config.logging)
    app = create_app(config)
except Exception as e:
    # Create a minimal app if configuration fails
    logger.warning(f"Failed to load configuration: {e}. Creating minimal app.")
    from vulcan.types import VulcanConfig

    app = create_app(VulcanConfig())


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="VULCAN 2.0 API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.config

        # Setup logging
        setup_logging(config.logging)

        # Create app
        app = create_app(config)

        print("🚀 Starting VULCAN 2.0 API Server")
        print(f"📍 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"📚 Docs: http://{args.host}:{args.port}/docs")
        print(f"🔄 Reload: {args.reload}")

        # Run server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # When running directly, use the main function
    main()
