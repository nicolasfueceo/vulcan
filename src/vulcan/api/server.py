"""FastAPI server for VULCAN system."""

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
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
from vulcan.utils import ResultsManager, setup_logging

# Global variables
config_manager: ConfigManager = None
orchestrator: VulcanOrchestrator = None
results_manager: ResultsManager = None
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global orchestrator, results_manager

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
    global config_manager, orchestrator, results_manager

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
    results_manager = ResultsManager(config)

    # Add routes
    @app.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            message="VULCAN 2.0 API is running",
            version="2.0.0",
        )

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
                experiments_count=len(results_manager.list_experiments()),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/experiments", response_model=List[Dict[str, Any]])
    async def get_experiments() -> List[Dict[str, Any]]:
        """Get list of past experiments."""
        try:
            if not results_manager:
                return []

            # Get experiments from results manager
            experiments = results_manager.list_experiments()

            # Transform to expected format
            formatted_experiments = []
            for i, exp in enumerate(experiments):
                formatted_experiments.append(
                    {
                        "id": i,
                        "experiment_name": exp.get("experiment_name", "unknown"),
                        "algorithm": exp.get("algorithm", "unknown"),
                        "start_time": exp.get("start_time", ""),
                        "status": exp.get("status", "unknown"),
                        "iterations_completed": exp.get("iterations_completed", 0),
                        "best_score": exp.get("best_score", 0.0),
                        "end_time": exp.get("end_time", ""),
                    }
                )

            return formatted_experiments

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/experiments/{experiment_name}/data")
    async def get_experiment_data(experiment_name: str) -> Dict[str, Any]:
        """Get experiment visualization data."""
        try:
            if not results_manager:
                raise HTTPException(
                    status_code=503, detail="Results manager not initialized"
                )

            experiment_data = results_manager.load_experiment_data(experiment_name)
            if not experiment_data:
                raise HTTPException(status_code=404, detail="Experiment not found")

            return ApiResponse(status="success", data=experiment_data).dict()

        except Exception as e:
            logger.error(f"Error getting experiment data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/experiments/latest/data")
    async def get_latest_experiment_data() -> Dict[str, Any]:
        """Get data from the most recent experiment."""
        try:
            if not results_manager:
                raise HTTPException(
                    status_code=503, detail="Results manager not initialized"
                )

            experiment_data = results_manager.get_latest_experiment_data()
            if not experiment_data:
                return ApiResponse(
                    status="success",
                    data={
                        "nodes": [],
                        "edges": [],
                        "stats": {
                            "total_nodes": 0,
                            "max_depth": 0,
                            "best_score": 0.0,
                            "iterations_completed": 0,
                            "avg_branching_factor": 0.0,
                        },
                        "generation_history": [],
                        "action_rewards": {"generate_new": [], "mutate_existing": []},
                        "best_candidate": None,
                    },
                ).dict()

            return ApiResponse(status="success", data=experiment_data).dict()

        except Exception as e:
            logger.error(f"Error getting latest experiment data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/experiments/start")
    async def start_experiment(request: ExperimentRequest):
        """Start a new experiment with given configuration."""
        global orchestrator, config, results_manager

        try:
            if not orchestrator:
                # Initialize orchestrator with proper config
                orchestrator = VulcanOrchestrator(config)
                await orchestrator.initialize_components()

            # Extract configuration overrides
            config_overrides = request.config_overrides or {}

            # Start experiment tracking in results manager
            experiment_id = f"evolution_{request.experiment_name}"
            experiment_metadata = {
                "algorithm": "evolution",
                "experiment_name": request.experiment_name,
                "config_overrides": config_overrides,
            }

            experiment_dir = results_manager.start_experiment(
                experiment_id, experiment_metadata
            )

            # Create data context based on configuration
            outer_fold = config_overrides.get("outer_fold", 1)
            inner_fold = config_overrides.get("inner_fold", 1)
            data_sample_size = config_overrides.get("data_sample_size", 5000)

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
            experiment_result_id = await orchestrator.start_experiment(
                experiment_name=request.experiment_name,
                config_overrides=config_overrides,
                data_context=data_context,
                results_manager=results_manager,  # Pass results manager
            )

            return ApiResponse(
                status="success", data={"experiment_id": experiment_result_id}
            )
        except Exception as e:
            logger.error(f"Error starting experiment: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.post("/api/mcts/start")
    async def start_mcts_experiment(request: ExperimentRequest):
        """Start a new MCTS experiment with given configuration."""
        global orchestrator, config, results_manager

        try:
            if not orchestrator:
                # Initialize orchestrator with proper config
                orchestrator = VulcanOrchestrator(config)
                await orchestrator.initialize_components()

            # Extract configuration overrides
            config_overrides = request.config_overrides or {}

            # Update config for MCTS
            config.mcts.max_iterations = config_overrides.get("max_iterations", 10)
            config.mcts.max_depth = config_overrides.get("max_depth", 5)
            config.mcts.exploration_factor = config_overrides.get(
                "exploration_factor", 1.4
            )
            config.llm.temperature = config_overrides.get("llm_temperature", 0.7)
            config.llm.model_name = config_overrides.get("llm_model", "gpt-4o-mini")

            # Start experiment tracking in results manager
            experiment_id = f"mcts_{request.experiment_name}"
            experiment_metadata = {
                "algorithm": "mcts",
                "experiment_name": request.experiment_name,
                "config_overrides": config_overrides,
            }

            experiment_dir = results_manager.start_experiment(
                experiment_id, experiment_metadata
            )

            # Create data context based on configuration
            outer_fold = config_overrides.get("outer_fold", 1)
            inner_fold = config_overrides.get("inner_fold", 1)
            data_sample_size = config_overrides.get("data_sample_size", 20000)

            from vulcan.data.goodreads_loader import GoodreadsDataLoader

            loader = GoodreadsDataLoader(
                db_path="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/goodreads.db",
                splits_dir="/Users/nicolasdhnr/Documents/Imperial/Imperial Thesis/Code/VULCAN/data/splits",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )

            # Load data context
            data_context = loader.get_data_context(sample_size=data_sample_size)

            # Create MCTS orchestrator
            from vulcan.mcts.mcts_orchestrator import MCTSOrchestrator
            from vulcan.utils import PerformanceTracker

            performance_tracker = PerformanceTracker(max_history=100)
            mcts_orchestrator = MCTSOrchestrator(config, performance_tracker)
            await mcts_orchestrator.initialize()

            # Store reference for later access
            orchestrator.mcts_orchestrator = mcts_orchestrator

            # Start the MCTS search asynchronously and save results
            import asyncio

            async def run_mcts():
                results = await mcts_orchestrator.run_search(
                    data_context=data_context,
                    max_iterations=config.mcts.max_iterations,
                    results_manager=results_manager,  # Pass results manager
                )
                # Mark experiment as finished
                results_manager.finish_experiment({"final_results": results})

            # Run in background
            asyncio.create_task(run_mcts())

            experiment_result_id = (
                f"mcts_{request.experiment_name}_{outer_fold}_{inner_fold}"
            )

            return ApiResponse(
                status="success", data={"experiment_id": experiment_result_id}
            )
        except Exception as e:
            logger.error(f"Error starting MCTS experiment: {str(e)}")
            return ApiResponse(status="error", message=str(e))

    @app.post("/api/experiments/stop")
    async def stop_experiment():
        """Stop the current experiment."""
        try:
            if not orchestrator:
                raise HTTPException(
                    status_code=503, detail="Orchestrator not initialized"
                )

            # Stop any running experiments
            if (
                hasattr(orchestrator, "evo_orchestrator")
                and orchestrator.evo_orchestrator
            ):
                await orchestrator.evo_orchestrator.cleanup()
                orchestrator.evo_orchestrator = None

            if (
                hasattr(orchestrator, "mcts_orchestrator")
                and orchestrator.mcts_orchestrator
            ):
                await orchestrator.mcts_orchestrator.cleanup()
                orchestrator.mcts_orchestrator = None

            # Finish current experiment in results manager
            if results_manager:
                results_manager.finish_experiment()

            return ApiResponse(status="success", message="Experiment stopped")
        except Exception as e:
            logger.error(f"Error stopping experiment: {str(e)}")
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
