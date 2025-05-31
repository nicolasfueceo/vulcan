"""Command-line interface for VULCAN system."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from vulcan.api import create_app
from vulcan.core import ConfigManager
from vulcan.utils import setup_logging

app = typer.Typer(
    name="vulcan",
    help="VULCAN: Autonomous Feature Engineering for Recommender Systems",
    add_completion=False,
)

console = Console()

# Constants
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000


@app.command()
def serve(
    host: str = typer.Option(DEFAULT_HOST, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Start the VULCAN API server."""
    try:
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.config

        # Setup logging
        setup_logging(config.logging)

        # Override API config if provided
        if host != DEFAULT_HOST:
            config.api.host = host
        if port != DEFAULT_PORT:
            config.api.port = port
        if reload:
            config.api.reload = reload

        console.print(f"🚀 Starting VULCAN API server on {host}:{port}")

        # Create and run FastAPI app
        fastapi_app = create_app(config)

        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            log_config=None,  # Use our custom logging
        )

    except Exception as e:
        console.print(f"❌ Failed to start server: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def config(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for configuration"
    ),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
) -> None:
    """Manage VULCAN configuration."""
    try:
        config_manager = ConfigManager(config_path)

        if validate:
            is_valid = config_manager.validate_config()
            if is_valid:
                console.print("✅ Configuration is valid", style="green")
            else:
                console.print("❌ Configuration validation failed", style="red")
                raise typer.Exit(1)
            return

        if output:
            config_manager.save_config(output)
            console.print(f"💾 Configuration saved to {output}")
        else:
            _display_config(config_manager)

    except Exception as e:
        console.print(f"❌ Configuration error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def status(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Show VULCAN system status."""
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.config

        # Display system status
        table = Table(title="VULCAN System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Configuration status
        table.add_row(
            "Configuration",
            "✅ Loaded" if config else "❌ Failed",
            f"From: {config_manager._config_path}",
        )

        # API status
        api_status = "✅ Enabled" if config.api.enabled else "⚠️ Disabled"
        table.add_row(
            "API Server",
            api_status,
            f"{config.api.host}:{config.api.port}",
        )

        # LLM status
        llm_status = (
            f"✅ {config.llm.provider}" if config.llm.provider else "❌ Not configured"
        )
        table.add_row(
            "LLM Provider",
            llm_status,
            config.llm.model_name,
        )

        # Experiment tracking status
        tracking_status = (
            "✅ Enabled" if config.experiment.wandb_enabled else "⚠️ Disabled"
        )
        table.add_row(
            "Experiment Tracking",
            tracking_status,
            config.experiment.wandb_project,
        )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def experiment(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Experiment name"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-i", help="Number of MCTS iterations"
    ),
) -> None:
    """Run a VULCAN experiment."""
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.config

        # Override iterations if provided
        if iterations:
            config.mcts.max_iterations = iterations

        console.print("🧪 Starting VULCAN experiment...")

        # Run experiment asynchronously
        asyncio.run(_run_experiment(config, name))

    except Exception as e:
        console.print(f"❌ Experiment failed: {e}", style="red")
        raise typer.Exit(1)


async def _run_experiment(config, experiment_name: Optional[str]) -> None:
    """Run experiment asynchronously."""
    from vulcan.core import VulcanOrchestrator

    # Setup logging
    setup_logging(config.logging)

    # Initialize orchestrator
    orchestrator = VulcanOrchestrator(config)
    await orchestrator.initialize_components()

    try:
        # Start experiment
        experiment_id = await orchestrator.start_experiment(experiment_name)
        console.print(f"🚀 Experiment started: {experiment_id}")

        # Wait for completion (in real implementation, this would be more sophisticated)
        while orchestrator.get_status().is_running:
            await asyncio.sleep(1)

        # Get results
        history = orchestrator.get_experiment_history()
        if history:
            result = history[-1]
            console.print("✅ Experiment completed!")
            console.print(f"   Best Score: {result.best_score:.4f}")
            console.print(f"   Iterations: {result.total_iterations}")
            console.print(f"   Time: {result.execution_time:.2f}s")

    finally:
        await orchestrator.cleanup()


def _display_config(config_manager: ConfigManager) -> None:
    """Display configuration in a formatted table."""
    config = config_manager.config

    table = Table(title="VULCAN Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="green")

    # MCTS settings
    table.add_row("MCTS", "Max Iterations", str(config.mcts.max_iterations))
    table.add_row("MCTS", "Exploration Factor", str(config.mcts.exploration_factor))
    table.add_row("MCTS", "Max Depth", str(config.mcts.max_depth))

    # LLM settings
    table.add_row("LLM", "Provider", config.llm.provider)
    table.add_row("LLM", "Model", config.llm.model_name)
    table.add_row("LLM", "Temperature", str(config.llm.temperature))

    # API settings
    table.add_row("API", "Enabled", str(config.api.enabled))
    table.add_row("API", "Host", config.api.host)
    table.add_row("API", "Port", str(config.api.port))

    # Experiment settings
    table.add_row("Experiment", "W&B Enabled", str(config.experiment.wandb_enabled))
    table.add_row("Experiment", "W&B Project", config.experiment.wandb_project)

    console.print(table)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
) -> None:
    """VULCAN: Autonomous Feature Engineering for Recommender Systems."""
    if version:
        from vulcan import __version__

        console.print(f"VULCAN v{__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
