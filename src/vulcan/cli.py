"""Command-line interface for VULCAN system, rewritten for clarity and robustness."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

# It's better to handle potential import errors for key components
try:
    from vulcan import __version__
    from vulcan.api import create_app
    from vulcan.core import ConfigManager
    from vulcan.utils import get_vulcan_logger, setup_logging
except ImportError as e:
    print(
        f"Error: A core VULCAN module is missing. Please check your installation. Details: {e}"
    )
    # Use typer.Exit if Typer is available, otherwise regular sys.exit
    try:
        raise typer.Exit(code=1)
    except NameError:
        import sys

        sys.exit(1)


# --- Typer App Setup ---
# Rename `app` to `main` to match the expected entry point name from the .venv/bin/vulcan script
main = typer.Typer(
    name="vulcan",
    help="VULCAN: Autonomous Feature Engineering for Recommender Systems",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# --- App-level callbacks (e.g., for --version) ---
def version_callback(value: bool):
    """Prints the version and exits."""
    if value:
        console.print(f"VULCAN version: [bold green]{__version__}[/bold green]")
        raise typer.Exit()


@main.callback()
def main_callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application version and exit.",
        callback=version_callback,
        is_eager=True,  # Process this before any command
    ),
):
    """
    VULCAN CLI main entry point.
    Handles global options like --version.
    """
    # This callback runs before any command.
    # The version logic is handled by the dedicated version_callback.
    pass


# --- CLI Commands ---


@main.command()
def serve(
    config_path: Path = typer.Option(
        "config/dev.yaml",  # Default config path
        "--config",
        "-c",
        help="Path to the configuration file.",
        resolve_path=True,  # Ensure path is absolute
        show_default=True,
    ),
    host: Optional[str] = typer.Option(
        None, "--host", "-h", help="Host to bind to (overrides config)."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Port to bind to (overrides config)."
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload (overrides config and is not recommended for production).",
    ),
):
    """
    Start the VULCAN API server.
    """
    console.print(
        f"-> Attempting to load configuration from: [cyan]{config_path}[/cyan]"
    )

    if not config_path.is_file():
        console.print(
            f"❌ [bold red]Configuration file not found![/bold red] Path does not exist or is not a file: '{config_path}'"
        )
        raise typer.Exit(code=1)

    try:
        # 1. Load configuration from the specified file.
        config_manager = ConfigManager(config_path=config_path)
        config = config_manager.config
        console.print(
            f"✅ Configuration loaded successfully from: [cyan]{config_manager._config_path}[/cyan]"
        )

        # 2. Set up logging based on the loaded configuration.
        setup_logging(config.logging)
        logger = get_vulcan_logger(__name__ + ".serve")

        # 3. Override config with command-line arguments if they were provided.
        # CLI options take ultimate precedence over the config file.
        effective_host = host if host is not None else config.api.host
        effective_port = port if port is not None else config.api.port
        # For a boolean flag, its presence on the CLI means True.
        effective_reload = reload or config.api.reload

        # Update the config object in case other parts of the app use it during server runtime
        config.api.host = effective_host
        config.api.port = effective_port
        config.api.reload = effective_reload

        # 4. Create the FastAPI app instance using the final effective config
        fastapi_app = create_app(config)

        # 5. Start the server using Uvicorn.
        logger.info(
            f"🚀 Starting VULCAN API server on http://{effective_host}:{effective_port}",
            reload_enabled=effective_reload,
        )

        # Uvicorn needs the app as an import string for reload mode to work.
        # This assumes the FastAPI app instance is named `app` in `vulcan.api.server`.
        app_import_string = "vulcan.api.server:app"

        if effective_reload:
            # When reloading, pass the import string.
            uvicorn.run(
                app_import_string,
                host=effective_host,
                port=effective_port,
                reload=True,
                log_config=None,
            )
        else:
            # When not reloading, create the app object here and pass it directly.
            fastapi_app = create_app(config)
            uvicorn.run(
                fastapi_app,
                host=effective_host,
                port=effective_port,
                reload=False,
                log_config=None,
            )

        logger.info("🛑 VULCAN API server has stopped.")

    except Exception:
        # This will catch errors during config loading, app creation, or server run.
        console.print(
            "❌ [bold red]An unexpected error occurred while trying to start the server:[/bold red]"
        )
        console.print_exception()
        raise typer.Exit(code=1)


@main.command()
def experiment(
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="A descriptive name for the experiment run."
    ),
    config_path: Path = typer.Option(
        "config/dev.yaml",
        "--config",
        "-c",
        help="Path to the configuration file for the experiment.",
        resolve_path=True,
        show_default=True,
    ),
):
    """
    Run a VULCAN feature engineering experiment from the command line.
    """
    console.print(
        f"-> Preparing to run experiment. Loading config from: [cyan]{config_path}[/cyan]"
    )
    if not config_path.is_file():
        console.print(
            f"❌ [bold red]Configuration file not found![/bold red] Path: '{config_path}'"
        )
        raise typer.Exit(code=1)

    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.config
        setup_logging(config.logging)
        local_logger = get_vulcan_logger(__name__ + ".experiment")
        local_logger.info("🧪 Starting VULCAN experiment command...")

        # This is an async function, so we run it with asyncio
        asyncio.run(_run_experiment(config, name, local_logger))

    except Exception:
        console.print("❌ [bold red]Experiment command failed unexpectedly:[/bold red]")
        console.print_exception()
        raise typer.Exit(code=1)


async def _run_experiment(config_obj, experiment_name: Optional[str], logger_instance):
    """Helper async function to set up and run the orchestrator."""
    from vulcan.core import VulcanOrchestrator

    logger_instance.info("Initializing orchestrator for experiment.")
    orchestrator = VulcanOrchestrator(config_obj)

    try:
        # Initialize components (data layer, agents, etc.)
        await orchestrator.initialize_components()

        # Start the experiment run
        logger_instance.info(
            f"Attempting to start experiment: {experiment_name or 'default_name'}"
        )
        experiment_id = await orchestrator.start_experiment(experiment_name)
        logger_instance.info(
            f"🚀 Experiment '{experiment_name}' is running with ID: {experiment_id}"
        )

        # In a real CLI run, we might want to tail logs or show a progress bar.
        # For now, we just wait for it to finish.
        while orchestrator.get_status().is_running:
            await asyncio.sleep(2)

        logger_instance.info(f"✅ Experiment {experiment_id} has completed.")

        # Display a summary of the results
        history = orchestrator.get_experiment_history()
        if history:
            result = history[-1]
            table = Table(title=f"Experiment '{result.experiment_name}' Summary")
            table.add_column("Metric", style="cyan", justify="right")
            table.add_column("Value", style="magenta")
            table.add_row("Best Score", f"{result.best_score:.6f}")
            table.add_row("Total Generations", str(result.total_iterations))
            table.add_row("Execution Time", f"{result.execution_time:.2f} seconds")

            best_features_str = ", ".join(
                [f.get("name", "N/A") for f in result.best_features or []]
            )
            table.add_row("Best Feature(s)", best_features_str or "N/A")

            console.print(table)
        else:
            logger_instance.warning(
                "Experiment finished, but no results were found in the history."
            )

    except Exception:
        logger_instance.error(
            "An exception occurred during the experiment run.", exc_info=True
        )
        raise  # Re-raise to be caught by the main command's handler
    finally:
        logger_instance.info("Cleaning up orchestrator resources...")
        await orchestrator.cleanup()
        logger_instance.info("Cleanup complete.")


# This allows running the CLI by executing `python -m src.vulcan.cli`
if __name__ == "__main__":
    main()
