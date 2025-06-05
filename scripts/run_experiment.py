import asyncio
import sys
from pathlib import Path

import structlog
import yaml

from vulcan.core.orchestrator import VulcanOrchestrator, load_presets
from vulcan.schemas import VulcanConfig

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def main():
    """Main entry point to run a VULCAN experiment from a preset."""
    if len(sys.argv) < 2:
        logger.error("Usage: python3 scripts/run_experiment.py <preset_name>")
        sys.exit(1)

    preset_name = sys.argv[1]
    logger.info(f"🚀 Starting experiment for preset: {preset_name}")

    # --- Load Configurations ---
    try:
        # Load base configuration
        config_path = Path(__file__).resolve().parents[1] / "configs" / "vulcan.yaml"
        with open(config_path) as f:
            base_config_dict = yaml.safe_load(f)
        config = VulcanConfig(**base_config_dict)
        logger.info("✅ Base configuration loaded successfully.")

        # Load all available presets
        presets = load_presets()
        if preset_name not in presets:
            logger.error(
                f"Preset '{preset_name}' not found.",
                available_presets=list(presets.keys()),
            )
            sys.exit(1)
        preset_config_dict = presets[preset_name]["config"]
        logger.info(f"✅ Preset '{preset_name}' loaded successfully.")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}", file_path=e.filename)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configurations: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize and Run Orchestrator ---
    orchestrator = VulcanOrchestrator(config)
    try:
        logger.info("🔧 Initializing and running the orchestrator...")
        await orchestrator.run_single_experiment_from_preset(
            preset_name, preset_config_dict
        )
        logger.info(f"🎉 Experiment '{preset_name}' completed successfully.")
    except Exception as e:
        logger.error(
            f"❌ An error occurred during the experiment execution for preset '{preset_name}'.",
            error=str(e),
            exc_info=True,
        )
    finally:
        logger.info("🧹 Cleaning up orchestrator resources...")
        await orchestrator.cleanup()
        logger.info("✅ Cleanup complete. Exiting.")


if __name__ == "__main__":
    # Set the Python path to include the project root
    # This allows for absolute imports from 'vulcan'
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    asyncio.run(main())
