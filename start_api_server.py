#!/usr/bin/env python3
"""Start the VULCAN API server with file-based experiment results."""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn

from vulcan.api.server import create_app
from vulcan.core import ConfigManager


def main():
    """Start the API server."""
    print("🚀 Starting VULCAN API Server with File-Based Results")
    print("=" * 60)

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.config

    print("✅ Configuration loaded")
    print(
        f"📊 Database: {config.data.db_path if hasattr(config.data, 'db_path') else 'Goodreads'}"
    )
    print(f"🤖 LLM: {config.llm.model_name}")
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")

    # Create FastAPI app
    app = create_app(config)

    print("\n🌐 Starting server...")
    print("💡 Tip: Open http://localhost:3000 in your browser for the UI")
    print("📁 Experiment results will be saved to ./results/ directory")
    print("🔄 Press Ctrl+C to stop")
    print("-" * 60)

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Disable reload for production-like behavior
    )


if __name__ == "__main__":
    main()
