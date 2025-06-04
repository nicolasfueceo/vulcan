#!/usr/bin/env python3
"""
VULCAN System Startup Script

This script starts both the backend API server and the frontend dashboard.
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests


def check_port(port: int) -> bool:
    """Check if a port is available."""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=1)
        return True  # Port is in use
    except requests.exceptions.RequestException:
        return False  # Port is available


def start_backend():
    """Start the VULCAN backend API server."""
    print("🚀 Starting VULCAN backend API server...")

    # Check if backend is already running
    if check_port(8000):
        print("✅ Backend already running on port 8000")
        return None

    # Start backend
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.vulcan.api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]

    try:
        backend_process = subprocess.Popen(
            backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_port(8000):
                print("✅ Backend started successfully on http://localhost:8000")
                return backend_process
            time.sleep(1)

        print("❌ Backend failed to start within 30 seconds")
        backend_process.terminate()
        return None

    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None


def start_frontend():
    """Start the Next.js frontend."""
    print("🎨 Starting VULCAN frontend dashboard...")

    # Check if frontend is already running
    if check_port(3000):
        print("✅ Frontend already running on port 3000")
        return None

    # Change to frontend directory
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None

    # Start frontend
    frontend_cmd = ["npm", "run", "dev"]

    try:
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for frontend to start
        print("⏳ Waiting for frontend to start...")
        for i in range(60):  # Wait up to 60 seconds for Next.js
            if check_port(3000):
                print("✅ Frontend started successfully on http://localhost:3000")
                return frontend_process
            time.sleep(1)

        print("❌ Frontend failed to start within 60 seconds")
        frontend_process.terminate()
        return None

    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None


def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="VULCAN System Startup")
    parser.add_argument(
        "--backend-only", action="store_true", help="Start only the backend API server"
    )
    parser.add_argument(
        "--frontend-only", action="store_true", help="Start only the frontend dashboard"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )

    args = parser.parse_args()

    print("🧬 VULCAN Progressive Evolution System")
    print("=" * 50)

    backend_process = None
    frontend_process = None

    try:
        # Start backend
        if not args.frontend_only:
            backend_process = start_backend()
            if not backend_process and not check_port(8000):
                print("❌ Failed to start backend. Exiting.")
                return

        # Start frontend
        if not args.backend_only:
            frontend_process = start_frontend()
            if not frontend_process and not check_port(3000):
                print("❌ Failed to start frontend. Exiting.")
                if backend_process:
                    backend_process.terminate()
                return

        print("\n🎉 VULCAN system started successfully!")
        print("\n📊 Access points:")
        if not args.frontend_only:
            print("  • Backend API: http://localhost:8000")
            print("  • API Docs: http://localhost:8000/docs")
        if not args.backend_only:
            print("  • Frontend Dashboard: http://localhost:3000")

        print("\n💡 Quick start:")
        print("  • Use the web dashboard at http://localhost:3000")
        print("  • Or run: python experiment_runner.py list")
        print("  • Or run: python experiment_runner.py run quick")

        # Open browser
        if not args.no_browser and not args.backend_only:
            print("\n🌐 Opening browser...")
            webbrowser.open("http://localhost:3000")

        print("\n⏹️  Press Ctrl+C to stop all services")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down VULCAN system...")

    except KeyboardInterrupt:
        print("\n🛑 Shutting down VULCAN system...")

    finally:
        # Clean up processes
        if backend_process:
            print("⏹️  Stopping backend...")
            backend_process.terminate()
            backend_process.wait()

        if frontend_process:
            print("⏹️  Stopping frontend...")
            frontend_process.terminate()
            frontend_process.wait()

        print("✅ VULCAN system stopped")


if __name__ == "__main__":
    main()
