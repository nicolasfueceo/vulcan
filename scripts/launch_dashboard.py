import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path


def print_header(title):
    print("\\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def start_service_and_log(command, name, cwd, log_file_path, env=None):
    print_header(f"Starting {name}")
    print(f"  - Command: {' '.join(command)}")
    print(f"  - Log File: {log_file_path}")

    # Open log file for writing
    log_file = open(log_file_path, "w")

    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=log_file,
        text=True,
        cwd=cwd,
        env=env,
        preexec_fn=os.setsid,
    )
    print(f"✅ {name} started with PID: {process.pid}")
    return process, log_file


async def tail_log(name, log_file_path):
    """Async function to tail a log file and print its output."""
    print(f"  -> Tailing logs for {name}...")
    with open(log_file_path) as f:
        # Go to the end of the file
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.1)  # Sleep briefly
                continue
            print(f"[{name}] {line.strip()}")


async def check_backend_health(url="http://localhost:8000/api/health", timeout=60):
    """Check if the backend API is healthy."""
    import requests

    print_header("Checking Backend Health")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=3)
            if (
                response.status_code == 200
                and response.json().get("status") == "healthy"
            ):
                print("✅ Backend is healthy and responding.")
                return True
        except requests.ConnectionError:
            pass  # Server not up yet
        except Exception:
            pass  # Other errors
        await asyncio.sleep(2)
    print("❌ Backend did not become healthy in time.")
    return False


async def main():
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"
    experiments_dir = project_root / "experiments"
    logs_dir = project_root / "logs"

    # Create directories
    experiments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Define log file paths
    backend_log = logs_dir / "backend.log"
    frontend_log = logs_dir / "frontend.log"
    tensorboard_log = logs_dir / "tensorboard.log"

    processes_and_files = []
    try:
        # 1. Start Services and log to files
        backend_command = [
            "python3",
            "-m",
            "vulcan.cli",
            "serve",
            "--config",
            "config/dev.yaml",
            "--reload",
        ]
        backend_process, bf = start_service_and_log(
            backend_command, "Backend", project_root, backend_log
        )
        processes_and_files.append(
            {"name": "Backend", "process": backend_process, "file": bf}
        )

        frontend_command = ["npm", "run", "dev"]
        frontend_process, ff = start_service_and_log(
            frontend_command, "Frontend", frontend_dir, frontend_log
        )
        processes_and_files.append(
            {"name": "Frontend", "process": frontend_process, "file": ff}
        )

        tensorboard_command = [
            "tensorboard",
            "--logdir",
            str(experiments_dir),
            "--port",
            "6006",
        ]
        tensorboard_process, tf = start_service_and_log(
            tensorboard_command, "TensorBoard", project_root, tensorboard_log
        )
        processes_and_files.append(
            {"name": "TensorBoard", "process": tensorboard_process, "file": tf}
        )

        print("\\n" + "*" * 80)
        print("✅ All services are launching.")
        print(f"   - Backend logs: {backend_log}")
        print(f"   - Frontend logs: {frontend_log}")
        print(f"   - TensorBoard logs: {tensorboard_log}")
        print("\\n")

        # 2. Check health and tail logs
        if not await check_backend_health():
            raise Exception(
                "Backend failed to start, check logs/backend.log for details."
            )

        print_header("Tailing Logs (Press Ctrl+C to stop all services)")
        # Create tailing tasks
        tail_tasks = [
            tail_log("Backend", backend_log),
            tail_log("Frontend", frontend_log),
            tail_log("TensorBoard", tensorboard_log),
        ]
        await asyncio.gather(*tail_tasks)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\\n🚫 Keyboard interrupt received. Shutting down all services...")
    except Exception as e:
        print(f"\\n❌ An error occurred: {e}")
    finally:
        for p_info in processes_and_files:
            process = p_info["process"]
            name = p_info["name"]
            log_file = p_info["file"]
            if process.poll() is None:
                print(f"Terminating {name} (PID: {process.pid})...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} terminated.")
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    print(f"✅ {name} killed.")
            log_file.close()  # Close the log file handle
        print("\\nAll services have been shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nDashboard launch cancelled by user.")
