#!/usr/bin/env python3
"""System validation script for VULCAN 2.0."""

import asyncio
import json
import subprocess
import time
from pathlib import Path

import requests  # For making HTTP requests
from rich.console import Console

console = Console()

# Configuration
VULCAN_CLI_PATH = "vulcan"  # Assumes vulcan is in PATH
# Make CONFIG_FILE path relative to the project root, assuming script is in VULCAN_ROOT/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = str(PROJECT_ROOT / "config" / "dev.yaml")
SERVER_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{SERVER_URL}/api/health"
START_EXPERIMENT_ENDPOINT = f"{SERVER_URL}/api/experiments/start"

SMALL_EXPERIMENT_CONFIG = {
    "experimentName": "system_test_ucb_evo_v2",
    "algorithm": "evolution",
    "max_generations": 1,  # Minimal generations
    "population_size": 3,
    "generation_size": 2,
    "data_sample_size": 500,  # Minimal data
    "llm": {"provider": "local"},
    "evaluation": {  # Add minimal evaluation config
        "sample_size": 500,
        "scoring_mode": "cluster",
    },
}


def print_header(title):
    print("\n" + "=" * 60)
    print(f"ੲ {title}")
    print("=" * 60)


def print_status(message, success=True):
    prefix = "✅" if success else "❌"
    print(f"{prefix} {message}")


def start_server():
    print_header("Starting VULCAN Server")
    print(f"Using config file: {CONFIG_FILE}")
    command = [
        VULCAN_CLI_PATH,
        "serve",
        "--config",
        CONFIG_FILE,
        "--reload",
    ]  # Added --reload for dev
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT,
        )
        print_status(f"Server process starting with PID: {process.pid}...")
        time.sleep(5)  # Wait a few seconds for initial output

        # Check for immediate errors
        try:
            stdout, stderr = process.communicate(
                timeout=1
            )  # Non-blocking read if possible
            if stdout:
                print("--- Initial Server STDOUT ---")
                print(stdout.strip())
            if stderr:
                print("--- Initial Server STDERR ---")
                print(stderr.strip())
                if "Traceback" in stderr or "Error" in stderr:
                    print_status(
                        "Potential server startup error detected in initial stderr!",
                        success=False,
                    )
        except subprocess.TimeoutExpired:  # Expected if server is running fine
            print("Server process running, no immediate errors from communicate().")
        except Exception as e:
            print(f"Error trying to get initial server output: {e}")

        if process.poll() is not None:  # Check if process terminated quickly
            print_status(
                f"Server process terminated unexpectedly with code {process.returncode}.",
                success=False,
            )
            stdout, stderr = process.communicate()  # Get all output
            if stdout:
                print(f"STDOUT:\n{stdout.strip()}")
            if stderr:
                print(f"STDERR:\n{stderr.strip()}")
            return None

        return process
    except FileNotFoundError:
        print_status(f"Error: '{VULCAN_CLI_PATH}' command not found.", success=False)
        return None
    except Exception as e:
        print_status(f"Failed to start server: {e}", success=False)
        return None


async def wait_for_server(timeout=60):
    print_header("Waiting for Server to be Ready")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    print_status("Server is healthy and ready!")
                    print(f"Health check response: {health_data}")
                    return True
                else:
                    print(f"Server status: {health_data.get('status')}. Retrying...")
            else:
                print(
                    f"Health check failed with status {response.status_code}. Retrying... (Body: {response.text[:200]})"
                )
        except requests.ConnectionError:
            print("Server not yet available. Retrying...")
        except requests.Timeout:
            print("Health check timed out. Retrying...")
        except Exception as e:
            print(f"Error during health check: {e}. Retrying...")
        await asyncio.sleep(3)
    print_status("Server did not become ready in time.", success=False)
    return False


async def queue_experiment(exp_config: dict):
    print_header("Queueing Small Experiment")
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "experiment_name": exp_config["experimentName"],
            "config_overrides": exp_config,
        }
        print(f"Sending payload to {START_EXPERIMENT_ENDPOINT}:")
        print(json.dumps(payload, indent=2))

        response = requests.post(
            START_EXPERIMENT_ENDPOINT, headers=headers, json=payload, timeout=30
        )

        if response.status_code == 200:
            response_data = response.json()
            print_status(f"Experiment queuing response: {response_data}")
            if response_data.get("status") == "success" and response_data.get(
                "data", {}
            ).get("experiment_id"):
                print_status("Experiment queued successfully!", success=True)
                return response_data["data"]["experiment_id"]
            else:
                print_status(
                    f"Experiment queuing failed or API returned unexpected success format: {response_data.get('message', 'No message')}",
                    success=False,
                )
                return None
        else:
            print_status(
                f"Failed to queue experiment. Status: {response.status_code}, Response: {response.text}",
                success=False,
            )
            return None
    except requests.Timeout:
        print_status("Request to queue experiment timed out.", success=False)
        return None
    except Exception as e:
        print_status(f"Error queueing experiment: {e}", success=False)
        return None


def stop_server(process):
    if process:
        print_header("Stopping VULCAN Server")
        stdout, stderr = "", ""  # Initialize stdout and stderr

        # Check if process is still running before trying to terminate
        if process.poll() is None:
            try:
                print_status(f"Sending SIGTERM to server process {process.pid}...")
                process.terminate()
                try:
                    # Wait for process to terminate and get output
                    stdout, stderr = process.communicate(
                        timeout=15
                    )  # Increased timeout
                    print_status(
                        f"Server with PID {process.pid} terminated gracefully."
                    )
                except subprocess.TimeoutExpired:
                    print_status(
                        f"Server with PID {process.pid} did not terminate gracefully after SIGTERM, attempting to kill.",
                        success=False,
                    )
                    process.kill()
                    # Wait for kill and get remaining output
                    stdout, stderr = process.communicate(timeout=5)
                    print_status(f"Server with PID {process.pid} killed.")
            except Exception as e:
                print_status(f"Error stopping server: {e}", success=False)
                # One last attempt to get output if an error occurred
                if not stdout and not stderr:
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                    except Exception:
                        pass
        else:
            print_status(
                f"Server process with PID {process.pid} already terminated (return code {process.returncode}). Retrieving final output."
            )
            stdout, stderr = process.communicate()

        if stdout and stdout.strip():
            print("\n--- Final Server STDOUT ---")
            print(stdout.strip())
            print("--- End Final Server STDOUT ---")
        if stderr and stderr.strip():
            print("\n--- Final Server STDERR ---")
            print(stderr.strip())
            print("--- End Final Server STDERR ---")


async def wait_for_queue_to_empty(timeout=300):  # 5 minute timeout
    print_header("Waiting for Experiment Queue to Empty")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{SERVER_URL}/api/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                is_running = status_data.get("is_running", False)
                queue = status_data.get("queued_experiments", [])
                print(
                    f"Polling status: Is Running = {is_running}, Queue Size = {len(queue)}"
                )

                if not is_running and not queue:
                    print_status("Queue is empty and no experiment is running.")
                    return True
            else:
                print(
                    f"Status check failed with code {response.status_code}. Retrying..."
                )
        except Exception as e:
            print(f"Error polling status: {e}. Retrying...")

        await asyncio.sleep(5)
    print_status("Timed out waiting for queue to empty.", success=False)
    return False


async def main():
    server_process = start_server()
    if not server_process:
        print_status(
            "Failed to start server process. Exiting test script.", success=False
        )
        return

    all_ok = False
    try:
        if await wait_for_server():
            # Queue the first experiment
            print_header("Queueing First Experiment")
            first_exp_config = SMALL_EXPERIMENT_CONFIG.copy()
            first_exp_config["experimentName"] = "system_test_exp_1"
            first_exp_id = await queue_experiment(first_exp_config)

            if first_exp_id:
                print_status(f"Experiment 1 ({first_exp_id}) queued successfully.")

                # Immediately queue the second experiment
                print_header("Queueing Second Experiment")
                second_exp_config = SMALL_EXPERIMENT_CONFIG.copy()
                second_exp_config["experimentName"] = "system_test_exp_2"
                second_exp_id = await queue_experiment(second_exp_config)

                if second_exp_id:
                    print_status(f"Experiment 2 ({second_exp_id}) queued successfully.")

                    # Wait for the queue to be fully processed
                    all_ok = await wait_for_queue_to_empty()
                else:
                    print_status("Failed to queue second experiment.", success=False)
            else:
                print_status("Failed to queue first experiment.", success=False)
        else:
            print_status(
                "Server did not become ready, skipping experiment queuing.",
                success=False,
            )
    finally:
        stop_server(server_process)
        if all_ok:
            print_header("System Test Script Completed Successfully")
        else:
            print_header("System Test Script Completed With Errors")


if __name__ == "__main__":
    asyncio.run(main())
