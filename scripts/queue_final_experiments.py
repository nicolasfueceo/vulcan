import time
from pathlib import Path

import requests
import yaml

# --- Configuration ---
SERVER_URL = "http://localhost:8000"
START_EXPERIMENT_ENDPOINT = f"{SERVER_URL}/api/experiments/start"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def print_header(title):
    print("\\n" + "=" * 80)
    print(f"📋 {title}")
    print("=" * 80)


def check_server_status():
    """Checks if the server is running and healthy."""
    print("Checking server status...")
    try:
        response = requests.get(f"{SERVER_URL}/api/health", timeout=5)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            print("✅ Server is up and healthy.")
            return True
    except requests.ConnectionError:
        print(
            "❌ Server is not running. Please start it with 'scripts/launch_dashboard.py'."
        )
        return False
    except Exception as e:
        print(f"❌ An error occurred while checking server status: {e}")
        return False
    return False


def queue_experiment(config_path: Path):
    """Loads a config file and sends a request to queue the experiment."""
    try:
        with open(config_path) as f:
            config_overrides = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load or parse config file {config_path.name}: {e}")
        return False

    experiment_name = config_overrides.get("experiment", {}).get(
        "name", config_path.stem
    )
    print(f"\\n-> Queueing experiment from config: [ {config_path.name} ]")

    payload = {
        "experiment_name": experiment_name,
        "config_overrides": config_overrides,
    }

    try:
        response = requests.post(START_EXPERIMENT_ENDPOINT, json=payload, timeout=20)
        response.raise_for_status()

        response_data = response.json()
        if (
            response_data.get("status") == "success"
            and response_data.get("data", {}).get("status") == "queued"
        ):
            exp_id = response_data["data"]["experiment_id"]
            position = response_data["data"]["position"]
            print(
                f"✅ Successfully queued '{experiment_name}'. Position: {position}, ID: {exp_id[:8]}..."
            )
            return True
        else:
            print(
                f"⚠️ API returned success but experiment was not queued. Response: {response_data}"
            )
            return False

    except requests.exceptions.HTTPError as e:
        print(
            f"❌ HTTP Error for '{experiment_name}': {e.response.status_code} {e.response.text}"
        )
        return False
    except Exception as e:
        print(f"❌ Failed to queue experiment '{experiment_name}': {e}")
        return False


def main():
    print_header("VULCAN Final Experiment Queuing Script")

    if not check_server_status():
        return

    if not CONFIG_DIR.exists() or not any(CONFIG_DIR.glob("*.yaml")):
        print(f"❌ No configuration files found in {CONFIG_DIR}.")
        print("Please run 'scripts/setup_config.py' first to generate them.")
        return

    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    print(f"Found {len(config_files)} experiment configurations to queue.")

    successful_queues = 0
    for i, config_file in enumerate(config_files):
        print("-" * 40)
        if queue_experiment(config_file):
            successful_queues += 1
        else:
            print(f"🚨 Failed to queue experiment from {config_file.name}. Stopping.")
            break
        time.sleep(1)

    print_header("Summary")
    print(
        f"✅ Successfully queued {successful_queues} / {len(config_files)} experiments."
    )
    if successful_queues > 0:
        print(
            "You can now monitor the progress on the VULCAN dashboard (http://localhost:3000) and TensorBoard (http://localhost:6006)."
        )


if __name__ == "__main__":
    main()
