# Setup & Usage

## 1. Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 2. Running VULCAN

- To start a new feature engineering run, use the main orchestration script:
  ```sh
  python src/orchestrator.py
  ```
- Outputs (insights, features, plots) are saved in the `runtime/runs/` and `report/` directories.

## 3. Building the Documentation

- To view the documentation locally:
  ```sh
  pip install mkdocs-material
  mkdocs serve
  ```
  Then open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 4. Directory Structure

- `src/`: Source code for agents, orchestration, and utilities
- `runtime/`: Run outputs and logs
- `report/`: Aggregated reports and plots
- `docs/`: Documentation sources for MkDocs
