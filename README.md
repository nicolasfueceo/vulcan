# VULCAN - AI-Powered Book Recommendation Analysis

This project is an AI-powered system that analyzes a book recommendation dataset to discover insights and formulate hypotheses for improving the recommendation engine. It uses a multi-agent system built with AutoGen to orchestrate the analysis.

## Project Structure

The project is organized into the following directories:

- **`data/`**: Contains the raw and curated data files in DuckDB format. This directory is in `.gitignore` and data is not checked in.
- **`data_curation/`**: Contains the data curation pipeline to process the raw data and generate the curated database.
  - `run.py`: The main script to run the entire curation pipeline.
  - `sql/`: Contains the SQL scripts that are executed in order by the pipeline.
  - `steps/`: Contains additional Python scripts for manual analysis and verification.
  - See `data_curation/README.md` for more details.
- **`src/`**: Contains the source code for the main application.
  - `orchestrator.py`: The main entry point for running the agent-based analysis.
  - `agents/`: Contains the prompts and definitions for the different AI agents.
  - `config/`: Contains configuration files for logging and other services.
  - `core/`: Contains the core components of the application, such as database and LLM interactions.
  - `schemas/`: Contains the Pydantic models for the data structures used in the application.
  - `utils/`: Contains various utility functions.
- **`runtime/`**: Contains the output of the application runs, such as logs, plots, and session state. This directory is in `.gitignore`.

## How to Run

### 1. Setup

First, create a virtual environment and install the required packages.

### 2. Data Curation

If you have the raw data, you can run the data curation pipeline to generate the curated database:

```bash
python data_curation/run.py
```

### 3. Run the Analysis

To run the main analysis pipeline, execute the orchestrator:

```bash
export PYTHONPATH="."
export OPENAI_API_KEY="your_api_key"
python src/orchestrator.py
```

This will start the multi-agent system to analyze the data and generate insights. The output of the run will be stored in a new directory inside `runtime/runs/`. 