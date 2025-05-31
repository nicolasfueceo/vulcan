# VULCAN Streamlit Dashboard

A real-time dashboard for monitoring the VULCAN MCTS exploration process, built with Streamlit.

## Features

- Interactive MCTS tree visualization
- Real-time metrics monitoring
- System resource usage tracking
- LLM interaction history
- Auto-refresh capability

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the VULCAN API server is running on http://localhost:8000

2. Start the Streamlit dashboard:
```bash
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Controls

- **Auto-refresh**: Toggle automatic data refresh
- **Refresh interval**: Adjust the refresh rate (1-30 seconds)

## Data Visualization

- **MCTS Tree**: Interactive visualization of the exploration tree
- **Current Metrics**: Real-time display of exploration and system metrics
- **Metrics Over Time**: Historical trends of key metrics
- **LLM Interactions**: Detailed history of LLM interactions 