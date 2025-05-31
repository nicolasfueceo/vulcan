# VULCAN 2.0: Autonomous Feature Engineering for Recommender Systems

A modern, production-ready system for autonomous feature engineering using Monte Carlo Tree Search (MCTS) and Large Language Models (LLMs).

## 🎯 Overview

VULCAN 2.0 is a complete rewrite of the original VULCAN system, following modern software engineering practices and providing a clean, maintainable codebase for autonomous feature engineering research.

### Key Features

- **🏗️ Modern Architecture**: Clean separation of concerns with typed interfaces
- **🚀 FastAPI Backend**: RESTful API with WebSocket support for real-time updates
- **📊 Experiment Tracking**: Weights & Biases integration for academic-grade logging
- **🔧 Type Safety**: Full TypeScript-style type annotations with Pydantic
- **🧪 Comprehensive Testing**: pytest-based test suite with fixtures and mocking
- **📝 Rich Logging**: Structured logging with Rich console output
- **⚙️ Configuration Management**: YAML-based configuration with validation
- **🎨 CLI Interface**: Beautiful command-line interface with Typer and Rich

## 🏗️ Architecture

```
src/vulcan/
├── types/              # Type definitions (Pydantic models)
├── core/               # Core system components
│   ├── config_manager.py
│   └── orchestrator.py
├── api/                # FastAPI server and routes
├── utils/              # Utilities (logging, experiment tracking)
├── data/               # Data access layer
├── agents/             # LLM and feature agents
├── mcts/               # Monte Carlo Tree Search
├── evaluation/         # Feature evaluation
└── features/           # Feature management
```

## 🚀 Quick Start

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd vulcan-clean
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Configure environment**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Usage

#### Command Line Interface

```bash
# Start the API server
vulcan serve --host localhost --port 8000

# Run an experiment
vulcan experiment --name "my_experiment" --iterations 100

# Check system status
vulcan status

# Validate configuration
vulcan config --validate

# Show configuration
vulcan config
```

#### Python API

```python
from vulcan import ConfigManager, VulcanOrchestrator, setup_logging

# Load configuration
config_manager = ConfigManager("config/vulcan.yaml")
config = config_manager.config

# Setup logging
setup_logging(config.logging)

# Initialize orchestrator
orchestrator = VulcanOrchestrator(config)
await orchestrator.initialize_components()

# Run experiment
experiment_id = await orchestrator.start_experiment("my_experiment")
```

#### API Endpoints

- **Health**: `GET /api/health`
- **Status**: `GET /api/status`
- **Start Experiment**: `POST /api/experiments/start`
- **Stop Experiment**: `POST /api/experiments/stop`
- **Experiment History**: `GET /api/experiments/history`
- **WebSocket**: `WS /ws`

## 📊 Configuration

Configuration is managed through YAML files with full validation:

```yaml
# config/vulcan.yaml
mcts:
  max_iterations: 50
  exploration_factor: 1.414
  max_depth: 10

llm:
  provider: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.7

api:
  enabled: true
  host: "localhost"
  port: 8000

experiment:
  wandb_enabled: true
  wandb_project: "vulcan-feature-engineering"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vulcan --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

## 📝 Development

### Code Quality

The project follows strict code quality standards:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff src/ tests/
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pre-commit install
```

## 🔧 Type System

VULCAN 2.0 uses a comprehensive type system with Pydantic models:

```python
from vulcan.types import VulcanConfig, ExperimentRequest, ExperimentResult

# Type-safe configuration
config = VulcanConfig(
    mcts={"max_iterations": 100},
    llm={"temperature": 0.5}
)

# Type-safe API requests
request = ExperimentRequest(
    experiment_name="test",
    config_overrides={"mcts": {"max_iterations": 25}}
)
```

## 📊 Experiment Tracking

Automatic experiment tracking with Weights & Biases:

- **Metrics**: Real-time logging of MCTS progress, feature scores
- **Artifacts**: Feature code, LLM reflections, experiment states
- **Visualization**: Learning curves, tree exploration patterns
- **Reproducibility**: Complete configuration and state tracking

## 🏃‍♂️ Performance

- **Async/Await**: Non-blocking experiment execution
- **WebSocket**: Real-time updates with minimal latency
- **Structured Logging**: Efficient JSON-based logging
- **Type Validation**: Runtime validation with Pydantic
- **Memory Management**: Proper resource cleanup and management

## 🔒 Security

- **Input Validation**: All inputs validated with Pydantic
- **CORS Configuration**: Proper cross-origin resource sharing
- **Environment Variables**: Secure API key management
- **Error Handling**: Comprehensive error handling and logging

## 📚 Documentation

- **Type Hints**: Full type annotations for all functions
- **Docstrings**: Google-style docstrings for all public APIs
- **API Docs**: Auto-generated OpenAPI documentation at `/api/docs`
- **Examples**: Comprehensive examples in the documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following the code quality standards
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for Imperial College London thesis research
- Utilizes Monte Carlo Tree Search for autonomous feature engineering
- Integrates with modern LLM APIs for intelligent reasoning
- Follows industry best practices for production-ready software

---

**VULCAN 2.0**: Where autonomous feature engineering meets modern software engineering. 🚀 