# VULCAN: Versatile User-Learning Conversational Agent for Nudging

A sophisticated two-phase recommender system that combines LLM-driven autonomous feature engineering using Monte Carlo Tree Search (MCTS) with conversational cold-start user assignment.

## 🎯 Overview

VULCAN addresses the cold-start problem in recommender systems through:

1. **Phase 1**: Autonomous feature engineering using MCTS to discover optimal user representations
2. **Phase 2**: Conversational agent for cold-start user assignment and recommendation

## 🏗️ Architecture

The system has been completely restructured into a modular, maintainable architecture:

```
src/autonomous_fe_env/
├── agents/              # LLM and stub agents for feature engineering
├── config/              # Configuration management
├── data/                # Data access and database utilities
├── evaluation/          # Feature evaluation and cold-start metrics
├── feature/             # Feature representation and management
├── mcts/                # Monte Carlo Tree Search implementation
├── reflection/          # LLM-based reflection and reasoning
├── sandbox/             # Safe code execution environment
├── state/               # State management and persistence
└── visualization/       # Monitoring and visualization tools
```

## 🚀 Key Features

### Modular Architecture
- **Clean separation of concerns** with dedicated modules for each functionality
- **Pluggable components** that can be easily extended or replaced
- **Comprehensive configuration** system with YAML support

### Advanced MCTS Implementation
- **Parallel MCTS** for exploring multiple nodes simultaneously
- **Reflection loops** for agents to reason about feature engineering decisions
- **State persistence** with checkpoint and recovery capabilities
- **Adaptive exploration** strategies

### Robust Feature Engineering
- **Feature validation** and safety checks
- **Feature registry** with dependency tracking and scoring
- **Sandbox execution** for safe code evaluation
- **Multiple agent types** (predefined, LLM-based, hybrid)

### Real LLM Integration
- **OpenAI GPT-4o-mini** integration with LangChain
- **Intelligent prompt engineering** for feature generation
- **Strategic reflection** using LLM-based analysis
- **Automatic fallback** to mock mode when API key unavailable

### Comprehensive Evaluation
- **Cold-start evaluation** with clustering-based metrics
- **Baseline comparisons** with multiple recommendation algorithms
- **Performance tracking** and visualization

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VULCAN
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API (for real LLM features)**:
   ```bash
   python3 setup_openai.py
   ```
   Or manually set your API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

5. **Verify installation**:
   ```bash
   python3 test_llm_integration.py
   ```

## 🔧 Configuration

The system uses YAML configuration files. The default configuration is in `src/autonomous_fe_env/config/default_config.yaml`.

Key configuration sections:

### Database Configuration
```yaml
database:
  path: "data/goodreads.db"
  type: "sqlite"
  connection_timeout: 30
```

### MCTS Configuration
```yaml
mcts:
  max_iterations: 50
  exploration_constant: 1.414
  max_depth: 10
  parallel_workers: 4
```

### Agent Configuration
```yaml
agents:
  feature_agent:
    type: "llm"  # or "stub" for testing
    model: "gpt-3.5-turbo"
    temperature: 0.7
```

## 🎮 Usage

### Basic Usage

```python
from autonomous_fe_env import ConfigManager, MCTSOrchestrator

# Load configuration
config_manager = ConfigManager("path/to/config.yaml")
config = config_manager.get_config()

# Initialize orchestrator
orchestrator = MCTSOrchestrator(config)

# Run feature engineering
results = orchestrator.run()
```

### Database Setup

1. **Merge existing databases** (if you have separate train/test/validation databases):
   ```python
   from autonomous_fe_env.data import DatabaseMerger
   
   merger = DatabaseMerger()
   merger.merge_databases(
       source_dbs=["data/train.db", "data/test.db", "data/validation.db"],
       target_db="data/goodreads.db"
   )
   ```

2. **Create data splits**:
   ```python
   from autonomous_fe_env.data import create_split_indices
   
   create_split_indices("data/goodreads.db", "data/splits/")
   ```

### Feature Engineering

```python
from autonomous_fe_env import get_agent, StateManager, FeatureEvaluator

# Initialize components
state_manager = StateManager("state/")
feature_agent = get_agent("feature", config={"mode": "predefined"})
evaluator = FeatureEvaluator(config)

# Generate and evaluate features
context = {"state_manager": state_manager}
result = feature_agent.execute(context)
feature = result["feature"]

if feature:
    score = evaluator.evaluate_feature(feature, state_manager)
    state_manager.update_state([feature], score)
```

### Reflection and Reasoning

```python
from autonomous_fe_env.reflection import ReflectionEngine

# Initialize reflection engine
reflection_engine = ReflectionEngine(config)

# Generate reflection on feature performance
reflection = reflection_engine.reflect_on_evaluation(
    feature=feature,
    score=score,
    context={"iteration": 1, "best_score": 0.5}
)

print(reflection.content)
```

### Interactive Dashboard

Launch the real-time Streamlit dashboard:

```bash
streamlit run streamlit_dashboard.py
```

The dashboard provides:
- **Real-time monitoring** of feature engineering progress
- **Live LLM prompts and responses** with comprehensive logging
- **Interactive visualizations** using Plotly
- **Agent activity tracking** and reflection timeline
- **Performance metrics** and feature history

Features of the dashboard:
- 🚀 **Start/Stop/Reset** controls for feature engineering
- 📊 **MCTS progress plots** with score tracking
- 🤖 **LLM prompt logging** with full conversation history
- 💭 **Reflection insights** with strategic analysis
- 📈 **Performance comparisons** against baselines
- 🔄 **Auto-refresh** functionality (2-second intervals)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_vulcan_system.py
```

This tests:
- ✅ Core component imports
- ✅ Configuration management
- ✅ Feature definition creation
- ✅ Agent initialization
- ✅ State management
- ✅ Visualization components

## 📊 Monitoring and Visualization

The system includes comprehensive monitoring:

```python
from autonomous_fe_env.visualization import PipelineVisualizer, AgentMonitor

# Initialize visualizers
visualizer = PipelineVisualizer(config)
monitor = AgentMonitor(config)

# Log MCTS progress
visualizer.log_mcts_iteration(
    iteration=1,
    best_score=0.75,
    nodes_explored=10,
    features=["user_avg_rating", "book_popularity"]
)

# Monitor agent activity
monitor.log_agent_activity(
    agent_name="FeatureAgent",
    activity="feature_generation",
    metadata={"feature_count": 5}
)
```

## 🔍 Key Components

### 1. MCTS Implementation (`src/autonomous_fe_env/mcts/`)
- **MCTSNode**: Enhanced node with unique IDs and serialization
- **MCTSOrchestrator**: Main orchestrator for MCTS execution
- **ParallelMCTS**: Parallel exploration of multiple nodes

### 2. Feature Management (`src/autonomous_fe_env/feature/`)
- **FeatureDefinition**: Comprehensive feature representation
- **FeatureRegistry**: Feature storage with scoring and dependencies
- **FeatureValidator**: Safety and syntax validation

### 3. Agents (`src/autonomous_fe_env/agents/`)
- **FeatureAgent**: Generates feature engineering proposals
- **ReflectionAgent**: Provides reasoning about feature decisions
- **BaseAgent**: Common interface for all agents

### 4. Data Access (`src/autonomous_fe_env/data/`)
- **SqlDAL**: Database access layer with connection pooling
- **DatabaseMerger**: Utility for merging multiple databases
- **Data splitting**: Tools for train/test/validation splits

### 5. Evaluation (`src/autonomous_fe_env/evaluation/`)
- **FeatureEvaluator**: Clustering-based feature evaluation
- **ColdStartEvaluator**: Cold-start specific metrics
- **Baseline comparisons**: Multiple recommendation algorithms

## 🔧 Advanced Features

### Parallel MCTS
Enable parallel exploration for faster convergence:

```yaml
parallel_mcts:
  enabled: true
  num_workers: 4
  batch_size: 10
  synchronization_interval: 5
```

### Reflection System
Enable LLM-based reasoning:

```yaml
reflection:
  enabled: true
  frequency: 5  # Every 5 iterations
  memory_size: 100
  reflection_types:
    - "feature_proposal"
    - "evaluation_result"
    - "strategy_adjustment"
```

### Sandbox Execution
Safe code execution with resource limits:

```yaml
sandbox:
  enabled: true
  timeout: 30
  memory_limit: "512MB"
  cpu_limit: 1.0
```

## 🛠️ Development

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required methods: `execute()`, `validate_context()`
3. Register in the agent factory function

### Adding New Evaluation Metrics

1. Extend `FeatureEvaluator` or create a new evaluator
2. Implement the evaluation logic
3. Update configuration to use the new metric

### Extending MCTS

1. Modify `MCTSNode` for new node behaviors
2. Update `MCTSOrchestrator` for new search strategies
3. Add configuration options for new parameters

## 📈 Performance Optimization

- **Parallel processing**: Use multiple workers for MCTS exploration
- **Caching**: Feature evaluation results are cached
- **Database optimization**: Connection pooling and query optimization
- **Memory management**: Configurable limits and garbage collection

## 🔒 Security

- **Code validation**: AST-based validation of feature code
- **Sandbox execution**: Isolated execution environment
- **Input sanitization**: Protection against code injection
- **Resource limits**: Memory and CPU usage constraints

## 📝 Logging

Comprehensive logging with configurable levels:

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "vulcan.log"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Built for Imperial College London thesis research
- Utilizes Monte Carlo Tree Search for autonomous feature engineering
- Integrates with modern LLM APIs for intelligent reasoning

---

For more detailed documentation, see the individual module documentation in the `src/autonomous_fe_env/` directory.

