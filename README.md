# 🧬 VULCAN 2.0: Progressive Feature Evolution

**Autonomous Feature Engineering through Progressive Evolution and Reinforcement Learning**

VULCAN 2.0 represents a paradigm shift from traditional MCTS-based feature engineering to a more effective **Progressive Feature Evolution** approach. Instead of revisiting states like traditional MCTS, our system generates loads of features and keeps the best ones, using reinforcement learning to decide between generating new features vs. mutating existing ones.

## 🚀 Key Innovation: Progressive Evolution vs Traditional MCTS

**Traditional MCTS Problem:**
- Visits states repeatedly (not suitable for feature engineering)
- Limited exploration of feature space
- Doesn't accumulate knowledge effectively

**Our Progressive Evolution Solution:**
- **Population-based**: Maintains a population of best features
- **RL-driven decisions**: Agent learns when to generate new vs. mutate existing
- **Automatic repair**: Failed features are automatically fixed
- **Progressive improvement**: Each generation builds on the best from previous ones

## 🎯 Core Concepts

### 1. **Progressive Evolution Actions**
- **Generate New**: Create completely new features from scratch
- **Mutate Existing**: Modify successful features from the population

### 2. **Automatic Code Repair**
- Failed features are automatically repaired (not an RL action)
- Multiple repair attempts with different strategies
- Learning from repair patterns

### 3. **Population Management**
- Keep top-performing features across generations
- Diversity maintenance through mutation
- Adaptive population sizing

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │  Evolution      │
│   Dashboard     │◄──►│   FastAPI        │◄──►│  Orchestrator   │
│   (Next.js)     │    │   WebSocket      │    │  (RL Agent)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Data Layer     │    │  Feature Agents │
                       │   (Experiments)  │    │  (LLM-powered)  │
                       └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Quick Start

1. **Clone and setup:**
```bash
git clone <repository>
cd VULCAN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install frontend dependencies:**
```bash
cd frontend
npm install
cd ..
```

3. **Start the complete system:**
```bash
python start_vulcan.py
```

This will start both backend (port 8000) and frontend (port 3000) and open your browser.

### Alternative: Manual Start

**Backend only:**
```bash
python start_vulcan.py --backend-only
```

**Frontend only:**
```bash
python start_vulcan.py --frontend-only
```

**Separate terminals:**
```bash
# Terminal 1: Backend
uvicorn src.vulcan.api.server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

## 🧪 Running Experiments

### Using the Web Dashboard
1. Open http://localhost:3000
2. Navigate to the "Experiment" tab
3. Choose a preset or configure custom parameters
4. Click "Start Evolution"
5. Monitor progress in real-time

### Using the Command Line

**List available presets:**
```bash
python experiment_runner.py list
```

**Run a quick test:**
```bash
python experiment_runner.py run quick
```

**Run with monitoring:**
```bash
python experiment_runner.py run standard --monitor
```

**Show preset details:**
```bash
python experiment_runner.py show intensive
```

**Check system status:**
```bash
python experiment_runner.py status
```

**Stop current experiment:**
```bash
python experiment_runner.py stop
```

### Available Presets

| Preset | Description | Time | Generations | Population |
|--------|-------------|------|-------------|------------|
| `quick` | Fast test for development | 2-3 min | 5 | 10 |
| `standard` | Balanced experiment | 10-15 min | 20 | 30 |
| `intensive` | Comprehensive evolution | 30-60 min | 50 | 50 |
| `exploration` | High exploration focus | 20-30 min | 30 | 40 |
| `exploitation` | Refinement focus | 15-25 min | 25 | 35 |
| `repair_focused` | Test auto-repair | 8-12 min | 15 | 25 |

## 📊 Dashboard Features

### 🧬 Evolution Visualization
- **Population Tree**: Visual representation of feature generations
- **Score Distribution**: Color-coded performance indicators
- **Parent-Child Relationships**: Track feature lineage
- **Repair Status**: See which features were auto-repaired

### 📈 Performance Analytics
- **Generation History**: Performance trends over time
- **Success Rates**: Feature execution success tracking
- **Action Rewards**: RL learning progress visualization
- **Best Candidates**: Top-performing features highlighted

### ⚙️ Experiment Control
- **Preset Configurations**: Quick-start templates
- **Custom Parameters**: Fine-tune evolution settings
- **Real-time Monitoring**: Live progress updates
- **WebSocket Integration**: Instant status updates

## 🔧 Configuration

### Evolution Parameters

```python
{
    "max_iterations": 20,        # Number of generations
    "population_size": 30,       # Best features to keep
    "generation_size": 15,       # Features per generation
    "data_sample_size": 2000,    # Training data size
    "mutation_rate": 0.3,        # Exploration vs exploitation
    "max_repair_attempts": 3,    # Auto-repair tries
    "epsilon": 0.1,              # RL exploration rate
    "learning_rate": 0.01        # RL learning rate
}
```

### Data Configuration
- **Outer/Inner Folds**: Cross-validation setup
- **Sample Sizes**: Control computational load
- **Feature Types**: Specify target feature categories

## 🧠 Reinforcement Learning

### State Representation
- Current population statistics
- Recent action rewards
- Generation performance metrics
- Feature diversity measures

### Action Space
- **Generate New Feature**: Create from scratch
- **Mutate Existing Feature**: Modify successful feature

### Reward Function
- Performance improvement
- Feature diversity bonus
- Execution success rate
- Population quality metrics

### Learning Algorithm
- Epsilon-greedy exploration
- Q-learning with experience replay
- Adaptive learning rates
- Multi-objective optimization

## 🔍 Monitoring & Debugging

### Real-time Monitoring
```bash
# Monitor current experiment
python experiment_runner.py monitor

# Custom check interval
python experiment_runner.py monitor --interval 5
```

### API Endpoints
- `GET /api/health` - System health check
- `GET /api/status` - Current system status
- `GET /api/tree` - Evolution data
- `POST /api/experiments/start` - Start experiment
- `POST /api/experiments/stop` - Stop experiment

### WebSocket Events
- `exploration_update` - Evolution progress
- `experiment_started` - New experiment
- `experiment_completed` - Experiment finished
- `feature_evaluation` - Feature assessment

## 🚨 Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
pkill -f uvicorn
```

**Frontend build errors:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Experiment fails to start:**
```bash
# Check system status
python experiment_runner.py status

# Verify configuration
python experiment_runner.py show quick
```

### Logs and Debugging
- Backend logs: Console output from uvicorn
- Frontend logs: Browser developer console
- Experiment logs: Structured logging in terminal

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before committing

### Code Style
- Python: Black formatting, type hints
- TypeScript: ESLint + Prettier
- Commit messages: Conventional commits

### Testing
```bash
# Backend tests
pytest src/tests/

# Frontend tests
cd frontend && npm test
```

## 📚 Research & Theory

### Key Papers
- Progressive Feature Evolution methodology
- Population-based reinforcement learning
- Automatic code repair in ML pipelines
- Multi-objective feature engineering

### Comparison with Traditional Methods
- **vs MCTS**: Better exploration, no state revisiting
- **vs Genetic Algorithms**: RL-guided selection
- **vs Manual Engineering**: Autonomous and scalable
- **vs AutoML**: Feature-focused, interpretable

## 🔮 Future Roadmap

### Short Term
- [ ] Multi-dataset experiments
- [ ] Advanced repair strategies
- [ ] Feature interpretability tools
- [ ] Performance optimizations

### Long Term
- [ ] Distributed evolution
- [ ] Meta-learning across domains
- [ ] Integration with MLOps pipelines
- [ ] Advanced visualization tools

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Research team at Imperial College London
- Open source community contributions
- LLM providers for feature generation
- Beta testers and early adopters

---

**Ready to evolve your features? Start with:**
```bash
python start_vulcan.py
python experiment_runner.py run quick
```

Visit http://localhost:3000 to see your features evolve in real-time! 🧬✨ 