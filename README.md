# FUEGO Benchmark System

This repository contains the benchmark system for the FUEGO (Emotionally Intelligent Conversational Travel Recommender) project. The benchmark system provides a standardized framework for evaluating and comparing different recommender systems.

## Overview

The FUEGO benchmark system is designed to evaluate recommender systems using the MovieLens dataset. It provides:

1. **Data Processing Pipeline**: Tools for downloading, processing, and preparing data
2. **Qdrant Integration**: Vector database storage for efficient similarity search
3. **Recommender Interfaces**: Standardized interfaces for different types of recommenders
4. **Benchmark Framework**: Comprehensive evaluation metrics and visualization tools
5. **Testing Pipeline**: Tools for running benchmarks and analyzing results

## Installation

### Prerequisites

- Python 3.8+
- Qdrant server (local or cloud)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/fuego-benchmark.git
cd fuego-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script:
```bash
./setup.sh
```

## Project Structure

```
fuego_project/
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── interim/               # Intermediate data products
├── models/                    # Recommender models
│   ├── recommender_interfaces.py  # Interface definitions
│   ├── baseline_recommenders.py   # Baseline recommenders
│   └── qdrant_recommenders.py     # Qdrant-based recommenders
├── scripts/                   # Scripts for running experiments
│   ├── data_processing.py     # Data processing pipeline
│   ├── benchmark_framework.py # Benchmark framework
│   ├── test_benchmark.py      # Testing pipeline
│   └── demonstrate_benchmark.py # Demonstration script
├── utils/                     # Utility modules
│   ├── data_downloader.py     # Data downloading utilities
│   ├── movielens_processor.py # MovieLens data processor
│   ├── embedding_pipeline.py  # Embedding generation pipeline
│   └── qdrant_manager.py      # Qdrant integration
├── results/                   # Benchmark results
│   └── benchmarks/            # Benchmark result files
├── requirements.txt           # Project dependencies
├── setup.sh                   # Setup script
└── README.md                  # Project documentation
```

## Data Processing

The data processing pipeline handles downloading, processing, and preparing data for the benchmark system:

```python
from scripts.data_processing import DataProcessor

# Initialize data processor
processor = DataProcessor()

# Download and process MovieLens dataset
benchmark_data = processor.prepare_benchmark_data(size="small")
```

## Recommender Models

The benchmark system includes several recommender models:

1. **Baseline Recommenders**:
   - `PopularityRecommender`: Recommends the most popular items
   - `UserKNNRecommender`: User-based collaborative filtering

2. **Qdrant-based Recommenders**:
   - `QdrantCFRecommender`: Matrix factorization with Qdrant storage
   - `QdrantItemKNNRecommender`: Item-based KNN with Qdrant storage

All recommenders implement standardized interfaces defined in `recommender_interfaces.py`.

## Benchmark Framework

The benchmark framework provides tools for evaluating and comparing recommender systems:

```python
from scripts.benchmark_framework import BenchmarkFramework
from models.baseline_recommenders import PopularityRecommender, UserKNNRecommender
from models.qdrant_recommenders import QdrantCFRecommender, QdrantItemKNNRecommender

# Initialize benchmark framework
benchmark = BenchmarkFramework()

# Load dataset
dataset = benchmark.load_dataset(
    train_path="data/processed/movielens/ml-small/train_ratings.csv",
    test_path="data/processed/movielens/ml-small/test_ratings.csv",
    items_path="data/processed/movielens/ml-small/movies.csv"
)

# Initialize recommenders
recommenders = [
    PopularityRecommender(),
    UserKNNRecommender(),
    QdrantCFRecommender(),
    QdrantItemKNNRecommender()
]

# Run benchmark
results = benchmark.benchmark_recommenders(
    recommenders=recommenders,
    dataset=dataset,
    dataset_name="movielens_small"
)

# Visualize results
benchmark.visualize_recommendation_metrics(
    dataset_name="movielens_small",
    metric="ndcg",
    k=10
)
```

## Running Benchmarks

The repository includes scripts for running benchmarks:

```bash
# Run full benchmark
python scripts/test_benchmark.py --mode benchmark --dataset small --n_users 100

# Generate recommendations for a specific user
python scripts/test_benchmark.py --mode user_recommendations --user_id 42 --k 10

# Find similar items for a specific item
python scripts/test_benchmark.py --mode similar_items --item_id 1 --k 10
```

For a step-by-step demonstration, run:

```bash
python scripts/demonstrate_benchmark.py
```

## Evaluation Metrics

The benchmark system includes the following evaluation metrics:

1. **Rating Prediction**:
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)

2. **Top-N Recommendation**:
   - Precision@k
   - Recall@k
   - Normalized Discounted Cumulative Gain (NDCG@k)
   - Hit Rate@k

## Qdrant Integration

The benchmark system integrates with Qdrant vector database for efficient similarity search:

```python
from utils.qdrant_manager import QdrantManager

# Initialize Qdrant manager
qdrant = QdrantManager(host="localhost", port=6333)

# Create collection
qdrant.create_collection("item_embeddings", vector_size=100)

# Upload vectors
qdrant.upload_vectors(
    collection_name="item_embeddings",
    vectors=item_vectors,
    ids=item_ids,
    payloads=item_payloads
)

# Search for similar vectors
similar_items = qdrant.search_similar_vectors(
    collection_name="item_embeddings",
    query_vector=query_vector,
    limit=10
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
