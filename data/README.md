# VULCAN Data Module

This module handles data loading, processing, and splitting for the VULCAN two-phase recommender system.

## Data Structure

The data is stored in SQLite databases:

- `train.db`: Main training database
- `test.db`: Test database
- `validation.db`: Validation database

Each database contains the following tables:
- `reviews`: User ratings and reviews for books
- `books`: Book metadata

## Data Splits

For the two-phase recommender system, we use a nested cross-validation approach:

1. **Outer Hold-Out (20% users)**: Final "warm-user" test in Phase 2
2. **Design Set (80% users)**: Phase 1 feature engineering & clustering
   - **K-Fold Outer CV** (K=5) on Design Set:
     - **TrainFE** (K-1 folds): Feature search
     - **ValClusters** (1 fold): Cluster validation
   - **Inside TrainFE**: M-fold inner split (M=3) for feature evaluation:
     - **FeatTrain**: Train models for feature
     - **FeatVal**: Validate feature (RMSE)

## Usage

### Generate Data Splits

To generate all data splits:

```bash
python -m data.generate_splits --config data/config.yaml
```

This will:
1. Create CSV files with user IDs for each split in `data/splits/`
2. Generate SQL queries for extracting data for each split in `data/queries/`

### Access Data Splits in Code

```python
from data.data_splits import DataSplitter

# Create splitter
splitter = DataSplitter()

# Get all users
all_users = splitter.get_all_users()

# Create outer split (design/test)
design_users, test_users = splitter.create_outer_split()

# Create outer CV folds
outer_cv_folds = splitter.create_outer_cv_folds()

# Create inner CV folds for a specific outer fold
inner_cv_folds = splitter.create_inner_cv_folds(fold_idx=0)
```

## Configuration

All data splitting parameters are configured in `config.yaml`:

```yaml
data_splits:
  test_size: 0.2            # Outer test set size (20% of all users)
  outer_folds: 5            # K-fold for outer CV
  inner_folds: 3            # M-fold for inner CV
  random_state: 42          # Random seed for reproducibility
  min_ratings_per_user: 5   # Minimum number of ratings a user must have
```

## Directory Structure

```
data/
├── __init__.py
├── config.yaml            # Configuration file
├── data_splits.py         # Data splitting module
├── generate_splits.py     # Script to generate splits
├── goodreads_fingerprinting.py  # Feature fingerprinting
├── train.db               # Main training database
├── test.db                # Test database
├── validation.db          # Validation database
├── splits/                # CSV files with user IDs for each split
└── queries/               # SQL queries for each split
``` 