#!/bin/zsh
# Source ~/.zshrc, activate conda env, install dependencies, and run the test
set -e

# Ensure we have conda in PATH
source ~/.zshrc

# Activate the vulcan environment
conda activate vulcan

# Install dependencies (pip preferred for requirements.txt)
pip install -r requirements.txt

# Run the feature realization test
PYTHONPATH=. python3 scripts/test_feature_realization.py
