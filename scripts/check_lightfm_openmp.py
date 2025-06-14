"""
Script to verify if LightFM was installed with OpenMP support.
"""
from lightfm import LightFM
import numpy as np
from loguru import logger

def check_openmp():
    """Check if LightFM was compiled with OpenMP support."""
    try:
        from scipy.sparse import csr_matrix
        
        # Create a small test dataset in sparse format
        interactions = csr_matrix(np.random.randint(0, 2, (100, 100)).astype(np.float32))
        
        # Create model with multiple threads
        model = LightFM(no_components=10, loss='warp', random_state=42)
        
        # Try to use multiple threads
        n_threads = 4
        logger.info(f"Fitting model with {n_threads} threads...")
        
        # Fit the model
        model.fit(interactions, epochs=1, num_threads=n_threads)
        
        # If we get here without error, OpenMP is likely working
        logger.success("LightFM appears to be using OpenMP successfully!")
        logger.info("Check your system monitor - you should see CPU usage spike briefly.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.warning("LightFM may not be compiled with OpenMP support.")
        logger.info("Try reinstalling with: CFLAGS=-fopenmp LDFLAGS=-fopenmp pip install --no-binary :all: lightfm")
        return False

if __name__ == "__main__":
    logger.info("Testing LightFM OpenMP support...")
    check_openmp()
