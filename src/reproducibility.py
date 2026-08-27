"""
Reproducibility helpers shared across training and simulation scripts.
"""

import random
import numpy as np

DEFAULT_RANDOM_SEED = 42


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Seed Python and NumPy RNGs for deterministic behavior where applicable."""
    random.seed(seed)
    np.random.seed(seed)
    return seed
