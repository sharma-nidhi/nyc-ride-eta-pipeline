# -----------------------------------------------------------------------------
# Project: Delivery / Ride ETA Prediction
# Copyright (c) 2026 - ETA Prediction Project Team
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

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
