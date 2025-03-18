import random
from typing import Any
import os
import numpy as np

def set_seeds(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def is_bool_or_binary(value: Any) -> bool:
    """Tests if a value is a boolean or binary (0, 1).

    Args:
        value Any: a value to test

    Returns:
        bool: boolean indicating if the value is a boolean or binary
    """
    return isinstance(value, bool) or value in (0, 1)