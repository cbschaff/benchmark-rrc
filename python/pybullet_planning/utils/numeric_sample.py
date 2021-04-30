import numpy as np
import random

from .shared_const import INF

def clip(value, min_value=-INF, max_value=+INF):
    """clamp a value
    """
    return min(max(min_value, value), max_value)

def randomize(iterable): # TODO: bisect
    sequence = list(iterable)
    random.shuffle(sequence)
    return sequence

def get_random_seed():
    return random.getstate()[1][0]

def get_numpy_seed():
    return np.random.get_state()[1][0]

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)

def set_numpy_seed(seed):
    # These generators are different and independent
    if seed is not None:
        np.random.seed(seed % (2**32))
        #print('Seed:', seed)
