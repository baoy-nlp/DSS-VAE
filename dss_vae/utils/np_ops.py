import numpy as np


def batch_shuffle_indices(batch_size, range_size):
    origin = np.arange(batch_size * range_size).reshape(batch_size, range_size)
    origin = origin % range_size

    for item in origin:  # using the loop, may could improve with other inline function
        np.random.shuffle(item)

    return origin
