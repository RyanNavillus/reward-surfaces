import numpy as np

def clip_norm_(values, max_norm):
    mag = 0
    length = 0
    for g in values:
        mag += (g*g).sum()
        length += np.prod(g.shape)
    norm = mag / np.sqrt(length)
    if norm > max_norm:
        clip_v = max_norm / norm
        for g in values:
            g.data *= clip_v
