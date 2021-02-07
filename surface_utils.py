import numpy as np

def readz(fname):
    outvecs = []
    with np.load(fname) as data:
        for item in data:
            outvecs.append(data[item])
    return outvecs
