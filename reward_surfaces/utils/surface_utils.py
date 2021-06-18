import numpy as np


def readz(fname):
    outvecs = []
    with np.load(fname) as data:
        for item in data:
            outvecs.append(data[item])
    return outvecs


def scale_dir(direction, scale):
    try:
        return [d * s for d, s in zip(direction, scale)]
    except TypeError:
        return [d * scale for d in direction]


def filter_normalize(param):
    # TODO: verify with loss landscapes code
    ndims = len(param.shape)
    if ndims == 1 or ndims == 0:
        # don't do any random direction for scalars
        return np.zeros_like(param)
    elif ndims == 2:
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(np.sum(np.square(direction), axis=0, keepdims=True))
        direction *= np.sqrt(np.sum(np.square(param), axis=0, keepdims=True))
        return direction
    elif ndims == 4:
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(np.sum(np.square(direction), axis=(0, 1, 2), keepdims=True))
        direction *= np.sqrt(np.sum(np.square(param), axis=(0, 1, 2), keepdims=True))
        return direction
    else:
        assert False, "only 1, 2, 4 dimentional filters allowed, got {}".format(param.shape)


def filter_normalized_params(agent):
    xdir = [filter_normalize(p) for p in agent.get_weights()]
    ydir = [filter_normalize(p) for p in agent.get_weights()]
    return xdir, ydir
