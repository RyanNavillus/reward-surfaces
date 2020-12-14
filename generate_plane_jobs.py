import argparse
from agents.make_agent import make_agent
import torch
import json
import os
import shutil
import numpy as np

def strip_lagging_slash(f):
    if f[-1] == '/':
        return f[:-1]
    else:
        return f

def filter_normalize(param):
    # TODO: verify with loss landscapes code
    ndims = len(param.shape)
    if ndims == 1 or ndims == 0:
        # don't do any random direction for scalars
        return np.zeros_like(param)
    elif ndims == 2:
        dir = np.random.normal(size=param.shape)
        dir /= np.sqrt(np.sum(np.square(dir),axis=0,keepdims=True))
        dir *= np.sqrt(np.sum(np.square(param),axis=0,keepdims=True))
        return dir
    elif ndims == 4:
        dir = np.random.normal(size=param.shape)
        dir /= np.sqrt(np.sum(np.square(dir),axis=(0,1,2),keepdims=True))
        dir *= np.sqrt(np.sum(np.square(param),axis=(0,1,2),keepdims=True))
        return dir
    else:
        assert False, "only 1, 2, 4 dimentional filters allowed, got {}".format(param.shape)


def filter_normalized_params(agent):
    xdir = [filter_normalize(p) for p in agent.get_weights()]
    ydir = [filter_normalize(p) for p in agent.get_weights()]
    return xdir, ydir

def fischer_normalization(agent):
    raise NotImplmenetedError()


def find_unscaled_alts(agent, method):
    if method == "filter":
        return filter_normalized_params(agent)
    elif method == "fischer":
        return fischer_normalization(agent)
    else:
        raise ValueError(f"bad directions argument {method}, must be 'filter' or `fischer`")


def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--directions', type=str, default="filter")
    parser.add_argument('--magnitude', type=float, default=1.)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    base_source_path = args.checkpoint_dir
    folder_argname = os.path.dirname(strip_lagging_slash(base_source_path))
    checkpoint_fname = next(fname for fname in os.listdir(args.checkpoint_dir) if "checkpoint" in fname)
    info_fname = "info.json"
    params_fname = "parameters.th"

    os.makedirs(args.output_path, exist_ok=False)
    shutil.copy(os.path.join(args.checkpoint_dir, checkpoint_fname), os.path.join(args.output_path, checkpoint_fname))
    #shutil.copy(os.path.join(folder_argname, info_fname), os.path.join(args.output_path, info_fname))
    shutil.copy(os.path.join(args.checkpoint_dir, params_fname), os.path.join(args.output_path, params_fname))

    info = json.load(open(os.path.join(folder_argname, info_fname)))

    agent = make_agent(info['agent_name'], info['env'], args.device, info['hyperparameters'])

    alts = find_unscaled_alts(agent, args.directions)
    scaled_alts = [[a*args.magnitude for a in alt] for alt in alts]

    np.savez(os.path.join(args.output_path, "xdir.npz"), *scaled_alts[0])
    np.savez(os.path.join(args.output_path, "ydir.npz"), *scaled_alts[1])


    # update info
    info['experiment_type'] = "plane"
    info['directions'] = args.directions
    info['magnitude'] = args.magnitude

    json.dump(info, open(os.path.join(args.output_path, info_fname),'w'), indent=4)




if __name__ == "__main__":
    main()
