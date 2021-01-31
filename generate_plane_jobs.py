import argparse
from agents.make_agent import make_agent
import torch
import json
import os
import shutil
from pathlib import Path
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
    parser.add_argument('--directions', type=str, default="filter", help="'filter' is only option right now")
    parser.add_argument('--copy-directions', type=str, help="overrides directions with directions from specified folder. Does not copy any other data. ")
    parser.add_argument('--magnitude', type=float, default=1., help="scales directions by given amount")
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--num-episodes', type=int)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_offset_critic', action='store_true', help="use critic at center or at offset for value estimation")
    parser.add_argument('--est-hesh', action='store_true')
    parser.add_argument('--calc-hesh', action='store_true')

    args = parser.parse_args()

    assert args.grid_size % 2 == 1, "grid-size must be odd"
    assert args.num_steps is not None or args.num_episodes is not None, "one of num_step or num_episodes must be specified"
    if args.num_steps is None:
        args.num_steps = 10000000000000
    if args.num_episodes is None:
        args.num_episodes = 10000000000000


    output_path = Path(args.output_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    folder_argname = Path(os.path.dirname(strip_lagging_slash(args.checkpoint_dir)))
    checkpoint_fname = next(fname for fname in os.listdir(checkpoint_dir) if "checkpoint" in fname)
    checkpoint_path = checkpoint_dir / checkpoint_fname
    info_fname = "info.json"
    params_fname = "parameters.th"

    os.makedirs(output_path, exist_ok=False)
    shutil.copy(checkpoint_path, (output_path / checkpoint_fname))
    #shutil.copy((folder_argname, info_fname), (output_path, info_fname))
    shutil.copy((checkpoint_dir / params_fname), (output_path / params_fname))

    info = json.load(open((folder_argname / info_fname)))

    device = "cpu"
    agent = make_agent(info['agent_name'], info['env'], device, info['hyperparameters'])
    agent.load_weights(checkpoint_path)

    if args.copy_directions is not None:
        # copy directions
        dir_path = Path(args.copy_directions)
        shutil.copy(dir_path / "dir1.npz", output_path / "dir1.npz")
        shutil.copy(dir_path / "dir2.npz", output_path / "dir2.npz")
    else:
        # generate directions normally
        alts = find_unscaled_alts(agent, args.directions)
        scaled_alts = [[a*args.magnitude for a in alt] for alt in alts]

        np.savez((output_path / "dir1.npz"), *scaled_alts[0])
        np.savez((output_path / "dir2.npz"), *scaled_alts[1])

    # update info
    info['experiment_type'] = "plane"
    info['directions'] = args.directions if args.copy_directions is None else "copy"
    info['checkpoint_dir'] = args.checkpoint_dir
    info['magnitude'] = args.magnitude
    info['grid_size'] = args.grid_size
    info['num_episodes'] = args.num_episodes
    info['num_steps'] = args.num_steps
    info['eval_device'] = args.device
    info['calc_hesh'] = args.device

    json.dump(info, open((output_path / info_fname),'w'), indent=4)

    job_out_path = os.mkdir((output_path / "results"))
    seperate_eval_arg = " --use_offset_critic " if args.use_offset_critic else ""

    job_list = []
    for i in range(args.grid_size):
        for j in range(args.grid_size):
            x = i - args.grid_size // 2
            y = j - args.grid_size // 2
            job = f"python eval_plane_job.py {args.output_path} --offset1={x}  --offset2={y} {seperate_eval_arg}"
            #print(job)
            job_list.append(job)

    jobs = "\n".join(job_list)+"\n"
    open((output_path / "jobs.sh"),'w').write(jobs)



if __name__ == "__main__":
    main()
