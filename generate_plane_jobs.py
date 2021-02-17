import argparse
from agents.make_agent import make_agent
import torch
import json
import os
import shutil
from pathlib import Path
import numpy as np
from surface_utils import readz

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

def readz(fname):
    outvecs = []
    with np.load(fname) as data:
        for item in data:
            outvecs.append(data[item])
    return outvecs

def scale_dir(dir, scale):
    # normalize scale value so magnitude isn't completely off (PROBABLY A BAD IDEA?)
    # scale_size = sum(np.abs(s).sum() for s in scale)
    # num_items = sum(s.size for s in scale)
    # scale_val = scale_size/num_items
    return [d * s for d,s in zip(dir, scale)]

def generate_plane_data(
    checkpoint_dir,
    output_path,
    dir1_vec,
    dir2_vec,
    info,
    grid_size=5,
    num_steps=None,
    num_episodes=None,
    device="cpu",
    use_offset_critic=False,
    est_hesh=False,
    calc_hesh=False,
    calc_grad=False):

    assert isinstance(dir1_vec, list) and isinstance(dir1_vec[0], np.ndarray), "dir1 and dir2 must be a list of numpy vectors. Use `from surface_utils import readz; readz(fname)` to read a numpy vector into this format"
    assert isinstance(dir2_vec, list) and isinstance(dir2_vec[0], np.ndarray), "dir1 and dir2 must be a list of numpy vectors. Use `from surface_utils import readz; readz(fname)` to read a numpy vector into this format"
    assert grid_size % 2 == 1, "grid-size must be odd"
    assert num_steps is not None or num_episodes is not None, "one of num_step or num_episodes must be specified"
    if num_steps is None:
        num_steps = 10000000000000
    if num_episodes is None:
        num_episodes = 10000000000000

    output_path = Path(output_path)
    folder_argname = Path(os.path.dirname(strip_lagging_slash(checkpoint_dir)))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_fname = next(fname for fname in os.listdir(checkpoint_dir) if "checkpoint" in fname)
    checkpoint_path = checkpoint_dir / checkpoint_fname
    info_fname = "info.json"
    params_fname = "parameters.th"

    os.makedirs(output_path, exist_ok=False)
    shutil.copy(checkpoint_path, (output_path / checkpoint_fname))
    shutil.copy((checkpoint_dir / params_fname), (output_path / params_fname))

    np.savez((output_path / "dir1.npz"), *dir1_vec)
    np.savez((output_path / "dir2.npz"), *dir2_vec)

    # update info
    info['experiment_type'] = "plane"
    info['checkpoint_dir'] = str(checkpoint_dir)
    info['grid_size'] = grid_size
    info['num_episodes'] = num_episodes
    info['num_steps'] = num_steps
    info['device'] = device
    info['calc_hesh'] = calc_hesh
    info['est_hesh'] = est_hesh
    info['calc_grad'] = calc_grad

    json.dump(info, open((output_path / info_fname),'w'), indent=4)

    job_out_path = os.mkdir((output_path / "results"))
    seperate_eval_arg = " --use_offset_critic " if use_offset_critic else ""

    job_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = i - grid_size // 2
            y = j - grid_size // 2
            job = f"python eval_plane_job.py {output_path} --offset1={x}  --offset2={y} {seperate_eval_arg}"
            #print(job)
            job_list.append(job)

    jobs = "\n".join(job_list)+"\n"
    open((output_path / "jobs.sh"),'w').write(jobs)



def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--directions', type=str, default="filter", help="'filter' is only option right now")
    parser.add_argument('--copy-directions', type=str, help="overrides directions with directions from specified folder. Does not copy any other data. ")
    parser.add_argument('--scale-vec', type=str, help="A .npz file of same shape as directions, which indicates how much each dimention should be scaled by.")
    parser.add_argument('--dir1', type=str, help="overrides dir1 with vector from specified path.")
    parser.add_argument('--dir2', type=str, help="overrides dir2 with vector from specified path.")
    parser.add_argument('--magnitude', type=float, default=1., help="scales directions by given amount")
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--num-episodes', type=int)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_offset_critic', action='store_true', help="use critic at center or at offset for value estimation")
    parser.add_argument('--est-hesh', action='store_true')
    parser.add_argument('--calc-hesh', action='store_true')
    parser.add_argument('--calc-grad', action='store_true')

    args = parser.parse_args()

    assert args.copy_directions is None or args.directions == "copy", "if --copy-directions is None, --directions=copy must be set"
    output_path = Path(args.output_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    folder_argname = Path(os.path.dirname(strip_lagging_slash(args.checkpoint_dir)))
    checkpoint_fname = next(fname for fname in os.listdir(checkpoint_dir) if "checkpoint" in fname)
    checkpoint_path = checkpoint_dir / checkpoint_fname

    info_fname = "info.json"
    info = json.load(open((folder_argname / info_fname)))

    device = "cpu"
    agent = make_agent(info['agent_name'], info['env'], device, info['hyperparameters'])
    agent.load_weights(checkpoint_path)

    # generate directions normally
    dir1_vec, dir2_vec = find_unscaled_alts(agent, args.directions)

    # copy directions
    if args.copy_directions is not None:
        dir_path = Path(args.copy_directions)
        dir1_vec = readz(dir_path / "dir1.npz")
        dir2_vec = readz(dir_path / "dir2.npz")
    if args.dir1 is not None:
        dir1_vec = readz(args.dir1)
        info['dir1'] = args.dir1
    if args.dir2 is not None:
        dir2_vec = readz(args.dir2)
        info['dir2'] = args.dir2

    if args.scale_vec is not None:
        scale_vec = readz(args.scale_vec)
        dir1_vec = scale_dir(dir1_vec, scale_vec)
        dir2_vec = scale_dir(dir2_vec, scale_vec)

    if args.magnitude is not None:
        info['magnitude'] = m = args.magnitude
        dir1_vec = [m*v for v in dir1_vec]
        dir2_vec = [m*v for v in dir2_vec]

    info['directions'] = args.directions if args.copy_directions is None else "copy"


    generate_plane_data(args.checkpoint_dir, args.output_path, dir1_vec, dir2_vec, info,
        grid_size=args.grid_size,
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        device=args.device,
        use_offset_critic=args.use_offset_critic,
        est_hesh=args.est_hesh,
        calc_hesh=args.calc_hesh,
        calc_grad=args.calc_grad,
    )


if __name__ == "__main__":
    main()
