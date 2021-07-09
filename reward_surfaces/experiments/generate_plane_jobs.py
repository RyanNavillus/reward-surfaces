import argparse
import torch
import json
import os
import shutil
from pathlib import Path
import numpy as np
from reward_surfaces.utils.surface_utils import readz
from reward_surfaces.utils.path_utils import strip_lagging_slash


def generate_plane_data(checkpoint_dir,
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
                        est_grad=False,
                        calc_hesh=False,
                        calc_grad=False):

    assert isinstance(dir1_vec, list) and isinstance(dir1_vec[0], np.ndarray), "dir1 and dir2 must be a list of numpy vectors. Use `from surface_utils import readz; readz(fname)` to read a numpy vector into this format"
    assert isinstance(dir2_vec, list) and isinstance(dir2_vec[0], np.ndarray), "dir1 and dir2 must be a list of numpy vectors. Use `from surface_utils import readz; readz(fname)` to read a numpy vector into this format"
    assert grid_size % 2 == 1, "grid-size must be odd"
    assert num_steps is not None or num_episodes is not None, "one of num_steps or num_episodes must be specified"
    if num_steps is None:
        num_steps = 10000000000000
    if num_episodes is None:
        num_episodes = 10000000000000

    output_path = Path(output_path)
    folder_argname = Path(os.path.dirname(strip_lagging_slash(str(checkpoint_dir))))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_fname = next(fname for fname in os.listdir(checkpoint_dir) if "checkpoint" in fname)
    checkpoint_path = checkpoint_dir / checkpoint_fname
    info_fname = "info.json"
    params_fname = "parameters.th"

    os.makedirs(output_path, exist_ok=True)
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
    info['est_grad'] = est_grad
    info['calc_grad'] = calc_grad

    json.dump(info, open((output_path / info_fname), 'w'), indent=4)

    job_out_path = os.mkdir((output_path / "results"))
    seperate_eval_arg = " --use_offset_critic " if use_offset_critic else ""

    job_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = i - grid_size // 2
            y = j - grid_size // 2
            job = f"python3 -m reward_surfaces.bin.eval_plane_job {output_path} --offset1={x}  --offset2={y} {seperate_eval_arg}"
            #print(job)
            job_list.append(job)

    jobs = "\n".join(job_list)+"\n"
    open((output_path / "jobs.sh"), 'w').write(jobs)
