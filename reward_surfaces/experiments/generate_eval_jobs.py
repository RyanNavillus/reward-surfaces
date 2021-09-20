import json
import os
import numpy as np
from pathlib import Path


def generate_eval_jobs(train_dir, out_dir,
                       num_steps=None,
                       num_episodes=None,
                       est_hesh=False,
                       est_grad=False,
                       calc_hesh=False,
                       calc_grad=False,
                       device="cpu",
                       checkpoint=None):
    assert not (est_hesh and calc_hesh), "calculating and estimating hessian cannot happen at the same time"
    assert num_steps is not None or num_episodes is not None, "one of num_steps or num_episodes must be specified"
    if num_steps is None:
        num_steps = 10000000000000
    if num_episodes is None:
        num_episodes = 10000000000000

    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    # TODO: Should we set exist_ok=True?
    os.makedirs(out_dir / "results")

    info_fname = "info.json"
    info = json.load(open((train_dir / info_fname)))

    info['num_episodes'] = num_episodes
    info['num_steps'] = num_steps
    info['est_hesh'] = est_hesh
    info['est_grad'] = est_grad
    info['calc_hesh'] = calc_hesh
    info['calc_grad'] = calc_grad

    json.dump(info, open((out_dir / info_fname), 'w'), indent=4)

    checkpoints = [folder for folder in os.listdir(train_dir) if os.path.isdir(train_dir / folder) and folder.isdigit()]
    checkpoints = sorted(checkpoints)

    # Limit to 30 checkpoints for line plots
    if len(checkpoints) > 30:
        print(f"Selecting 30 checkpoints out of {len(checkpoints)}")
        idx = list(np.round(np.linspace(0, len(checkpoints) - 1, 30)).astype(int))
        checkpoints = [checkpoints[i] for i in idx]

    # Always eval best checkpoint
    if "best" in os.listdir(train_dir) and os.path.isdir(train_dir / "best"):
        checkpoints.append("best")

    all_jobs = []
    if checkpoint:
        job = f"python3 -m reward_surfaces.bin.eval_tradj {train_dir} {checkpoint} {out_dir} --device={device}"
        all_jobs.append(job)
    else:
        for checkpt in checkpoints:
            job = f"python3 -m reward_surfaces.bin.eval_tradj {train_dir} {checkpt} {out_dir} --device={device}"
            all_jobs.append(job)

    jobs = "\n".join(all_jobs)+"\n"
    open((out_dir / "jobs.sh"), 'w').write(jobs)
