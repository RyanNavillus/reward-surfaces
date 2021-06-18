import json
import os
from pathlib import Path


def generate_eval_jobs(train_dir, out_dir,
                       num_steps=None,
                       num_episodes=None,
                       est_hesh=False,
                       est_grad=False,
                       calc_hesh=False,
                       calc_grad=False,
                       device="cpu"):
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

    checkpoints = [dir for dir in os.listdir(train_dir) if os.path.isdir(train_dir / dir) and dir.isdigit()]
    all_jobs = []
    for checkpoint in checkpoints:
        job = f"python -m reward_surfaces.bin.eval_tradj {train_dir} {checkpoint} {out_dir} --device={device}"
        all_jobs.append(job)

    jobs = "\n".join(all_jobs)+"\n"
    open((out_dir / "jobs.sh"), 'w').write(jobs)
