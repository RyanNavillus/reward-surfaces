from reward_surfaces.experiments import generate_plane_data
from reward_surfaces.plotting import plot_plane
from reward_surfaces.runners.run_jobs_multiproc import run_job_list
from reward_surfaces.utils.job_results_to_csv import job_results_to_csv
from reward_surfaces.utils.surface_utils import readz, filter_normalize
from reward_surfaces.agents.make_agent import make_agent

import json
import os
import shutil
import torch
import sklearn.decomposition
from pathlib import Path
import numpy as np
from reward_surfaces.algorithms.evaluate_est_hesh import npvec_to_nplist


# points = [1,4,8,16,32,64,128,256,450]
# for point in points:
#     p_str = f"{point:03d}0000"
def plot_contour(train_dir):
    ant_train_points = Path(train_dir)
    env_name = os.path.basename(ant_train_points)
    generated_dirs_dir = f"generated_dirs/_temp_dir/"

    if os.path.exists(generated_dirs_dir):
        shutil.rmtree(generated_dirs_dir)
    checkpoint_names = [checkpoint for checkpoint in os.listdir(ant_train_points) if checkpoint[-1] == '0']
    checkpoint_names.sort()

    vectors = []
    for checkpoint in (checkpoint_names):
        check_param_fname = ant_train_points / checkpoint / "parameters.th"
        params = torch.load(check_param_fname,map_location=torch.device('cpu')).values()
        dir1 = [v.detach().flatten() for v in params]
        vector = torch.cat(dir1,axis=0).numpy()
        vectors.append(vector)

    decomposer = sklearn.decomposition.PCA(2)
    coords = decomposer.fit_transform(vectors)
    # center points at final coordinate, i.e. center of plot
    coords -= coords[-1]
    dir1, dir2 = decomposer.components_
    dir1 = npvec_to_nplist(dir1, params)
    dir2 = npvec_to_nplist(dir2, params)
    scale1,scale2 = np.max(np.abs(coords),axis=0)*1.15

    dir1 = [d*scale1 for d in dir1]
    dir2 = [d*scale2 for d in dir2]

    num_episodes = 25
    train_info = json.load(open(ant_train_points/"info.json"))
    base_checkpoint = str(ant_train_points / checkpoint_names[-1])
    generate_plane_data(base_checkpoint, generated_dirs_dir, dir1, dir2, train_info, grid_size=15, num_episodes=num_episodes)
    run_job_list(generated_dirs_dir+"jobs.sh")
    job_results_to_csv(generated_dirs_dir)
    plot_plane(generated_dirs_dir+"results.csv",
        outname="contours/"+env_name,
        type="contour",
        dir1_name="component 1",
        dir2_name="component 2",
        dir1_scale=scale1,
        dir2_scale=scale2,
        points=coords,
    )

if __name__ == "__main__":
    plot_contour("train_results/muj_res/ant/")
    plot_contour("train_results/muj_res/half_cheetah/")
    plot_contour("train_results/muj_res/hopper/")
    plot_contour("train_results/muj_res/humanoid/")
    plot_contour("train_results/muj_res/inv_double_pendulum/")
    plot_contour("train_results/muj_res/inv_pendulum/")
    plot_contour("train_results/muj_res/humanoid_flagrun/")
