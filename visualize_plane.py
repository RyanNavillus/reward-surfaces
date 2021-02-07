from generate_plane_jobs import generate_plane_data
from visualize import visualize_csv
from run_jobs_multiproc import run_job_list
from job_results_to_csv import job_results_to_csv
from surface_utils import readz
from agents.make_agent import make_agent

import json
import os
import shutil

trained_checkpoint = "train_results/bullet/hopper/0010000"
generated_dirs_dir = "generated_dirs/test_script/"

dir1_fname = "generated_dirs/hopper_eig_vecs/results/0010000/mineigvec.npz"
dir2_fname = "generated_dirs/hopper_eig_vecs/results/0010000/maxeigvec.npz"

dir1 = readz(dir1_fname)
dir2 = readz(dir2_fname)

train_info = json.load(open(generated_dirs_dir+"info.json"))

if os.path.exists(generated_dirs_dir):
    shutil.rmtree(generated_dirs_dir)
print("removed")

generate_plane_data(trained_checkpoint, generated_dirs_dir, dir1, dir2, train_info, num_steps=1000)
run_job_list(generated_dirs_dir+"jobs.sh")
job_results_to_csv(generated_dirs_dir)
visualize_csv(generated_dirs_dir+"results.csv")
