import os
import json
import pathlib
import torch
from reward_surfaces.agents import make_agent
from reward_surfaces.utils.compute_results import save_results
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


def main():
    base_source_path = pathlib.Path("./reward_surfaces/algorithms/test_data/")
    checkpoint_fname = "mountaincar_test_checkpoints/best/checkpoint"
    checkpoint_path = base_source_path / checkpoint_fname
    info_fname = "mountaincar_test_eig_vecs/info.json"
    info = json.load(open(base_source_path / info_fname))
    agent, _ = make_agent(info['agent_name'], info['env'], "cpu", "./test_data", info['hyperparameters'])
    agent.load_weights(checkpoint_path)

    job_name = "test"
    cur_results = {
        "dim0": 0,
        "dim1": 0,
        "scale": 5,
        "magnitude": 1
    }
    save_results(agent, info, base_source_path, cur_results, job_name)


if __name__ == "__main__":
    main()
