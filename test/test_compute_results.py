from reward_surfaces.agents import make_agent
from reward_surfaces.utils import save_results
import os
import pathlib

if __name__ == "__main__":
    agent = make_agent("SB3_ON","Pendulum-v0","cpu",{'ALGO':'PPO'})
    info = { }
    info['num_episodes'] = 10
    info['num_steps'] = 1000
    info['est_hesh'] = False
    info['calc_hesh'] = False
    info['calc_grad'] = False
    folder = "compute_res_test"
    os.makedirs(folder+"/results")
    save_results(agent, info, pathlib.Path(folder), {}, "test_eval")
    info['calc_hesh'] = True
    save_results(agent, info, pathlib.Path(folder), {}, "test_calc_hesh")
