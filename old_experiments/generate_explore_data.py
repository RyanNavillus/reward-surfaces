import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from generate_test_files import generate_test_file
from test_env import test_env_true_reward
from generate_random_filter import generate_random_directions
import os
import multiprocessing as mp
import numpy as np
import shutil
import tempfile

from generate_data import get_data_path

def generate_env_run(algo, env_name):
    pass

def get_rewards(args):
    (data_path, ext, env_name, algo, num_episodes, max_frames, xwin, ywin, xpos, ypos) = args
    with tempfile.NamedTemporaryFile(suffix=ext) as file:
        generate_test_file(data_path, file.name, xwin, ywin, xpos, ypos)
        print("genereated")
        #return test_env_true_reward(file.name, env_name, algo, num_episodes, max_frames)


def run_on_algo(source_file, num_episodes, max_frames, x_window, y_window):
    data_folder, output_folder, algo, env_name = get_data_path(source_file)

    for x_pos in range(x_window):
        for y_pos in range(y_window):
            #print(data_fname)
            args = (data_folder, ext, env_name, algo, num_episodes, max_frames, x_window, y_window, x_pos, y_pos)
            all_args.append(args)

    for args in all_args:
        get_rewards(args)

if __name__ == "__main__":
    data_of_interest = open("envs.txt").readlines()
    data_of_interest = [line.strip() for line in data_of_interest]
    data_of_interest = data_of_interest#[:60]
    data_of_interest = [name for name in data_of_interest if "NoFrameskip-v4" in name]
    #data_of_interest = data_of_interest[::-1]
    data_of_interest = [
        "trained_agents/ppo2/BeamRiderNoFrameskip-v4.pkl",
        "trained_agents/ppo2/BreakoutNoFrameskip-v4.pkl",
        "trained_agents/ppo2/EnduroNoFrameskip-v4.pkl",
        "trained_agents/ppo2/MsPacmanNoFrameskip-v4.pkl",
        "trained_agents/ppo2/PongNoFrameskip-v4.pkl",
        "trained_agents/ppo2/QbertNoFrameskip-v4.pkl",
        "trained_agents/ppo2/SeaquestNoFrameskip-v4.pkl",
        "trained_agents/ppo2/SpaceInvadersNoFrameskip-v4.pkl"
    ]
    load_file = data_of_interest[0]


    #[
        #"trained_agents/trpo/LunarLander-v2.pkl",
        #"trained_agents/ppo2/LunarLander-v2.pkl",
        # "trained_agents/ddpg/BipedalWalker-v2.pkl",
        # "trained_agents/td3/BipedalWalker-v2.zip",
        # "trained_agents/dqn/QbertNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/QbertNoFrameskip-v4.pkl",
    #]
    x_window = 3
    y_window = 3

    run_on_algo(load_file, 5, max_frames, x_window, y_window)
