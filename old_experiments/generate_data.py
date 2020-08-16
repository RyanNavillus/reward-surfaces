import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from generate_test_files import generate_test_file
from test_env import test_env_true_reward
from generate_random_filter import generate_random_directions
import os
import multiprocessing as mp
import numpy as np
import shutil
import json
import tempfile

def get_data_path(source_file):
    source_path = source_file.split("/")
    algo = source_path[-2]
    env_name = source_path[-1][:-4]
    data_folder = os.path.join("generated_dirs", algo, env_name)
    out_folder = os.path.join("output_files", algo, env_name)
    return data_folder+"/", out_folder+"/", algo, env_name

def extension(path):
    return "."+path.strip().split(".")[-1]

def generate_data(source_file, x_window, y_window):
    data_folder, _, algo, env_name = get_data_path(source_file)
    os.makedirs(data_folder)
    ext = extension(source_file)
    new_base_data = os.path.join(data_folder,"base_params"+ext)
    shutil.copyfile(source_file, new_base_data)
    generate_random_directions(new_base_data, data_folder)

def get_rewards(args):
    (data_path, ext, env_name, algo, num_episodes, max_frames, xwin, ywin, xpos, ypos) = args
    with tempfile.NamedTemporaryFile(suffix=ext) as file:
        generate_test_file(data_path, file.name, xwin, ywin, xpos, ypos)
        print("genereated")
        return test_env_true_reward(file.name, env_name, algo, num_episodes, max_frames)

def run_on_data(source_file, num_episodes, max_frames, x_window, y_window):
    data_folder, output_folder, algo, env_name = get_data_path(source_file)
    ext = extension(source_file)

    all_args = []
    for x_pos in range(x_window):
        for y_pos in range(y_window):
            #print(data_fname)
            args = (data_folder, ext, env_name, algo, num_episodes, max_frames, x_window, y_window, x_pos, y_pos)
            all_args.append(args)

    #results = [sing_src_rew_eval(arg) for arg in all_args]
    print(f"pooled result: {len(all_args)}")
    print(f"pooled result: {x_window}")
    print(f"pooled result: {y_window}")
    print(f"pooled result: {len(all_args)}")
    print(f"pooled result: {len(all_args)}")
    print(f"pooled result: {len(all_args)}")
    pool = mp.Pool(6)
    results = list(pool.map(get_rewards, all_args))
    print("fetched results")
    true_rewards = [true_rew for true_rew, est_val, true_val in results]
    est_vals = [est_val for true_rew, est_val, true_val in results]
    true_vals = [true_val for true_rew, est_val, true_val in results]
    os.makedirs(output_folder,exist_ok=True)
    #results = np.zeros(144)
    def save_data(results, name):
        results = np.asarray(results).reshape([x_window, y_window]).T
        result_fname = f"{output_folder}/{name}_<{x_window},{y_window}>_{num_episodes}_{max_frames}.npy"
        np.save(result_fname,results)
        print(f"saved {result_fname}")
    save_data(true_rewards, "true_values")
    save_data(est_vals, "mean_est_values")
    save_data(true_vals, "mean_values")

if __name__ == "__main__":
    data_of_interest = open("envs.txt").readlines()
    data_of_interest = [line.strip() for line in data_of_interest]
    data_of_interest = data_of_interest#[:60]
    data_of_interest = [name for name in data_of_interest if "NoFrameskip-v4" in name]
    #data_of_interest = data_of_interest[::-1]
    data_of_interest = [
        "trained_agents/a2c/BeamRiderNoFrameskip-v4.pkl",
        "trained_agents/a2c/BreakoutNoFrameskip-v4.pkl",
        "trained_agents/a2c/EnduroNoFrameskip-v4.pkl",
        "trained_agents/a2c/MsPacmanNoFrameskip-v4.pkl",
        "trained_agents/a2c/PongNoFrameskip-v4.pkl",
        "trained_agents/a2c/QbertNoFrameskip-v4.pkl",
        "trained_agents/a2c/SeaquestNoFrameskip-v4.pkl",
        "trained_agents/a2c/SpaceInvadersNoFrameskip-v4.pkl",
        "trained_agents/acer/BeamRiderNoFrameskip-v4.pkl",
        "trained_agents/acer/EnduroNoFrameskip-v4.pkl",
        "trained_agents/acer/MsPacmanNoFrameskip-v4.pkl",
        "trained_agents/acer/PongNoFrameskip-v4.pkl",
        "trained_agents/acer/QbertNoFrameskip-v4.pkl",
        "trained_agents/acer/SeaquestNoFrameskip-v4.pkl",
        "trained_agents/acer/SpaceInvadersNoFrameskip-v4.pkl",
        "trained_agents/acktr/BeamRiderNoFrameskip-v4.pkl",
        "trained_agents/acktr/BreakoutNoFrameskip-v4.pkl",
        "trained_agents/acktr/EnduroNoFrameskip-v4.pkl",
        "trained_agents/acktr/MsPacmanNoFrameskip-v4.pkl",
        "trained_agents/acktr/PongNoFrameskip-v4.pkl",
        "trained_agents/acktr/QbertNoFrameskip-v4.pkl",
        "trained_agents/acktr/SeaquestNoFrameskip-v4.pkl",
        "trained_agents/acktr/SpaceInvadersNoFrameskip-v4.pkl",
        "trained_agents/dqn/BeamRiderNoFrameskip-v4.pkl",
        "trained_agents/dqn/BreakoutNoFrameskip-v4.pkl",
        "trained_agents/dqn/EnduroNoFrameskip-v4.pkl",
        "trained_agents/dqn/MsPacmanNoFrameskip-v4.pkl",
        "trained_agents/dqn/PongNoFrameskip-v4.pkl",
        "trained_agents/dqn/QbertNoFrameskip-v4.pkl",
        "trained_agents/dqn/SeaquestNoFrameskip-v4.pkl",
        "trained_agents/dqn/SpaceInvadersNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/BeamRiderNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/BreakoutNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/EnduroNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/MsPacmanNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/PongNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/QbertNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/SeaquestNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/SpaceInvadersNoFrameskip-v4.pkl"
    ]
    print(json.dumps(data_of_interest,indent=4))
    #exit(0)
    #[
        #"trained_agents/trpo/LunarLander-v2.pkl",
        #"trained_agents/ppo2/LunarLander-v2.pkl",
        # "trained_agents/ddpg/BipedalWalker-v2.pkl",
        # "trained_agents/td3/BipedalWalker-v2.zip",
        # "trained_agents/dqn/QbertNoFrameskip-v4.pkl",
        # "trained_agents/ppo2/QbertNoFrameskip-v4.pkl",
    #]
    x_window = 17
    y_window = 17
    for data in data_of_interest:
        try:
            generate_data(data, x_window, y_window)
        except FileExistsError:
            pass
    for data in data_of_interest:
        # folder_path = get_data_path(data)
        # print(folder_path)
        max_frames = 10000
        run_on_data(data, 50, max_frames, x_window, y_window)
