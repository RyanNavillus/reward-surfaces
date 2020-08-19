import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from old_experiments.generate_test_files import generate_test_file
from eval_ppo import test_env_true_reward
from old_experiments.generate_random_filter import generate_random_directions
import os
import multiprocessing as mp
import numpy as np
import shutil
import json
import tempfile

def extension(path):
    return "."+path.strip().split(".")[-1]

def generate_data(source_file, dest_folder):
    data_folder = dest_folder
    os.makedirs(data_folder)
    ext = extension(source_file)
    new_base_data = os.path.join(data_folder,"base_params"+ext)
    shutil.copyfile(source_file, new_base_data)
    generate_random_directions(new_base_data, data_folder)

def get_rewards(args):
    (data_path, eval_file, ext, env_name, algo, num_episodes, max_frames, xwin, ywin, xpos, ypos) = args
    with tempfile.NamedTemporaryFile(suffix=ext) as file:
        generate_test_file(data_path, file.name, xwin, ywin, xpos, ypos)
        return test_env_true_reward(file.name, eval_file, env_name, num_episodes, max_frames, single_threaded=True)

def run_on_data(data_folder, hyperparams, num_episodes, x_window, y_window):
    algo = "ppo2"
    env_name = hyperparams['env_name']
    max_frames = hyperparams.pop('max_frames',10000)
    ext = ".zip"
    eval_file = os.path.join(data_folder, "base_params"+ext)
    #ext = extension(source_file)

    fold_name = f"{x_window},{y_window}_{num_episodes}_{max_frames}"
    output_folder = os.path.join(data_folder, "maps", fold_name)
    os.makedirs(output_folder)

    all_args = []
    for x_pos in range(x_window):
        for y_pos in range(y_window):
            #print(data_fname)
            args = (data_folder, eval_file, ext, env_name, algo, num_episodes, max_frames, x_window, y_window, x_pos, y_pos)
            all_args.append(args)

    #results = [sing_src_rew_eval(arg) for arg in all_args]
    pool = mp.Pool(24)
    results = list(pool.map(get_rewards, all_args))
    print("fetched results")
    true_rewards = [true_rew for true_rew, est_val, true_val, td_err in results]
    est_vals = [est_val for true_rew, est_val, true_val, td_err in results]
    true_vals = [true_val for true_rew, est_val, true_val, td_err in results]
    td_errs = [td_err for true_rew, est_val, true_val, td_err in results]

    #results = np.zeros(144)
    def save_data(results, name):
        results = np.asarray(results).reshape([x_window, y_window]).T
        result_fname = f"{output_folder}/{name}.npy"
        np.save(result_fname,results)
        print(f"saved {result_fname}")
    save_data(true_rewards, "true_values")
    save_data(est_vals, "mean_est_values")
    save_data(true_vals, "mean_values")
    save_data(td_errs, "td_errs")

def run_on_train_run(max_step, folder, num_episodes, x_window, y_window):
    hyperparams = os.path.join(folder, "hyperparams.json")
    hyperparams = json.load(open(hyperparam_path))
    for i in range(max_step):
        model_path = os.path.join(folder, "models", f"{i}.zip")
        output_path = os.path.join("generated_dirs","test3",str(i))

        try:
            generate_data(model_path, output_path)
        except FileExistsError:
            pass
        try:
            run_on_data(output_path, hyperparams, num_episodes, x_window, y_window)
        except FileExistsError:
            pass


if __name__ == "__main__":
    x_window = 17
    y_window = 17
    num_episodes = 15

    model_path = "train_results/test_save3/models/0.zip"
    hyperparam_path = "train_results/test_save3/hyperparams.json"
    output_path = "generated_dirs/test3/"

    run_on_train_run(10, "train_results/test_save3/", num_episodes, x_window, y_window )
