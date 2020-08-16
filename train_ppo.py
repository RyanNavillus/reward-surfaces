from ppo import PPO2
import json
from eval_ppo import test_env_true_reward
from create_env import create_env
import os
from stable_baselines.common.callbacks import EventCallback

class EvalCallback(EventCallback):
    def __init__(self, eval_freq, eval_fn):
        super(EvalCallback, self).__init__(None, verbose=False)
        self.eval_freq = eval_freq
        self.eval_fn = eval_fn

    def _init_callback(self):
        self.learn_step = 0
        self.save_step = 0
        print("Started eval callback")

    def _on_step(self):
        if self.learn_step % self.eval_freq == 0:
            self.eval_fn(self.save_step)
            self.save_step += 1
            print("eval callback called!")

        self.learn_step += 1

def train_ppo(env_name, eval_freq, hyperparams, save_dir):
    os.mkdir(save_dir)
    model_saves = os.path.join(save_dir, "models")
    os.mkdir(model_saves)
    n_train_envs = hyperparams.pop("n_train_envs", 4)
    network = hyperparams.pop("network", "default")
    max_frames = hyperparams.pop("max_frames", 10000)
    if network == "default":
        policy = "CnnPolicy"
    else:
        raise ValueError("bad network value")
    eval_results_file = open(os.path.join(save_dir, "results.csv"),'w')
    eval_results_file.write("rew, value_ests, values, td_err\n")
    eval_results_file.flush()
    with open(os.path.join(save_dir, "hyperparams.json"),'w') as file:
        file.write(json.dumps(hyperparams, indent=4))
    eval_num_done = 4
    eval_num_steps = None
    n_timesteps = 60000
    log_interval = eval_freq
    algo_name = "ppo2"

    tensorboard_log = "tb_log"
    load_env, train_env  = create_env(env_name, algo_name, n_envs=n_train_envs, max_frames=max_frames)
    load_env, test_env  = create_env(env_name, algo_name, n_envs=4, max_frames=max_frames)
    model = PPO2(policy="CnnPolicy",env=train_env, tensorboard_log=tensorboard_log, verbose=False, **hyperparams)

    if log_interval > -1:
        kwargs = {'log_interval': log_interval}

    # Account for the number of parallel environments
    eval_freq = max(eval_freq // n_train_envs, 1)

    def eval_fn(save_step):
        model_save = os.path.join(model_saves, f"{save_step}.pkl")
        model.save(model_save)
        eval_results = test_env_true_reward(model_save, model_save, env_name, eval_num_done, max_frames)
        result_str = ", ".join([str(r) for r in eval_results])+"\n"
        eval_results_file.write(result_str)
        eval_results_file.flush()

    # Do not normalize the rewards of the eval env
    eval_callback = EvalCallback(eval_freq, eval_fn)
    callbacks = [eval_callback]

    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    model.learn(n_timesteps, **kwargs)

if __name__ == "__main__":
    env_name = "BeamRiderNoFrameskip-v4"
    train_ppo(env_name, 16, hyperparams={}, save_dir="test_save")
