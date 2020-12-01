from ppo import PPO2
import json
from eval_ppo import test_env_true_reward_loaded
from create_env import create_env
import os
import sys
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
            print("eval callback started!")
            self.eval_fn(self.save_step)
            self.save_step += 1
            print("eval callback finished!")

        self.learn_step += 1

def train_ppo(eval_freq, hyperparams, save_dir, eval_num_done=4, n_timesteps=100000000):
    os.mkdir(save_dir)
    with open(os.path.join(save_dir, "hyperparams.json"),'w') as file:
        file.write(json.dumps(hyperparams, indent=4))
    model_saves = os.path.join(save_dir, "models")
    os.mkdir(model_saves)
    env_name = hyperparams.pop("env_name")
    n_train_envs = hyperparams.pop("n_train_envs", 16)
    network = hyperparams.pop("network", "default")
    max_frames = hyperparams.pop("max_frames", 10000)
    max_frames = hyperparams.pop("n_timesteps", 10000000)
    policy = network
    eval_results_file = open(os.path.join(save_dir, "results.csv"),'w')
    eval_results_file.write("rew, value_ests, values, td_err\n")
    eval_results_file.flush()

    log_interval = eval_freq
    algo_name = "ppo2"

    tensorboard_log = "tb_log"
    train_env  = create_env(env_name, algo_name, n_envs=n_train_envs, max_frames=max_frames, multiproc=True)
    test_env  = create_env(env_name, algo_name, n_envs=8, max_frames=max_frames, multiproc=True)
    model = PPO2(policy="CnnPolicy",env=train_env, tensorboard_log=tensorboard_log, verbose=False, **hyperparams)

    if log_interval > -1:
        kwargs = {'log_interval': log_interval}

    # Account for the number of parallel environments
    eval_freq = max(eval_freq // n_train_envs, 1)

    def eval_fn(save_step):
        model_save = os.path.join(model_saves, f"{save_step}.zip")
        model.save(model_save)
        eval_results = test_env_true_reward_loaded(model, model, test_env, eval_num_done)
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
    eval_freq = 10000
    hyperparams = {
      'env_name': env_name,
      'network': 'CnnPolicy',
      'n_train_envs': 8,
      'n_steps': 16,
      'noptepochs': 4,
      'nminibatches': 4,
      'n_timesteps': 1e7,
      'learning_rate': 2.5e-4,
      'cliprange': 0.1,
      'vf_coef': 0.5,
      'ent_coef': 0.01,
      'cliprange_vf': -1,
    }
    num = sys.argv[1]

    train_ppo(eval_freq, hyperparams=hyperparams, save_dir=f"train_results/test_save{num}/")
