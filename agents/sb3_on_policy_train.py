from stable_baselines3.common.callbacks import CheckpointCallback
from extract_params import ParamLoader
import tempfile
import time
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

class SB3OnPolicyTrainer:
    def __init__(self, env_fn, sb3_algorithm):
        self.env_fn = env_fn
        self.algorithm = sb3_algorithm
        self.device = sb3_algorithm.device

    def train(self, num_steps, save_dir, save_freq=1000):
        save_prefix = f'sb3_{type(self.algorithm).__name__}'
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_dir,
                                                 name_prefix=save_prefix)
        self.algorithm.learn(num_steps,callback=checkpoint_callback)
        saved_files = [f"{save_dir}/{save_prefix}_{i}_steps.zip" for i in range(save_freq,num_steps,save_freq)]
        return saved_files

    def get_weights(self):
        with tempfile.NamedTemporaryFile(suffix=".zip") as save_file:
            fname = save_file.name
            self.save_weights(fname)
            loader = ParamLoader(fname)
        return loader.get_params()

    def set_weights(self, params):
        with tempfile.NamedTemporaryFile(suffix=".zip") as save_file:
            fname = save_file.name
            self.save_weights(fname)
            loader = ParamLoader(fname)
            loader.set_params(params)
            loader.save(fname)
            self.load_weights(fname)

    def load_weights(self, load_file):
        self.algorithm = type(self.algorithm).load(load_file, self.env_fn(), self.device)

    def save_weights(self, save_file):
        self.algorithm.save(save_file)

    def evaluate(self, num_episodes, get_convexity=False, num_envs=2, eval_trainer=None):
        env = DummyVecEnv([self.env_fn]*num_envs)
        policy_policy = self.algorithm.policy
        gamma = self.algorithm.gamma

        obs = env.reset()
        obs = torch.as_tensor(obs)

        ep_rews = [[] for i in range(num_envs)]
        ep_vals = [[] for i in range(num_envs)]
        episode_rewards = []
        episode_value_ests = []
        episode_values = []
        episode_td_err = []
        start_t = time.time()
        while len(episode_rewards) < num_episodes:
            action, policy_val, policy_log_prob = policy_policy.forward(obs)#, deterministic=True)

            if eval_trainer is None:
                value = policy_val
            else:
                eval_policy = eval_trainer.algorithm.policy
                value, eval_log_prob, eval_entropy = eval_policy.evaluate_actions(obs, action)

            action = action.detach().numpy()
            obs, rew, done, info = env.step(action)
            obs = torch.as_tensor(obs)

            for i in range(num_envs):
                ep_vals[i].append(value[i])
                ep_rews[i].append(rew[i])

            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(sum(ep_rews[i]))
                    episode_value_ests.append(mean(ep_vals[i]))
                    episode_values.append(calc_mean_value(ep_rews[i], gamma))
                    episode_td_err.append(calc_mean_td(ep_vals[i], ep_rews[i], gamma))#TODO make this real

                    ep_rews[i] = []
                    ep_vals[i] = []
                    end_t = time.time()
                    print("done!", (end_t - start_t)/len(episode_rewards))

        return mean(episode_rewards),mean(episode_value_ests),mean(episode_values)

def mean(vals):
    return sum(vals)/len(vals)

def calc_mean_value(rews, gamma):
    decayed_rew = 0
    tot_value = 0
    for i in range(len(rews)-1,-1,-1):
        decayed_rew += rews[i]
        tot_value += decayed_rew
        decayed_rew *= gamma
    return tot_value / len(rews)

def calc_mean_td(est_vals, rewards, gamma):
    assert len(est_vals) == len(rewards)
    ep_len = len(est_vals)
    td_errs = []
    for i in range(ep_len):
        true_val = est_vals[i+1] * gamma + rewards[i] if i < ep_len-1 else rewards[i]
        est_val = est_vals[i]
        td_err = (true_val - est_val)**2
        td_errs.append(td_err)
    return mean(td_errs)
