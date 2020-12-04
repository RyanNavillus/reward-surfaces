from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from extract_params import ParamLoader
import tempfile
import time
import torch
from collections import OrderedDict
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
import os

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.old_params = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls - 1 % self.save_freq == 0:
            self.old_params = [param.clone() for param in self.model.policy.parameters()]
        if self.n_calls % self.save_freq == 0 and self.old_params is not None:
            print("saved")
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            os.makedirs(path, exist_ok=True)
            self.model.save(os.path.join(path,"checkpoint"))
            model_parameters = self.model.policy.state_dict()
            grads = OrderedDict([(name, param.grad) for name, param in model_parameters.items()])
            delta = OrderedDict([(name, param - old_param) for old_param, (name, param) in zip(self.old_params, model_parameters.items())])
            torch.save(model_parameters,os.path.join(path,"parameters.th"))
            torch.save(grads,os.path.join(path,"grads.th"))
            torch.save(delta,os.path.join(path,"prev_step.th"))

            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class SB3OnPolicyTrainer:
    def __init__(self, env_fn, sb3_algorithm):
        self.env_fn = env_fn
        self.algorithm = sb3_algorithm
        self.device = sb3_algorithm.device
        # print([param[0] for param in self.parameters])

    def train(self, num_steps, save_dir, save_freq=1000):
        save_prefix = f'sb3_{type(self.algorithm).__name__}'
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_dir,
                                                 name_prefix=save_prefix)
        self.algorithm.learn(num_steps,callback=checkpoint_callback)
        saved_files = [f"{save_dir}/{save_prefix}_{i}_steps/checkpoint.zip" for i in range(save_freq,num_steps,save_freq)]
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

        return mean(episode_rewards),mean(episode_value_ests),mean(episode_values),mean(episode_td_err)

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


class SB3OffPolicyTrainer(SB3OnPolicyTrainer):
    def to_agent_obs(self, obs):
        return torch.as_tensor(obs)

    def evaluate(self, num_episodes, get_convexity=False, num_envs=1, eval_trainer=None):
        env = DummyVecEnv([self.env_fn]*num_envs)
        policy_policy = self.algorithm.policy
        gamma = self.algorithm.gamma
        eval_policy = policy_policy if eval_trainer is None else eval_trainer.policy

        obs = env.reset()
        obs = self.to_agent_obs(obs)

        ep_rews = [[] for i in range(num_envs)]
        ep_vals = [[] for i in range(num_envs)]
        episode_rewards = []
        episode_value_ests = []
        episode_values = []
        episode_td_err = []
        start_t = time.time()
        while len(episode_rewards) < num_episodes:
            action = policy_policy.forward(obs)#, deterministic=True)
            value = eval_policy.critic.forward(obs, action)

            action = action.detach().numpy()
            obs, rew, done, info = env.step(action)
            obs = self.to_agent_obs(obs)

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

        return mean(episode_rewards),mean(episode_value_ests),mean(episode_values),mean(episode_td_err)


class SB3HerPolicyTrainer(SB3OffPolicyTrainer):
    def __init__(self, env_fn, sb3_algorithm):
        super().__init__(env_fn, sb3_algorithm.model)
        self.her_algo = sb3_algorithm
        self._base_model = sb3_algorithm.model

    def to_agent_obs(self, obs):
        return torch.as_tensor(ObsDictWrapper.convert_dict(obs))

    def train(self, num_steps, save_dir, save_freq=1000):
        self.algorithm = self.her_algo
        values = super().train(num_steps, save_dir, save_freq)
        self.algorithm = self._base_model
        return values
