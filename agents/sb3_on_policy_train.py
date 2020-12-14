from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from .extract_params import ParamLoader
import tempfile
import time
import torch
from collections import OrderedDict
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
import os
from .evaluate import evaluate

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
        self.save_folders = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls > 0 and (self.n_calls - 1) % self.save_freq == 0:
            self.old_params = [param.clone() for param in self.model.policy.parameters()]
        if self.n_calls % self.save_freq == 1:
            print(f"saved checkpoint {self.n_calls}")
            path = os.path.join(self.save_path, f"{self.num_timesteps-1:07}")
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path,"checkpoint")
            self.model.save(save_path)
            self.save_folders.append(save_path)
            model_parameters = self.model.policy.state_dict()
            grads = OrderedDict([(name, param.grad) for name, param in model_parameters.items()])
            torch.save(model_parameters,os.path.join(path,"parameters.th"))
            torch.save(grads,os.path.join(path,"grads.th"))

            if self.old_params is not None:
                delta = OrderedDict([(name, param - old_param) for old_param, (name, param) in zip(self.old_params, model_parameters.items())])
                torch.save(delta,os.path.join(path,"prev_step.th"))

            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class OnPolicyEvaluator:
    def __init__(self, vec_env, gamma, algo, eval_trainer):
        env = vec_env
        self.state = env.reset()
        self.env = env
        self.gamma = gamma
        self.algo = algo
        self.eval_trainer = eval_trainer

    def _next_state(self):
        policy_policy = self.algo.policy

        obs = torch.as_tensor(self.state, device=self.algo.device)
        action, policy_val, policy_log_prob = policy_policy.forward(obs)#, deterministic=True)
        if self.eval_trainer is None:
            value = policy_val.item()
        else:
            eval_policy = self.eval_trainer.algorithm.policy
            value, eval_log_prob, eval_entropy = eval_policy.evaluate_actions(obs, action)
            value = value.item()

        action = action.detach().cpu().numpy()
        self.state, rew, done, info = self.env.step(action)
        return rew[0], done[0], value


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
        return checkpoint_callback.save_folders

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

    def calculate_eigenvalues(self, num_steps, tol=1e-2):
        return self.algorithm.calculate_hesh_eigenvalues(num_steps,tol)

    def evaluate(self, num_episodes, num_steps, eval_trainer=None):
        evaluator = OnPolicyEvaluator(self.env_fn(), self.algorithm.gamma, self.algorithm, eval_trainer)
        return evaluate(evaluator, num_episodes, num_steps)


class OffPolicyEvaluator(OnPolicyEvaluator):
    def _next_state(self):
        policy_policy = self.algo.policy
        eval_policy = policy_policy if self.eval_trainer is None else self.eval_trainer.algorithm.policy

        obs = torch.as_tensor(self.state, device=self.algo.device)
        action = policy_policy.forward(obs)#, deterministic=True)
        value = eval_policy.critic.forward(obs, action)

        action = action.detach().cpu().numpy()
        self.state, rew, done, info = self.env.step(action)
        return rew[0], done[0], value[0].item()


class SB3OffPolicyTrainer(SB3OnPolicyTrainer):
    def evaluate(self, num_episodes, num_steps, num_envs=1, eval_trainer=None):
        evaluator = OffPolicyEvaluator(self.env_fn(), self.algorithm.gamma, self.algorithm, eval_trainer)
        return evaluate(evaluator, num_episodes, num_steps)


class HERPolicyEvaluator(OnPolicyEvaluator):
    def _next_state(self):
        policy_policy = self.algo.policy
        eval_policy = policy_policy if self.eval_trainer is None else self.eval_trainer.algorithm.policy

        obs = torch.as_tensor(ObsDictWrapper.convert_dict(self.state))
        action = policy_policy.forward(obs)#, deterministic=True)
        value = eval_policy.critic.forward(obs, action)

        action = action.detach().cpu().numpy()
        self.state, rew, done, info = self.env.step(action)
        return rew[0], done[0], value[0].item()


class SB3HerPolicyTrainer(SB3OffPolicyTrainer):
    def __init__(self, env_fn, sb3_algorithm):
        super().__init__(env_fn, sb3_algorithm.model)
        self.her_algo = sb3_algorithm
        self._base_model = sb3_algorithm.model

    def train(self, num_steps, save_dir, save_freq=1000):
        self.algorithm = self.her_algo
        values = super().train(num_steps, save_dir, save_freq)
        self.algorithm = self._base_model
        return values

    def evaluate(self, num_episodes, num_steps, num_envs=1, eval_trainer=None):
        evaluator = HERPolicyEvaluator(self.env_fn(), self.algorithm.gamma, self.algorithm, eval_trainer)
        return evaluate(evaluator, num_episodes, num_steps)
