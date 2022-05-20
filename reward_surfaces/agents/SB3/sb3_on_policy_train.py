from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, Union
import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from .extract_params import ParamLoader
import tempfile
import torch
from collections import OrderedDict
from stable_baselines3.common import base_class
import os

# Circular import here? Fix this
#from reward_surfaces.algorithms.evaluate_est_hesh import calculate_est_hesh_eigenvalues


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class CheckpointParamCallback(CheckpointCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", n_envs: int = 1,
                 init_timesteps: int = 0, verbose: int = 0):
        super(CheckpointParamCallback, self).__init__(save_freq, save_path, name_prefix=name_prefix, verbose=verbose)
        self.save_folders = []
        self.n_envs = n_envs

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        super(CheckpointParamCallback, self).init_callback(model)
        self.old_params = [param.clone() for param in self.model.policy.parameters()]

    def _on_step(self) -> bool:
        if self.n_calls > 0 and (self.n_calls - 1) % self.save_freq == 0:
            self.old_params = [param.clone() for param in self.model.policy.parameters()]
        if self.n_calls % self.save_freq == 0:
            print(f"saved checkpoint {self.num_timesteps:07}")
            path = os.path.join(self.save_path, f"{self.num_timesteps:07}")
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, "checkpoint")
            self.save_folders.append(save_path)
            self.model.save(save_path)
            model_parameters = self.model.policy.state_dict()
            grads = OrderedDict([(name, param.grad) for name, param in model_parameters.items()])
            torch.save(model_parameters, os.path.join(path, "parameters.th"))
            torch.save(grads, os.path.join(path, "grads.th"))

            if self.old_params is not None:
                delta = OrderedDict([
                    (name, param - old_param) for old_param, (name, param) in zip(self.old_params, model_parameters.items())
                ])
                torch.save(delta, os.path.join(path, "prev_step.th"))

            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        return True


class EvalParamCallback(EvalCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 25,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        init_timesteps: int = 0,
        baselines_agent: "base_class.BaseAlgorithm" = None,
        verbose: int = 0,
        warn: bool = True,
    ):
        super(EvalParamCallback, self).__init__(eval_env, callback_on_new_best, n_eval_episodes, eval_freq, log_path,
                                                best_model_save_path, deterministic, render, verbose, warn)
        self.old_params = None
        self.save_folders = []
        self.log_path = log_path
        self.baseline_agent = baselines_agent

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        super(EvalParamCallback, self).init_callback(model)
        self.num_timesteps = self.model.num_timesteps
        if self.baseline_agent:
            print("eval baseline")
            self.old_params = [param.clone() for param in self.model.policy.parameters()]
            episode_rewards, episode_lengths = evaluate_policy(
                self.baseline_agent,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

    def _on_step(self) -> bool:
        if self.n_calls > 0 and (self.n_calls - 1) % self.eval_freq == 0:
            self.old_params = [param.clone() for param in self.model.policy.parameters()]
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward >= self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "checkpoint"))
                self.best_mean_reward = mean_reward
                os.makedirs(self.log_path, exist_ok=True)
                save_path = os.path.join(self.log_path, "checkpoint")
                self.save_folders.append(save_path)
                model_parameters = self.model.policy.state_dict()
                grads = OrderedDict([(name, param.grad) for name, param in model_parameters.items()])
                torch.save(model_parameters, os.path.join(self.log_path, "parameters.th"))
                torch.save(grads, os.path.join(self.log_path, "grads.th"))

                if self.old_params is not None:
                    delta = OrderedDict([
                        (name, param - old_param) for old_param, (name, param) in zip(self.old_params, model_parameters.items())
                    ])
                    torch.save(delta, os.path.join(self.log_path, "prev_step.th"))

                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


class OnPolicyEvaluator:
    def __init__(self, vec_env, gamma, algo, eval_trainer):
        env = vec_env
        env.seed(seed=1) # Testing
        self.state = env.reset()
        self.env = env
        self.gamma = gamma
        self.algo = algo
        self.eval_trainer = eval_trainer

    def _next_state(self):
        return self._next_state_act()[:3]

    def _next_state_act(self):
        policy_policy = self.algo.policy

        old_state = self.state
        obs = torch.as_tensor(self.state, device=self.algo.device)
        action, policy_val, policy_log_prob = policy_policy.forward(obs, deterministic=True)
        if self.eval_trainer is None:
            value = policy_val.item()
        else:
            eval_policy = self.eval_trainer.algorithm.policy
            value, eval_log_prob, eval_entropy = eval_policy.evaluate_actions(obs, action)
            value = value.item()

        action = action.detach().cpu().numpy()
        if isinstance(self.algo.action_space, gym.spaces.Box):
            action = np.clip(action, self.algo.action_space.low, self.algo.action_space.high)
        self.state, rew, done, info = self.env.step(action)
        #time.sleep(0.01)

        # Access the rewards created by the SB3 Monitor wrapper
        original_rew = 0
        if hasattr(self.env, "n_stack"):
            original_rew = sum(self.env.envs[0].rewards[-self.env.n_stack:])
        else:
            if len(self.env.envs[0].rewards) > 0:
                original_rew = self.env.envs[0].rewards[-1]
        return rew[0], original_rew, done[0], value, old_state, action, info[0]


class SB3OnPolicyTrainer:
    def __init__(self, env_fn, sb3_algorithm, n_envs, env_id, deterministic_eval=False, eval_env_fn=None,
                 eval_freq=100000, n_eval_episodes=25, n_eval_envs=5, pretraining=None):
        self.env_fn = env_fn
        self.eval_env_fn = eval_env_fn
        self.algorithm = sb3_algorithm
        self.device = sb3_algorithm.device
        self.pretraining = pretraining

        # Callbacks
        self.callbacks = []
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.n_envs = n_envs
        self.env_id = env_id
        self.deterministic_eval = deterministic_eval

    def create_callbacks(self, save_dir, save_freq=100000, prefix='rl_model'):
        checkpoint_callback = None
        init_timesteps = 0
        if self.pretraining:
            init_timesteps = self.pretraining["trained_steps"]

        if save_freq > 0:
            # Account for the number of parallel environments
            save_freq = max(save_freq // self.n_envs, 1)
            checkpoint_callback = CheckpointParamCallback(save_freq=save_freq,
                                                          save_path=save_dir,
                                                          name_prefix=prefix,
                                                          n_envs=self.n_envs,
                                                          init_timesteps=init_timesteps,
                                                          verbose=0)
            self.callbacks.append(checkpoint_callback)

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0:
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)
            baseline_model = None
            if self.pretraining:
                baseline_model = self.pretraining["best_model"]

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=save_dir)
            eval_env = self.eval_env_fn()
            eval_callback = EvalParamCallback(
                eval_env,
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=save_dir + '/best/',
                n_eval_episodes=self.n_eval_episodes,
                log_path=save_dir + '/best/',
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
                baselines_agent=baseline_model,
                init_timesteps=init_timesteps,
                verbose=1
            )

            self.callbacks.append(eval_callback)
        return checkpoint_callback

    def train(self, num_steps, save_dir, save_freq=10000):
        save_prefix = f'sb3_{type(self.algorithm).__name__}'
        checkpoint_callback = self.create_callbacks(save_dir, save_freq=save_freq, prefix=save_prefix)

        self.algorithm.env.reset()
        self.algorithm.learn(num_steps, callback=self.callbacks, reset_num_timesteps=False)
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
        self.algorithm = type(self.algorithm).load(load_file, self.eval_env_fn(), self.device)

    def save_weights(self, save_file):
        self.algorithm.save(save_file)

    def calculate_eigenvalues(self, num_steps, tol=1e-2):
        maxeig, mineig = calculate_est_hesh_eigenvalues(self.algorithm, num_steps, tol)
        buffer_stats = self.algorithm.buffer_stats
        buffer_stats['maxeig'] = maxeig
        buffer_stats['mineig'] = mineig
        buffer_stats['ratio'] = -min(0, mineig)/maxeig
        return buffer_stats

    def evaluator(self, eval_trainer=None):
        return OnPolicyEvaluator(self.eval_env_fn(), self.algorithm.gamma, self.algorithm, eval_trainer)

    def action_evalutor(self):
        return self.algorithm


class OffPolicyEvaluator(OnPolicyEvaluator):
    def _next_state_act(self):
        policy_policy = self.algo.policy
        eval_policy = policy_policy if self.eval_trainer is None else self.eval_trainer.algorithm.policy

        old_state = self.state
        obs = torch.as_tensor(self.state, device=self.algo.device)
        action = policy_policy.forward(obs)#, deterministic=True)
        value = eval_policy.critic.forward(obs, action)

        action = action.detach().cpu().numpy()
        self.state, rew, done, info = self.env.step(action)
        return rew[0], done[0], value[0].item(), old_state, action


class SB3OffPolicyTrainer(SB3OnPolicyTrainer):
    def evaluator(self, eval_trainer=None):
        return OffPolicyEvaluator(self.eval_env_fn(), self.algorithm.gamma, self.algorithm, eval_trainer)


class HERPolicyEvaluator(OnPolicyEvaluator):
    def _next_state_act(self):
        policy_policy = self.algo.policy
        eval_policy = policy_policy if self.eval_trainer is None else self.eval_trainer.algorithm.policy

        old_state = self.state
        obs = torch.as_tensor(self.state, device=self.algo.device)
        action = policy_policy.forward(obs)#, deterministic=True)
        value = eval_policy.critic.forward(obs, action)

        action = action.detach().cpu().numpy()
        self.state, rew, done, info = self.env.step(action)
        return rew[0], done[0], value[0].item(), old_state, action


class SB3HerPolicyTrainer(SB3OffPolicyTrainer):
    def __init__(self, env_fn, sb3_algorithm, eval_env_fn=None):
        super().__init__(env_fn, sb3_algorithm.model, eval_env_fn=eval_env_fn)
        self.her_algo = sb3_algorithm
        self._base_model = sb3_algorithm.model

    def train(self, num_steps, save_dir, save_freq=1000):
        self.algorithm = self.her_algo
        values = super().train(num_steps, save_dir, save_freq)
        self.algorithm = self._base_model
        return values

    def evaluator(self, eval_trainer=None):
        return HERPolicyEvaluator(self.env_fn(), self.algorithm.gamma, self.algorithm, eval_trainer)
