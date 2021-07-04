import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import yaml
import gym
from pprint import pprint
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.callbacks import BaseCallback


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def clip_norm_(values, max_norm):
    mag = 0
    length = 0
    for g in values:
        mag += (g*g).sum()
        length += np.prod(g.shape)
    norm = mag / np.sqrt(length)
    if norm > max_norm:
        clip_v = max_norm / norm
        for g in values:
            g.data *= clip_v


class HyperparameterManager:
    """
    Manages hyperparameter parsing from yaml files

    Simplified version of the ExperimentManager from RL Zoo, with only the components
    necessary to parse hyperparameters from yaml files
    """

    def __init__(self, algo, env_id, custom_hyperparams=None, verbose=1):
        self.algo = algo
        self.env_id = env_id
        self._is_atari = self.is_atari(env_id)
        self.custom_hyperparams = custom_hyperparams
        self._hyperparams = {}
        self.verbose = verbose
        self.normalize = False

    def get_hyperparams(self):
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams = self._preprocess_hyperparams(hyperparams)
        return hyperparams

    @staticmethod
    def is_atari(env_id: str) -> bool:
        return "AtariEnv" in gym.envs.registry.env_specs[env_id].entry_point

    # @staticmethod
    # def is_bullet(env_id: str) -> bool:
    #     return "pybullet_envs" in gym.envs.registry.env_specs[env_id].entry_point

    # @staticmethod
    # def is_robotics_env(env_id: str) -> bool:
    #    entry_point = gym.envs.registry.env_specs[env_id].entry_point
    #    return "gym.envs.robotics" in entry_point or "panda_gym.envs" in entry_point

    @staticmethod
    def linear_schedule(initial_value):
        """
        Linear learning rate schedule.
        :param initial_value: (float or str)
        :return: (function)
        """
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0
            :param progress_remaining: (float)
            :return: (float)
            """
            return progress_remaining * initial_value

        return func

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load hyperparameters from yaml file
        with open(f"./reward_surfaces/hyperparams/{self.algo}.yml", "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            elif self._is_atari:
                hyperparams = hyperparams_dict["atari"]
            else:
                raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_id}")

        if self.custom_hyperparams is not None:
            # Overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)
        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        if self.verbose > 0:
            print("Default hyperparameters for environment (ones being tuned will be overridden):")
            pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        # Create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def _preprocess_normalization(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            # Special case, instead of both normalizing
            # both observation and reward, we can normalize one of the two.
            # in that case `hyperparams["normalize"]` is a string
            # that can be evaluated as python,
            # ex: "dict(norm_obs=False, norm_reward=True)"
            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            # Use the same discount factor as for the algorithm
            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams

    def _preprocess_hyperparams(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:

        # Convert schedule strings to objects
        hyperparams = self._preprocess_schedules(hyperparams)

        # Pre-process train_freq
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        # Pre-process normalize config
        hyperparams = self._preprocess_normalization(hyperparams)

        # Pre-process policy/buffer keyword arguments
        # Convert to python object if needed
        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        if "frame_stack" in hyperparams.keys():
            del hyperparams["frame_stack"]

        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        if "callback" in hyperparams.keys():
            del hyperparams["callback"]

        if "policy" in hyperparams.keys():
            del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        return hyperparams
