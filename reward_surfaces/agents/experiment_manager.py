import argparse
import os
import glob
import importlib
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gym
import numpy as np
import yaml
import pybulletgym

# For using HER with GoalEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import (DummyVecEnv,
                                              SubprocVecEnv,
                                              VecEnv,
                                              VecFrameStack,
                                              VecNormalize,
                                              VecTransposeImage)

# Register custom envs
# import utils.import_envs  # noqa: F401 pytype: disable=import-error
from torch import nn as nn  # noqa: F401
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from .SB3.sb3_extended_algos import ExtA2C, ExtPPO, ExtSAC
ALGOS = {
    "a2c": ExtA2C,
    "ppo": ExtPPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": ExtSAC,
}


def get_wrapper_class(hyperparams: Dict[str, Any]) -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
    for multiple, specify a list:
    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper
    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    return None


def get_latest_run_id(log_path: str, env_id: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :param log_path: path to log folder
    :param env_id:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_id + "_[0-9]*")):
        file_name = os.path.basename(path)
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


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


class ExperimentManager:
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.
    Please take a look at `train.py` to have the details for each argument.
    """

    def __init__(
        self,
        algo: str,
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        trained_agent: str = "",
        truncate_last_trajectory: bool = False,
        uuid_str: str = "",
        seed: int = None,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
    ):
        super(ExperimentManager, self).__init__()
        self.algo = algo
        self.env_id = env_id
        # Custom params
        self.custom_hyperparams = hyperparams
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.n_timesteps = n_timesteps
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.seed = seed

        self.vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]

        self.vec_env_kwargs = {}
        # self.vec_env_kwargs = {} if vec_env_type == "dummy" else {"start_method": "fork"}

        self.n_envs = 1  # it will be updated when reading hyperparams
        self.n_actions = None  # For DDPG/TD3 action noise objects
        self._hyperparams = {}

        self.trained_agent = trained_agent
        self.continue_training = trained_agent.endswith(".zip") and os.path.isfile(trained_agent)
        self.truncate_last_trajectory = truncate_last_trajectory

        self._is_atari = self.is_atari(env_id)
        # maximum number of trials for finding the best hyperparams
        self.deterministic_eval = not self.is_atari(self.env_id)

        # Logging
        self.log_folder = log_folder
        self.tensorboard_log = None if tensorboard_log == "" else os.path.join(tensorboard_log, env_id)
        self.verbose = verbose
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer

        self.log_path = f"{log_folder}/{self.algo}/"
        self.save_path = os.path.join(
            self.log_path, f"{self.env_id}_{get_latest_run_id(self.log_path, self.env_id) + 1}{uuid_str}"
        )
        self.params_path = f"{self.save_path}/{self.env_id}"

    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, action noise objects)
        create the environment and possibly the model.
        :return: the initialized RL model
        """
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper = self._preprocess_hyperparams(hyperparams)

        self.create_log_folder()

        # Create env to have access to action space for action noise
        env = self.create_envs(self.n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)
        print(self._hyperparams)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=0,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, env, self.n_timesteps

    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be use later
        to reproduce an experiment.
        :param saved_hyperparams:
        """
        # Save hyperparams
        with open(os.path.join(self.params_path, "config.yml"), "w") as config_file:
            yaml.dump(saved_hyperparams, config_file)

        print(f"Log path: {self.save_path}")

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load hyperparameters from yaml file
        with open(f"./reward_surfaces/hyperparams/{self.algo}.yml", "r") as hyper_file:
            hyperparams_dict = yaml.safe_load(hyper_file)
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
    ) -> Tuple[Dict[str, Any], Optional[Callable]]:
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        # Convert schedule strings to objects
        hyperparams = self._preprocess_schedules(hyperparams)

        # Pre-process train_freq
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        # Should we overwrite the number of timesteps?
        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        # Pre-process normalize config
        hyperparams = self._preprocess_normalization(hyperparams)

        # Pre-process policy/buffer keyword arguments
        # Convert to python object if needed
        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        # Delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        if "callback" in hyperparams.keys():
            del hyperparams["callback"]

        return hyperparams, env_wrapper

    def _preprocess_action_noise(
        self, hyperparams: Dict[str, Any], saved_hyperparams: Dict[str, Any], env: VecEnv
    ) -> Dict[str, Any]:
        # Parse noise string
        # Note: only off-policy algorithms are supported
        if hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            # Save for later (hyperparameter optimization)
            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams

    def create_log_folder(self):
        os.makedirs(self.params_path, exist_ok=True)

    @staticmethod
    def is_atari(env_id: str) -> bool:
        return "AtariEnv" in gym.envs.registry.env_specs[env_id].entry_point

    @staticmethod
    def is_bullet(env_id: str) -> bool:
        return "pybullet_envs" in gym.envs.registry.env_specs[env_id].entry_point

    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "gym.envs.robotics" in entry_point or "panda_gym.envs" in entry_point

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.
        :param env:
        :param eval_env:
        :return:
        """
        # Pretrained model, load normalization
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                print("Eval normalize")
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env

    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.
        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        monitor_kwargs = {}
        # Special case for GoalEnvs: log success rate too
        if "Neck" in self.env_id or self.is_robotics_env(self.env_id) or "parking-v0" in self.env_id:
            monitor_kwargs = dict(info_keywords=("is_success",))

        # On most env, SubprocVecEnv does not help and is quite memory hungry
        # therefore we use DummyVecEnv by default
        env = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=monitor_kwargs,
        )

        # Wrap the env into a VecNormalize wrapper if needed
        # and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        # Optional Frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        # Wrap if needed to re-order channels
        # (switch from channel last to channel first convention)
        if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
            if self.verbose > 0:
                print("Wrapping into a VecTransposeImage")
            env = VecTransposeImage(env)

        return env

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            # `truncate_last_traj` will be taken into account only if we use HER replay buffer
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model
