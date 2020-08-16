import os
import sys
import argparse
import importlib
import warnings

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds


# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer

seed_counter = 0

def create_env(env_id, algo, max_frames=None, n_envs=1, seed=None, folder="trained_agents"):
    global seed_counter
    # Going through custom gym packages to let them register in the global registory
    # for env_module in args.gym_packages:
    #     importlib.import_module(env_module)
    if seed is None:
        seed = seed_counter
        seed_counter += 1

    # Sanity checks
    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        n_envs = 1

    set_global_seeds(seed)

    is_atari = 'NoFrameskip' in env_id

    log_dir = None
    stats_path = None
    hyperparams = None

    if is_atari and max_frames is not None:
        env_kwargs = {"max_episode_steps": max_frames}
    else:
        env_kwargs = {}

    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=seed, log_dir=log_dir,
                          should_render=False,
                          hyperparams=hyperparams, env_kwargs=env_kwargs)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    return load_env, env


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None, env_kwargs=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                    env with
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id, **env_kwargs)

        env.seed(seed + rank)
        return env

    return _init

def create_test_env(env_id, n_envs=1, is_atari=False,
                    stats_path=None, seed=0,
                    log_dir='', should_render=True, hyperparams=None, env_kwargs=None):

    if hyperparams is None:
        hyperparams = {}

    if env_kwargs is None:
        env_kwargs = {}

    # Create the environment and wrap it if necessary
    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    else:
        # start_method = 'spawn' for thread safe
        env = DummyVecEnv([make_env(env_id, i, seed, log_dir, wrapper_class=None, env_kwargs=env_kwargs) for i in range(n_envs)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams['normalize']:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])

            if os.path.exists(os.path.join(stats_path, 'vecnormalize.pkl')):
                env = VecNormalize.load(os.path.join(stats_path, 'vecnormalize.pkl'), env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                # Legacy:
                env.load_running_average(stats_path)

        n_stack = hyperparams.get('frame_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env
