import os
import sys
import argparse
import importlib
import warnings

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import utils.import_envs  # pytype: disable=import-error
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds


from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from utils.utils import StoreDict, get_wrapper_class, make_env

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def create_env(env_id, algo, max_frames=None, n_envs=1, seed=None, folder="trained_agents"):
    # Going through custom gym packages to let them register in the global registory
    # for env_module in args.gym_packages:
    #     importlib.import_module(env_module)

    # Sanity checks
    log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=False)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        n_envs = 1

    set_global_seeds(seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    norm_reward = False
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    log_dir = None

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

def create_test_env(env_id, n_envs=1, is_atari=False,
                    stats_path=None, seed=0,
                    log_dir='', should_render=True, hyperparams=None, env_kwargs=None):

    if hyperparams is None:
        hyperparams = {}

    if env_kwargs is None:
        env_kwargs = {}

    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    else:
        # start_method = 'spawn' for thread safe
        env = DummyVecEnv([make_env(env_id, i, seed, log_dir, wrapper_class=env_wrapper, env_kwargs=env_kwargs) for i in range(n_envs)])

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
