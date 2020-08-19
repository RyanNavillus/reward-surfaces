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
from stable_baselines.common.atari_wrappers import wrap_deepmind, make_atari, NoopResetEnv, MaxAndSkipEnv
from stable_baselines.common import set_global_seeds
from vector import MakeCPUAsyncConstructor
from allinoneatari import AtariWrapper


# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer

seed_counter = 0

def make_atari(env_id, max_episode_steps):
    """
    Create a wrapped atari Environment

    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def create_env(env_id, algo, max_frames=None, n_envs=1, seed=None, folder="trained_agents", multiproc=False):
    global seed_counter
    # Going through custom gym packages to let them register in the global registory
    # for env_module in args.gym_packages:
    #     importlib.import_module(env_module)
    if seed is None:
        seed = seed_counter
        seed_counter += 1921

    # Sanity checks
    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        n_envs = 1

    #set_global_seeds(seed)

    is_atari = 'NoFrameskip' in env_id

    log_dir = None
    stats_path = None
    hyperparams = None

    if is_atari and max_frames is not None:
        env_kwargs = {"max_episode_steps": max_frames}
    else:
        env_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env._max_episode_steps = max_frames
            # default env creation
            env = AtariWrapper(env)
            return env
            # env = make_atari(env_id, **env_kwargs)
            # env.seed(seed + rank)
            # return wrap_deepmind(env)
        return _thunk

    env_fns = [make_env(i) for i in range(n_envs)]
    if not multiproc:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = MakeCPUAsyncConstructor(24)(env_fns)
        vec_env.metadata = {}
    #vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env
