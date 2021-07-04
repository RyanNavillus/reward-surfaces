from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.utils import constant_fn
from .SB3 import SB3OnPolicyTrainer, SB3OffPolicyTrainer, SB3HerPolicyTrainer
from .rainbow.rainbow_trainer import RainbowTrainer
import gym
from .SB3.sb3_extended_algos import ExtA2C, ExtPPO, ExtSAC
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from .utils import HyperparameterManager


SB3_ON_ALGOS = {
    "A2C": ExtA2C,
    "PPO": ExtPPO,
}
SB3_OFF_ALGOS = {
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": ExtSAC,
}


class SimpleObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['observation']

    def reset(self):
        return super().reset()['observation']

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs['observation'], rew, done, info


def make_vec_env_fn(env_name, simple_obs=True, is_eval_env=False):
    def env_fn_v(num_envs=1):
        if "NoFrameskip" in env_name:
            # Atari environment
            def env_fn():
                env = gym.make(env_name)
                if is_eval_env:
                    env = Monitor(env)
                env = AtariWrapper(env)
                return env
            env = DummyVecEnv([env_fn]*num_envs)
            env = VecFrameStack(env, 4)
            env = VecTransposeImage(env)
            env = VecNormalize(env)
            return env
        elif ("Fetch" in env_name or "Hand" in env_name) and simple_obs:
            # Gym robotics environment
            def env_fn():
                env = gym.make(env_name)
                if is_eval_env:
                    env = Monitor(env)
                env = SimpleObsWrapper(env)
                return env
            return DummyVecEnv([env_fn]*num_envs)
        else:
            def env_fn():
                env = gym.make(env_name)
                if is_eval_env:
                    env = Monitor(env)
                return env
            return DummyVecEnv([env_fn]*num_envs)
    return env_fn_v


def make_agent(agent_name, env_name, device, hyperparams):
    hyperparams = dict(**hyperparams)
    if 'rainbow' == agent_name:
        return RainbowTrainer(env_name, device=device, **hyperparams)
    elif "SB3_OFF" == agent_name:
        env_fn = make_vec_env_fn(env_name)
        env = env_fn()
        algo = SB3_OFF_ALGOS[hyperparams.pop('ALGO')]
        model = "MlpPolicy" if len(env.observation_space.shape) != 3 else "CnnPolicy"
        alg = algo(model, env, device=device, **hyperparams)
        return SB3OffPolicyTrainer(env_fn, alg)
    elif "SB3_ON" == agent_name:
        env_fn = make_vec_env_fn(env_name)
        eval_env_fn = make_vec_env_fn(env_name, is_eval_env=True)
        algo_name = hyperparams.pop('ALGO')
        algo = SB3_ON_ALGOS[algo_name]
        manager = HyperparameterManager(algo_name.lower(), env_name, custom_hyperparams=hyperparams)
        hyperparams = manager.get_hyperparams()
        num_envs = hyperparams.pop('n_envs', 16)
        env = env_fn(num_envs)
        eval_env = eval_env_fn(1)
        model = "MlpPolicy" if len(env.observation_space.shape) != 3 else "CnnPolicy"
        print(hyperparams)
        steps = hyperparams.pop('n_timesteps')
        return SB3OnPolicyTrainer(env_fn, algo(model, env, device=device, **hyperparams), eval_env_fn=eval_env), steps
    elif "SB3_HER" == agent_name:
        env_fn = make_vec_env_fn(env_name, simple_obs=False)
        algo = SB3_OFF_ALGOS[hyperparams.pop('ALGO')]
        return SB3HerPolicyTrainer(
            env_fn,
            HerReplayBuffer("MlpPolicy", env_fn(), model_class=algo, device=device, **hyperparams)
        )
    else:
        raise ValueError("bad agent name, must be one of ['rainbow', 'SB3_OFF', 'SB3_ON', 'SB3_HER']")
