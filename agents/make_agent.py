from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.her import HER
from .sb3_on_policy_train import SB3OnPolicyTrainer,SB3OffPolicyTrainer,SB3HerPolicyTrainer
from .rainbow_trainer import RainbowTrainer
import gym
from .sb3_extended_algos import ExtA2C, ExtPPO, ExtSAC
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
import pybulletgym


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


def make_vec_env_fn(env_name, simple_obs=True):
    def env_fn_v(num_envs=1):
        if "NoFrameskip" in env_name:
            # Atari environment
            def env_fn():
                env = gym.make(env_name)
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
                env = SimpleObsWrapper(env)
                return env
            return DummyVecEnv([env_fn]*num_envs)
        else:
            def env_fn():
                return gym.make(env_name)
            return DummyVecEnv([env_fn]*num_envs)
    return env_fn_v

def make_agent(agent_name, env_name, device, hyperparams):
    hyperparams = dict(**hyperparams)
    if 'rainbow' == agent_name:
        return RainbowTrainer(env_name,device=device,**hyperparams)
    elif "SB3_OFF" == agent_name:
        env_fn = make_vec_env_fn(env_name)
        env = env_fn()
        algo = SB3_OFF_ALGOS[hyperparams.pop('ALGO')]
        model = "MlpPolicy" if len(env.observation_space.shape) != 3 else "CnnPolicy"
        return SB3OffPolicyTrainer(env_fn,algo(model,env,device=device,**hyperparams))
    elif "SB3_ON" == agent_name:
        env_fn = make_vec_env_fn(env_name)
        num_envs = hyperparams.pop('num_envs', 16)
        env = env_fn(num_envs)
        algo = SB3_ON_ALGOS[hyperparams.pop('ALGO')]
        model = "MlpPolicy" if len(env.observation_space.shape) != 3 else "CnnPolicy"
        return SB3OnPolicyTrainer(env_fn,algo(model,env,device=device,**hyperparams))
    elif "SB3_HER" == agent_name:
        env_fn = make_vec_env_fn(env_name, simple_obs=False)
        algo = SB3_OFF_ALGOS[hyperparams.pop('ALGO')]
        return SB3HerPolicyTrainer(env_fn,HER("MlpPolicy",env_fn(),model_class=algo,device=device,**hyperparams))
    else:
        raise ValueError("bad agent name, must be one of ['rainbow', 'SB3_OFF', 'SB3_ON', 'SB3_HER']")

if __name__ == "__main__":
    agent = make_agent("rainbow","space_invaders","cpu",{})
    agent = make_agent("SB3_OFF","InvertedPendulumPyBulletEnv-v0","cpu",{'ALGO':'TD3'})
    agent = make_agent("SB3_ON","Pendulum-v0","cpu",{'ALGO':'PPO'})
    # agent = make_agent("SB3_HER","Pendulum-v0","cpu",{'ALGO':'TD3',"max_episode_length":100})
