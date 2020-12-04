import gym
import numpy as np
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.her import HER
import tempfile
import gym  # open ai gym
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from sb3_on_policy_train import SB3OnPolicyTrainer,SB3OffPolicyTrainer,SB3HerPolicyTrainer
from sb3_extended_algos import ExtA2C

def test_curvature(env_fn, trainer):
    # test trainer learning
    saved_files = trainer.train(100,"test_results",save_freq=1000)
    maxeig, mineig = trainer.calculate_eigenvalues(20000,1.e-5)
    print(maxeig, mineig)

def discrete_env_fn():
    return gym.make("CartPole-v1")

def continious_env_fn():
    return gym.make("Pendulum-v0")

if __name__ == "__main__":
    print("testing SB3 A2C")
    test_curvature(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,ExtA2C("MlpPolicy",discrete_env_fn(),device="cuda")))
