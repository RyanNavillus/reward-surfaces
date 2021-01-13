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
from .sb3_on_policy_train import SB3OnPolicyTrainer,SB3OffPolicyTrainer,SB3HerPolicyTrainer
from .sb3_extended_algos import ExtA2C, ExtPPO, ExtSAC
from .evaluate_est_hesh import calculate_est_hesh_eigenvalues
from .rainbow_trainer import RainbowTrainer


def test_curvature(env_fn, trainer):
    # test trainer learning
    saved_files = trainer.train(100,"test_results",save_freq=1000)
    results = trainer.evaluate_policy_hess(1000,2000,'baselined_vals',1.0,tol=1.0)
    print(results)

def discrete_env_fn():
    return gym.make("CartPole-v1")

def continious_env_fn():
    return gym.make("Pendulum-v0")

if __name__ == "__main__":
    # print("testing SB3 SAC curvature")
    # test_curvature(continious_env_fn, SB3OffPolicyTrainer(continious_env_fn,ExtSAC("MlpPolicy",continious_env_fn(),device="cuda")))
    print("testing SB3 A2C curvature")
    test_curvature(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,ExtA2C("MlpPolicy",discrete_env_fn(),device="cpu")))
    print("testing SB3 PPO curvature")
    test_curvature(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,ExtPPO("MlpPolicy",discrete_env_fn(),device="cpu")))
