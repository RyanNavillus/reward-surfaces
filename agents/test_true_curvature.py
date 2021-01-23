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
from .make_agent import make_vec_env_fn


def test_curvature(env_fn, trainer):
    # test trainer learning
    saved_files = trainer.train(100,"test_results",save_freq=1000)
    results = trainer.evaluate_policy_hess(10000,3, "NOT_USED", gae_lambda=1., tol=1e-2)
    print(results)

discrete_env_fn = make_vec_env_fn("CartPole-v1")
continious_env_fn = make_vec_env_fn("Pendulum-v0")

if __name__ == "__main__":
    # print("testing SB3 SAC curvature")
    # test_curvature(continious_env_fn, SB3OffPolicyTrainer(continious_env_fn,ExtSAC("MlpPolicy",continious_env_fn(),device="cuda")))
    print("testing SB3 A2C curvature")
    test_curvature(continious_env_fn, SB3OnPolicyTrainer(continious_env_fn,ExtPPO("MlpPolicy",continious_env_fn(),device="cpu")))
    print("testing SB3 PPO curvature")
    test_curvature(continious_env_fn, SB3OnPolicyTrainer(continious_env_fn,ExtPPO("MlpPolicy",continious_env_fn(),device="cpu")))
