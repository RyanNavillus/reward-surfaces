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
from reward_surfaces.agents import SB3OnPolicyTrainer,SB3OffPolicyTrainer,SB3HerPolicyTrainer
from reward_surfaces.algorithms import gather_policy_hess_data, calculate_true_hesh_eigenvalues
from reward_surfaces.agents import RainbowTrainer, make_vec_env_fn
from reward_surfaces.agents import ExtA2C, ExtPPO, ExtSAC


def test_curvature(env_fn, trainer):
    # test trainer learning
    saved_files = trainer.train(100,"test_results",save_freq=1000)
    num_episodes = 10
    num_steps = 1000
    evaluator = trainer.evaluator()
    action_evalutor = trainer.action_evalutor()
    all_states, all_returns, all_actions = gather_policy_hess_data(evaluator, num_episodes, num_steps, action_evalutor.gamma, "UNUSED", gae_lambda=1.0)
    maxeig, mineig, maxeigvec, mineigvec = calculate_true_hesh_eigenvalues(action_evalutor, all_states, all_returns, all_actions, tol=0.01, device=action_evalutor.device)

    np.savez("test_results/vec1.npz", *maxeigvec)
    np.savez("test_results/vec2.npz", *mineigvec)
    print(maxeig, mineig)

discrete_env_fn = make_vec_env_fn("CartPole-v1")
continious_env_fn = make_vec_env_fn("Pendulum-v0")

if __name__ == "__main__":
    # print("testing SB3 SAC curvature")
    # test_curvature(continious_env_fn, SB3OffPolicyTrainer(continious_env_fn,ExtSAC("MlpPolicy",continious_env_fn(),device="cuda")))
    print("testing SB3 A2C curvature")
    test_curvature(continious_env_fn, SB3OnPolicyTrainer(continious_env_fn,ExtPPO("MlpPolicy",continious_env_fn(),device="cpu")))
    print("testing SB3 PPO curvature")
    test_curvature(continious_env_fn, SB3OnPolicyTrainer(continious_env_fn,ExtPPO("MlpPolicy",continious_env_fn(),device="cpu")))
