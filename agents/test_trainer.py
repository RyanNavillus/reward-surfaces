import numpy as np
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.her import HER
import tempfile
import gym
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from .sb3_on_policy_train import SB3OnPolicyTrainer,SB3OffPolicyTrainer,SB3HerPolicyTrainer
from .rainbow_trainer import RainbowTrainer

def test_trainer(learn_steps, save_freq, trainer):
    # test trainer learning
    saved_files = trainer.train(learn_steps,"test_results",save_freq=save_freq)
    assert isinstance(saved_files,list) and isinstance(saved_files[0],str)
    # test trainer IO
    trainer.load_weights(saved_files[0])
    weights = trainer.get_weights()
    assert isinstance(weights, list) and isinstance(weights[0], np.ndarray)
    trainer.set_weights(weights)
    trainer.save_weights(saved_files[0])
    trainer.load_weights(saved_files[0])
    trainer.evaluate(10, 1000)

def discrete_env_fn():
    return gym.make("CartPole-v1")

def continious_env_fn():
    return gym.make("Pendulum-v0")

def robo_env_fn():
    return BitFlippingEnv(continuous=True)

if __name__ == "__main__":
    print("testing Rainbow")
    test_trainer(1500,1000,RainbowTrainer("space_invaders",learning_starts=1000))
    print("testing SB3 HER")
    test_trainer(100,100,SB3HerPolicyTrainer(robo_env_fn,HER("MlpPolicy",robo_env_fn(),model_class=TD3,device="cpu",max_episode_length=100)))
    print("testing SB3 TD3")
    test_trainer(100,100,SB3OffPolicyTrainer(continious_env_fn,TD3("MlpPolicy",continious_env_fn(),device="cpu")))
    print("testing SB3 SAC")
    test_trainer(100,100,SB3OffPolicyTrainer(continious_env_fn,SAC("MlpPolicy",continious_env_fn(),device="cpu")))
    print("testing SB3 DDPG")
    test_trainer(100,100,SB3OffPolicyTrainer(continious_env_fn,DDPG("MlpPolicy",continious_env_fn(),device="cpu")))
    print("testing SB3 PPO")
    test_trainer(100,100,SB3OnPolicyTrainer(discrete_env_fn,PPO("MlpPolicy",discrete_env_fn(),device="cpu",n_steps=10)))
    print("testing SB3 PPO with continuous env")
    test_trainer(100,100,SB3OnPolicyTrainer(continious_env_fn,PPO("MlpPolicy",continious_env_fn(),device="cpu",n_steps=10)))
    print("testing SB3 A2C")
    test_trainer(100,100,SB3OnPolicyTrainer(discrete_env_fn,A2C("MlpPolicy",discrete_env_fn(),device="cpu")))
