import gym
import numpy as np
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
import tempfile
from sb3_on_policy_train import SB3OnPolicyTrainer

def test_trainer(env_fn, trainer):
    # test trainer learning
    saved_files = trainer.train(100,"test_results",save_freq=10)
    assert isinstance(saved_files,list) and isinstance(saved_files[0],str)
    # test trainer IO
    trainer.load_weights(saved_files[0])
    weights = trainer.get_weights()
    assert isinstance(weights, list) and isinstance(weights[0], np.ndarray)
    trainer.set_weights(weights)
    trainer.save_weights(saved_files[0])
    trainer.load_weights(saved_files[0])
    trainer.evaluate(10)

def discrete_env_fn():
    return gym.make("CartPole-v1")

if __name__ == "__main__":
    print("testing SB3 PPO")
    test_trainer(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,PPO("MlpPolicy",discrete_env_fn(),device="cpu")))
    print("testing SB3 A2C")
    test_trainer(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,A2C("MlpPolicy",discrete_env_fn(),device="cpu")))
