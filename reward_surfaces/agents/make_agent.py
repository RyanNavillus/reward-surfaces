from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.her import HerReplayBuffer
import gym
from .SB3 import SB3OnPolicyTrainer, SB3OffPolicyTrainer, SB3HerPolicyTrainer
from .rainbow.rainbow_trainer import RainbowTrainer
from .SB3.sb3_extended_algos import ExtA2C, ExtPPO, ExtSAC
from .experiment_manager import ExperimentManager

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


def make_vec_env_fn(env_name, manager, simple_obs=True, is_eval=False):
    def env_fn_v(num_envs=1):
        def env_fn():
            return manager.create_envs(num_envs, eval_env=is_eval)
        return env_fn()
    return env_fn_v


def make_agent(agent_name, env_name, save_dir, hyperparams, device="cuda", pretraining=None, eval_freq=50, n_eval_episodes=50):
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
        algo_name = hyperparams.pop('ALGO')
        manager = ExperimentManager(algo_name.lower(), env_name, save_dir, hyperparams=hyperparams,
                                    pretraining=pretraining, verbose=1, device=device)
        model, _, steps = manager.setup_experiment()
        if pretraining:
            best_model = manager.get_best_model()
            pretraining["best_model"] = best_model
            model.num_timesteps = pretraining["trained_steps"]
            steps -= pretraining["trained_steps"]
        env_fn = make_vec_env_fn(env_name, manager)
        eval_env_fn = make_vec_env_fn(env_name, manager, is_eval=True)
        return SB3OnPolicyTrainer(env_fn, model, manager.n_envs, env_name, eval_env_fn=eval_env_fn,
                                  pretraining=pretraining, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes), steps
    elif "SB3_HER" == agent_name:
        env_fn = make_vec_env_fn(env_name, simple_obs=False)
        algo = SB3_OFF_ALGOS[hyperparams.pop('ALGO')]
        return SB3HerPolicyTrainer(
            env_fn,
            HerReplayBuffer("MlpPolicy", env_fn(), model_class=algo, device=device, **hyperparams)
        )
    else:
        raise ValueError("bad agent name, must be one of ['rainbow', 'SB3_OFF', 'SB3_ON', 'SB3_HER']")
