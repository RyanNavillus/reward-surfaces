import numpy as np

class SingleVecEnv:
    def __init__(self, gym_env_fns):
        assert len(gym_env_fns) == 1
        self.gym_env = gym_env_fns[0]()
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.num_envs = 1

    def reset(self):
        return np.expand_dims(self.gym_env.reset(),0)

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def step(self, actions):
        observations, reward, done, info = self.gym_env.step(actions[0])
        if done:
            observations = self.gym_env.reset()
        observations =  np.expand_dims(observations,0)
        rewards = np.array([reward], dtype=np.float32)
        dones = np.array([done], dtype=np.bool)
        infos = [info]
        return observations, rewards, dones,infos
