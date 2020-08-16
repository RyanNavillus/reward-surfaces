import gym

class SpaceWrap(gym.Env):
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = []
