import gym
import numpy
import lycon
import numpy as np

xsize = 84
ysize = 84

def downsize(obs):
    return lycon.resize(obs, width=xsize, height=ysize, interpolation=lycon.Interpolation.AREA)

class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(84,84,3*4),dtype=np.uint8)

    def step(self, action):
        obss = []
        tot_rew = 0
        tot_done = False
        tot_info = {}
        for i in range(4):
            if not tot_done:
                obs, rew, done, info = self.env.step(action)
                tot_rew += rew
                tot_done = done
            obss.append(downsize(obs))

        res_obs = np.concatenate(obss,axis=2)
        return res_obs, tot_rew, tot_done, tot_info

    def reset(self):
        return np.concatenate([downsize(self.env.reset())]*4,axis=2)
