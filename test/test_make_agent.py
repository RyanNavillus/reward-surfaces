from reward_surfaces.agents import make_agent

if __name__ == "__main__":
    agent = make_agent("rainbow","space_invaders","cpu",{})
    agent = make_agent("SB3_OFF","InvertedPendulumPyBulletEnv-v0","cpu",{'ALGO':'TD3'})
    agent = make_agent("SB3_ON","Pendulum-v0","cpu",{'ALGO':'PPO'})
    # agent = make_agent("SB3_HER","Pendulum-v0","cpu",{'ALGO':'TD3',"max_episode_length":100})
