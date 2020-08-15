import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import stable_baselines
#from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from utils import ALGOS
import numpy as np
from stable_baselines.common.policies import BasePolicy
from create_env import create_env

def test_env_seperate_eval(load_file_policy,load_file_evaluator,env_name,algo_name):
    num_envs = 10
    env = DummyVecEnv([lambda:gym.make(env_name)]*num_envs)
    policy_model = ALGOS[algo_name].load(load_file_policy,env=DummyVecEnv([lambda:env]))
    eval_model = ALGOS[algo_name].load(load_file_evaluator,env=DummyVecEnv([lambda:env]))
    for key,value in eval_model.__dict__.items():
        if isinstance(value, BasePolicy):
            eval_policy = value
            break

    #policy = trainer.policy
    obs = env.reset()
    total_value
    episode_rewards = []
    state = None
    while len(episode_rewards) < 5:
        action, state = policy_model.predict(obs, state=state)#, deterministic=deterministic)

        _, values = eval_policy.step(obs)[:2]

        obs, rew, done, info = env.step(action)
    print(episode_rewards)

def calc_mean_value(rews, gamma):
    decayed_rew = 0
    tot_value = 0
    for i in range(len(rews)-1,-1,-1):
        decayed_rew += rews[i]
        tot_value += decayed_rew
        decayed_rew *= gamma
    return tot_value / len(rews)

def mean(vals):
    return sum(vals)/len(vals)

def test_env_true_reward(load_file, env_name, algo_name, num_episodes, max_frames):
    #env =
    load_env, test_env = create_env(env_name, algo_name, n_envs=1, max_frames=max_frames)
    num_envs = test_env.num_envs
    env = test_env
    model_pred = ALGOS[algo_name].load(load_file,env=load_env,n_cpu_tf_sess=1)
    model_val = ALGOS[algo_name].load(load_file,env=load_env,n_cpu_tf_sess=1)
    for key,value in model_val.__dict__.items():
        if isinstance(value, BasePolicy):
            policy = value
            break

    #policy = trainer.policy
    obs = env.reset()
    #tot_rew = np.zeros(num_envs)
    ep_rews = [[] for i in range(num_envs)]
    ep_vals = [[] for i in range(num_envs)]
    episode_rewards = []
    episode_value_ests = []
    episode_values = []
    state = None
    while len(episode_rewards) < num_episodes:
        action, state = model_pred.predict(obs, state=state)#, deterministic=True)
        # if hasattr(policy, "values"):
        #     value = policy.value(obs)
        try:
            _, cur_vals = policy.step(obs)[:2]
        except Exception as e:
            print(e)
            raise e
            cur_vals = [0]*num_envs

        # #values = policy.value(obs)
        # print(values)
        # #actions, values, states, neglogp = trainer.act(obs)
        obs, rew, done, info = env.step(action)
        #print(rew)
        #tot_rew += rew
        for i in range(num_envs):
            ep_vals[i].append(cur_vals[i])
            ep_rews[i].append(rew[i])

        for i, d in enumerate(done):
            if d:
                episode_rewards.append(sum(ep_rews[i]))
                episode_value_ests.append(mean(ep_vals[i]))
                episode_values.append(calc_mean_value(ep_rews[i], model_pred.gamma))
                ep_rews[i] = []
                ep_vals[i] = []
                print("done!")
    #print(episode_rewards)
    return mean(episode_rewards),mean(episode_value_ests),mean(episode_values)

if __name__ == "__main__":
    # import gym
    # env = gym.make("Acrobot-v1")
    # import stable_baselines
    # #from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
    # from stable_baselines.common.vec_env import DummyVecEnv
    # trainer = stable_baselines.A2C.load("test.pkl",env=DummyVecEnv([lambda:env]))
    # trainer.learn(1000)
    #test_env_true_reward("trained_agents/a2c/Acrobot-v1.pkl", "Acrobot-v1", "a2c")
    test_env_true_reward("trained_agents/dqn/BeamRiderNoFrameskip-v4.pkl", "BeamRiderNoFrameskip-v4", "dqn", 5)
    #test_env("trained_agents/dqn/Acrobot-v1.pkl", "Acrobot-v1", "dqn")
