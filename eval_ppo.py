from ppo import PPO2
import gym
import stable_baselines
#from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines.common.policies import BasePolicy
from create_env import create_env

def calc_mean_value(rews, gamma):
    decayed_rew = 0
    tot_value = 0
    for i in range(len(rews)-1,-1,-1):
        decayed_rew += rews[i]
        tot_value += decayed_rew
        decayed_rew *= gamma
    return tot_value / len(rews)

def calc_mean_td(est_vals, rewards, gamma):
    assert len(est_vals) == len(rewards)
    ep_len = len(est_vals)
    td_errs = []
    for i in range(ep_len):
        true_val = est_vals[i+1] * gamma + rewards[i] if i < ep_len-1 else rewards[i]
        est_val = est_vals[i]
        td_err = (true_val - est_val)**2
        td_errs.append(td_err)
    return mean(td_errs)

def mean(vals):
    return sum(vals)/len(vals)

def test_env_true_reward(pred_file, eval_file, env_name, num_episodes, max_frames, single_threaded=False):
    #env =
    n_cpu_sess = 1 if single_threaded else 4
    algo_name = "ppo2"
    load_env, test_env = create_env(env_name, algo_name, n_envs=4, max_frames=max_frames)
    num_envs = test_env.num_envs
    env = test_env
    model_pred = PPO2.load(pred_file,env=load_env,n_cpu_tf_sess=n_cpu_sess)
    model_val = PPO2.load(eval_file,env=load_env,n_cpu_tf_sess=n_cpu_sess)
    gamma = model_val.gamma
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
    episode_td_err = []
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
                episode_td_err.append(calc_mean_td(ep_vals[i], ep_rews[i], gamma))#TODO make this real
                ep_rews[i] = []
                ep_vals[i] = []
                print("done!")
    #print(episode_rewards)
    return mean(episode_rewards),mean(episode_value_ests),mean(episode_values),mean(episode_td_err)

if __name__ == "__main__":
    # import gym
    # env = gym.make("Acrobot-v1")
    # import stable_baselines
    # #from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
    # from stable_baselines.common.vec_env import DummyVecEnv
    # trainer = stable_baselines.A2C.load("test.pkl",env=DummyVecEnv([lambda:env]))
    # trainer.learn(1000)
    #test_env_true_reward("trained_agents/a2c/Acrobot-v1.pkl", "Acrobot-v1", "a2c")
    model_fname = "old_experiments/trained_agents/ppo2/BeamRiderNoFrameskip-v4.pkl"
    rew, value_ests, values, td_err = test_env_true_reward(model_fname, model_fname, "BeamRiderNoFrameskip-v4", 4, 1000)
    print(rew, value_ests, values, td_err)
    #test_env("trained_agents/dqn/Acrobot-v1.pkl", "Acrobot-v1", "dqn")
