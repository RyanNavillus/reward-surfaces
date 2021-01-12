import time
import torch
import numpy as np

class Evaluator:
    def _next_state(self):
        '''
        return rew, done, value
        '''

def evaluate(evaluator, num_episodes, num_steps):
    gamma = evaluator.gamma

    all_datas = []
    tot_steps = 0
    tot_eps = 0
    tot_rew = 0
    start_t = time.time()
    done = False
    while not done or (tot_eps < num_episodes and tot_steps < num_steps):
        with torch.no_grad():
            rew, done, value = evaluator._next_state()#, deterministic=True)
        all_datas.append((rew, done, value))
        tot_steps += 1
        tot_rew += rew
        if done:
            tot_eps += 1
            end_t = time.time()
            print("done!", (end_t - start_t)/tot_steps, (end_t - start_t)/tot_eps,tot_rew/tot_eps)
    return calculate_stats(all_datas,gamma)


def calculate_stats(all_datas, gamma):
    ep_rews = []
    ep_vals = []
    episode_rewards = []
    episode_value_ests = []
    episode_values = []
    episode_td_err = []
    episode_lens = []
    tot_steps = len(all_datas)
    for rew, done, value in all_datas:
        ep_vals.append(value)
        ep_rews.append(rew)
        if done:
            episode_rewards.append(sum(ep_rews))
            episode_value_ests.append(sum(ep_vals))
            episode_values.append(calc_sum_value(ep_rews, gamma))
            episode_td_err.append(calc_sum_td(ep_vals, ep_rews, gamma))#TODO make this real
            episode_lens.append(len(ep_rews))

            ep_rews = []
            ep_vals = []

    episode_rewards = np.array(episode_rewards,dtype=np.float64)
    episode_value_ests = np.array(episode_value_ests)
    episode_values = np.array(episode_values)
    episode_td_err = np.array(episode_td_err)
    episode_lens = np.array(episode_lens,dtype=np.float64)
    return {
        "episode_rewards": float(np.mean(episode_rewards)),
        "episode_value_ests": float(np.mean(episode_value_ests/episode_lens)),
        "episode_values": float(np.mean(episode_values/episode_lens)),
        "episode_td_err": float(np.mean(episode_td_err/episode_lens)),
        "episode_avg_len": float(np.mean(episode_lens)),
        "step_rewards": float(np.sum(episode_rewards)/tot_steps),
        "step_value_ests": float(np.sum(episode_value_ests)/tot_steps),
        "step_values": float(np.sum(episode_values)/tot_steps),
        "step_td_err": float(np.sum(episode_td_err)/tot_steps),
    }


def mean(vals):
    return sum(vals)/len(vals)


def calc_sum_value(rews, gamma):
    decayed_rew = 0
    tot_value = 0
    for i in range(len(rews)-1,-1,-1):
        decayed_rew += rews[i]
        tot_value += decayed_rew
        decayed_rew *= gamma
    return tot_value


def calc_sum_td(est_vals, rewards, gamma):
    assert len(est_vals) == len(rewards)
    ep_len = len(est_vals)
    td_errs = []
    for i in range(ep_len):
        true_val = est_vals[i+1] * gamma + rewards[i] if i < ep_len-1 else rewards[i]
        est_val = est_vals[i]
        td_err = (true_val - est_val)**2
        td_errs.append(td_err)
    return sum(td_errs)
