import torch
import numpy as np


def generate_data(evaluator, num_episodes, num_steps):
    all_datas = []
    tot_steps = 0
    tot_eps = 0
    tot_rew = 0
    done = False

    while not done or (tot_eps < num_episodes and tot_steps < num_steps):
        with torch.no_grad():
            # Next state can be evaluated determinsitically for testing
            rew, original_rew, done, value, _, _, info = evaluator._next_state_act()
        all_datas.append((original_rew, done, value))
        tot_steps += 1
        tot_rew += original_rew
        if done:
            tot_eps += 1
    return all_datas


def evaluate(evaluator, num_episodes, num_steps):
    all_datas = generate_data(evaluator, num_episodes, num_steps)
    return calculate_stats(all_datas, evaluator.gamma)


def calculate_stats(all_datas, gamma):
    """
    Calculates statistics for the given episodic data

    Calculates episode rewards (mean and std), episode values, episode value estimates, episode td error,
    episode length (mean and std), average rewards per step, value per step, average value estimates per step, and
    average td error per step.
    """
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
            # TODO: Make this real
            episode_td_err.append(calc_sum_td(ep_vals, ep_rews, gamma))
            episode_lens.append(len(ep_rews))
            ep_rews = []
            ep_vals = []

    episode_rewards = np.array(episode_rewards, dtype=np.float64)
    episode_value_ests = np.array(episode_value_ests)
    episode_values = np.array(episode_values)
    episode_td_err = np.array(episode_td_err)
    episode_lens = np.array(episode_lens, dtype=np.float64)
    print(episode_rewards)
    return {
        "episode_rewards": float(np.mean(episode_rewards)),
        "episode_std_rewards": float(np.std(episode_rewards)),
        "episode_value_ests": float(np.mean(episode_value_ests/episode_lens)),
        "episode_values": float(np.mean(episode_values/episode_lens)),
        "episode_td_err": float(np.mean(episode_td_err/episode_lens)),
        "episode_avg_len": float(np.mean(episode_lens)),
        "episode_std_avg_len": float(np.std(episode_lens)),
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
    for i in range(len(rews)-1, -1, -1):
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
