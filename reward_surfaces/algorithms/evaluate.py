import time
import torch
import numpy as np
import warnings
from stable_baselines3.common.vec_env import VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


def generate_data(evaluator, num_episodes, num_steps):
    all_datas = []
    tot_steps = 0
    tot_eps = 0
    tot_rew = 0
    last_rew = 0
    start_t = time.time()
    done = False

    while not done or (tot_eps < num_episodes and tot_steps < num_steps):
        with torch.no_grad():
            # Next state can be evaluated determinsitically for testing
            rew, done, value, _, _, info = evaluator._next_state_act()
        all_datas.append((rew, done, value))
        tot_steps += 1
        tot_rew += rew
        if done:
            tot_eps += 1
            end_t = time.time()
            last_rew = tot_rew
            #if tot_eps % 25 == 0:
                #print("done!", (end_t - start_t)/tot_steps, (end_t - start_t)/tot_eps, tot_rew/tot_eps)
    return all_datas


def generate_monitor_data(evaluator, num_episodes, num_steps):
    print(num_episodes, num_steps)
    all_datas = []
    done = False
    tot_eps = 0
    tot_steps = 0

    while not done or (tot_eps < num_episodes and tot_steps < num_steps):
        with torch.no_grad():
            # Next state can be evaluated determinsitically for testing
            rew, done, value, _, _, info = evaluator._next_state_act()
            tot_steps += 1
        if "episode" in info.keys():
            episode_reward = info["episode"]["r"]
            episode_length = info["episode"]["l"]
            tot_eps += 1
            all_datas.append((rew, value, episode_reward, episode_length))
        else:
            all_datas.append((rew, value))
    return all_datas


def evaluate(evaluator, num_episodes, num_steps):
    is_monitor_wrapped = is_vecenv_wrapped(evaluator.env, VecMonitor) or evaluator.env.env_is_wrapped(Monitor)[0]
    if not is_monitor_wrapped:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    if is_monitor_wrapped:
        all_datas = generate_monitor_data(evaluator, num_episodes, num_steps)
        return calculate_monitor_stats(all_datas, evaluator.gamma)
    else:
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
        "episode_std_rewards": float(np.std(episode_rewards/episode_lens)),
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


def calculate_monitor_stats(all_datas, gamma):
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
    for data in all_datas:
        if len(data) == 4:
            rew, value, episode_rew, episode_len = data
            ep_vals.append(value)
            ep_rews.append(rew)

            episode_rewards.append(episode_rew)
            episode_value_ests.append(sum(ep_vals))
            episode_values.append(calc_sum_value(ep_rews, gamma))
            # TODO: Make this real
            episode_td_err.append(calc_sum_td(ep_vals, ep_rews, gamma))
            episode_lens.append(episode_len)
            ep_rews = []
            ep_vals = []
        elif len(data) == 2:
            rew, value = data
            ep_vals.append(value)
            ep_rews.append(rew)

    episode_rewards = np.array(episode_rewards, dtype=np.float64)
    episode_value_ests = np.array(episode_value_ests)
    episode_values = np.array(episode_values)
    episode_td_err = np.array(episode_td_err)
    episode_lens = np.array(episode_lens, dtype=np.float64)
    return {
        "episode_rewards": float(np.mean(episode_rewards)),
        "episode_std_rewards": float(np.std(episode_rewards/episode_lens)),
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
