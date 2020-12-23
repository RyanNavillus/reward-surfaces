import time

class Evaluator:
    def _next_state(self):
        '''
        return rew, done, value
        '''

def evaluate(evaluator, num_episodes, num_steps):
    gamma = evaluator.gamma

    ep_rews = []
    ep_vals = []
    episode_rewards = []
    episode_value_ests = []
    episode_values = []
    episode_td_err = []
    tot_steps = 0
    start_t = time.time()
    done = False
    while not done or (len(episode_rewards) < num_episodes and tot_steps < num_steps):
        rew, done, value = evaluator._next_state()#, deterministic=True)
        ep_vals.append(value)
        ep_rews.append(rew)
        tot_steps += 1
        if done:
            ep_rew_tot = sum(ep_rews)
            episode_rewards.append(sum(ep_rews))
            episode_value_ests.append(mean(ep_vals))
            episode_values.append(calc_mean_value(ep_rews, gamma))
            episode_td_err.append(calc_mean_td(ep_vals, ep_rews, gamma))#TODO make this real

            ep_rews = []
            ep_vals = []
            end_t = time.time()
            print("done!", (end_t - start_t)/tot_steps, (end_t - start_t)/len(episode_rewards),ep_rew_tot)

    return {
        "episode_rewards": mean(episode_rewards),
        "episode_value_ests": mean(episode_value_ests),
        "episode_values": mean(episode_values),
        "episode_td_err": mean(episode_td_err),
    }


def mean(vals):
    return sum(vals)/len(vals)


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
