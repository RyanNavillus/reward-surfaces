from reward_surfaces.agents.make_agent import make_agent
from reward_surfaces.algorithms import evaluate
import multiprocessing
import numpy as np
import math


def gen_bounds(lower, upper, exp):
    cur = lower
    res = []
    while cur < upper:
        res.append(cur)
        cur *= exp
    return res


def calc_result(offset, params, info, dir, device, key, num_steps, num_episodes):
    agent = make_agent(info['agent_name'], info['env'], device, info['hyperparameters'])

    weights = [p+g*offset for p,g in zip(params, dir)]
    agent.set_weights(weights)

    evaluator = agent.evaluator()
    eval_results = evaluate(evaluator, num_episodes, num_steps)

    return eval_results[key]

def calc_result_single(args):
    return calc_result(*args)

def which_meet_threshold(offsets, thresh, n_cpus, *calc_args):
    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count()
    if n_cpus <= 1:
        return [calc_result(off, *calc_args) > thresh for off in offsets]
    else:
        pool = multiprocessing.Pool(n_cpus)
        args = [(off,)+calc_args for off in offsets]

        result = pool.map(calc_result_single, args)
        met = [res > thresh for res in result]
        pool.close()
        return met

bigint = 1000000000000
def search(info, params, dir, num_steps=bigint, num_episodes=bigint, key="episode_rewards", device="cpu", tolerance=0.05):

    assert num_steps < bigint or num_episodes < bigint, "need to set one of num-steps or num-episodes"

    calc_args = (params, info, dir, device, key, num_steps, num_episodes)

    orig_calc = (1. - tolerance) * calc_result(0.0, *calc_args)

    n_cpus = 1
    exp = 10
    lower_bound = 1e-7
    upper_bound = 1e3
    while exp > 1.1:
        bounds = gen_bounds(lower_bound, upper_bound, exp)
        print(bounds)
        if len(bounds) > 3:
            met = which_meet_threshold(bounds, orig_calc, n_cpus, *calc_args)
            if not any(met):
                return lower_bound
            met_loc = max(loc for loc, val in enumerate(met) if val)
            high_loc = min(len(met)-1, met_loc+2)
            low_loc = max(0, high_loc-2)
            met_bound = bounds[met_loc]
            lower_bound = bounds[low_loc]
            upper_bound = bounds[high_loc]

        exp = math.pow(exp, 0.7)

    return met_bound
