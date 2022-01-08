import json
import os
import argparse
import math
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_ind
from stable_baselines3 import PPO
from reward_surfaces.agents.make_agent import make_agent
from reward_surfaces.algorithms import evaluate

parser = argparse.ArgumentParser(description='Train an agent and keep track of important information.')
parser.add_argument('--device', type=str, default="cuda", help="Device used for training ('cpu' or 'cuda')")
parser.add_argument('--cliff-file', type=str, help="File of cliff checkpoints to test")
parser.add_argument('--noncliff-file', type=str, help="File of non-cliff checkpoints to test")
parser.add_argument('--num-episodes', type=int, default=10, help="Number of evaluation episodes")
args = parser.parse_args()


def compare_a2c_ppo(environment, agent_name, checkpoint, episodes, baseline_reward):
    directory = f"./runs/{agent_name}_checkpoints"
    pretraining = {
        "latest": f"{directory}/{checkpoint}/checkpoint.zip",
        "best": "./runs/{agent_name}/best/checkpoint.zip",
        "trained_steps": int(f"{checkpoint}"),
    }

    # Baseline
    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "A2C"}'),
                              eval_freq=100000000,
                              n_eval_episodes=1,
                              pretraining=pretraining,
                              device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    # agent.set_weights(weights)
    evaluator = agent.evaluator()

    # One step A2C
    agent.train(1, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
    a2c = evaluate(evaluator, episodes, 100000000)
    print("A2C: ", a2c["episode_rewards"])

    # One step PPO
    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "PPO"}'),
                              eval_freq=100000000,
                              n_eval_episodes=1,
                              pretraining=pretraining,
                              device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    agent.train(1, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
    ppo = evaluate(evaluator, episodes, 100000000)
    print("PPO: ", ppo["episode_rewards"])

    # Calculate statistics
    a2c_reward = a2c["episode_rewards"]
    ppo_reward = ppo["episode_rewards"]
    ppo_percent = math.copysign(1, baseline_reward) * ((ppo_reward / baseline_reward) - 1)
    a2c_percent = math.copysign(1, baseline_reward) * ((a2c_reward / baseline_reward) - 1)
    return ppo_percent, a2c_percent


def evaluate_checkpoint(inputs):
    env, name, number, episodes, baseline = inputs
    return compare_a2c_ppo(env, name, number.strip(), episodes, baseline)


# Cliff test
def test_checkpoints(checkpoint_filename, episodes, baselines):
    checkpoints = []
    with open(checkpoint_filename, "r") as checkpoint_file:
        for i, line in enumerate(checkpoint_file):
            words = line.split(" ")
            assert len(words) == 3, f"Incorrectly formatted checkpoints file (Expected 3 words per line, found {len(words)}"
            env, name, number = words[0], words[1], words[2]
            checkpoints.append((env, name, number, episodes, baselines[i]))

    pool = Pool(20)
    ppo_percents, a2c_percents = zip(*pool.map(evaluate_checkpoint, checkpoints))
    return ppo_percents, a2c_percents


def test_baselines(checkpoint_filename, episodes):
    checkpoints = []
    with open(checkpoint_filename, "r") as checkpoint_file:
        for line in checkpoint_file:
            words = line.split(" ")
            assert len(words) == 3, f"Incorrectly formatted checkpoints file (Expected 3 words per line, found {len(words)}"
            env, name, number = words[0], words[1], words[2]
            checkpoints.append((env, name, number, episodes))

    pool = Pool(20)
    baselines = pool.map(evaluate_baselines, checkpoints)
    return baselines


def evaluate_baselines(inputs):
    env, name, number, episodes = inputs
    return compute_baselines(env, name, number.strip(), episodes)


def compute_baselines(environment, agent_name, checkpoint, episodes):
    directory = f"./runs/{agent_name}_checkpoints"
    pretraining = {
        "latest": f"{directory}/{checkpoint}/checkpoint.zip",
        "best": "./runs/{agent_name}/best/checkpoint.zip",
        "trained_steps": int(f"{checkpoint}"),
    }

    # Baseline
    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "PPO"}'),
                              eval_freq=100000000,
                              n_eval_episodes=1,
                              pretraining=pretraining,
                              device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    # agent.set_weights(weights)
    evaluator = agent.evaluator()
    baseline = evaluate(evaluator, episodes, 100000000)
    print("Baseline: ", baseline["episode_rewards"])
    return baseline["episode_rewards"]


cliff_baselines = test_baselines(args.cliff_file, args.num_episodes)
noncliff_baselines = test_baselines(args.noncliff_file, args.num_episodes)
cliff_ppo_percents, cliff_a2c_percents = test_checkpoints(args.cliff_file, args.num_episodes, cliff_baselines)
noncliff_ppo_percents, noncliff_a2c_percents = test_checkpoints(args.noncliff_file, args.num_episodes, noncliff_baselines)


def calculate_stats(data):
    return np.mean(data), np.std(data, ddof=1), len(data)


ppo_diffs = np.array(list(cliff_ppo_percents)) - np.array(list(noncliff_ppo_percents))
a2c_diffs = np.array(list(cliff_a2c_percents)) - np.array(list(noncliff_a2c_percents))
ppo_mean, ppo_std, ppo_n = calculate_stats(ppo_diffs)
a2c_mean, a2c_std, a2c_n = calculate_stats(a2c_diffs)

# Perform T-Test
# Evaluate the probability that the difference between the difference between the percent improvement for cliff vs. noncliff checkpoints for a2c and ppo is due to chance
t_stat = (ppo_mean - a2c_mean) / math.sqrt(((ppo_std**2) / ppo_n) + ((a2c_std**2) / a2c_n))
print("T test statistic: ", t_stat)

print(ttest_ind(ppo_diffs, a2c_diffs, equal_var=False))


print("PPO diff:", ppo_mean)
print("A2C diff:", a2c_mean)


# agent, steps = make_agent("SB3_ON", "InvertedDoublePendulum-v2", "./runs/vpg_test", json.loads('{"ALGO": "PPO", "n_timesteps": 100000}'), eval_freq=10000, device="cuda")
os.makedirs("./runs/vpg_test", exist_ok=True)
# agent.train(steps, "./runs/vpg_test", save_freq=10000)
# old_weights = agent.get_weights()



# norms = []
# for d1, d2 in zip(old_weights, agent.get_weights()):
#     norms.append(abs(np.linalg.norm(d1) - np.linalg.norm(d2)))
# print(sum(norms))
