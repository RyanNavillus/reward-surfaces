import json
import os
import gym
import argparse
import math
import torch
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_ind
from stable_baselines3 import PPO, A2C
from reward_surfaces.agents.make_agent import make_agent
from reward_surfaces.algorithms import evaluate


def evaluate_agent(agent, env_name, num_steps=100000000, num_episodes=1, seed=1):
    total_steps = 0
    total_eps = 0
    total_rew = 0
    done = False
    env = gym.make(env_name)
    env.seed(seed=1)
    obs = env.reset()

    while not done or (total_eps < num_episodes and total_steps < num_steps):
        action, _ = agent.predict(obs, deterministic=True)
        with torch.no_grad():
            # Next state can be evaluated determinsitically for testing
            #rew, original_rew, done, value, _, _, info = evaluator._next_state_act()
            obs, reward, done, info = env.step(action)
        #all_datas.append((original_rew, done, value))
        total_steps += 1
        total_rew += reward
        if done:
            total_eps += 1
            obs = env.reset()
    return {"episode_rewards": float(total_rew) / total_eps}


def compare_a2c_ppo(environment, agent_name, checkpoint, episodes, trials, baseline_reward):
    directory = f"./runs/{agent_name}_checkpoints"
    pretraining = {
        "latest": f"{directory}/{checkpoint}/checkpoint.zip",
        "best": "./runs/{agent_name}/best/checkpoint.zip",
        "trained_steps": int(f"{checkpoint}"),
    }

    a2c_hypers = {
        "ALGO": "A2C",
        "seed": 1,
        "policy": "MlpPolicy",
        "n_envs": 8,
        "n_steps": 5,
        "ent_coef": 0.0,
        "learning_rate": 7e-4,
        "gae_lambda": 1,
        "policy_kwargs": '''{
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {
                "eps": 1e-5,
            },
        }''',
    }
    ppo_as_a2c_hypers = {
        "ALGO": "A2C",
        "seed": 1,
        "policy": "MlpPolicy",
        "n_envs": 8,
        "n_steps": 5,
        "ent_coef": 0.0,
        "learning_rate": 7e-4,
        "normalize_advantage": False,
        "gae_lambda": 1,
        "n_epochs": 1,
        "batch_size": 5,
        "clip_range": 0.2,
        "clip_range_vf": "None",
        "policy_kwargs": '''{
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {
                "eps": 1e-5,
            },
        }''',
    }
    policy_kwargs = {
        "optimizer_class": torch.optim.RMSprop,
        "optimizer_kwargs": {
            "alpha": 0.99,
            "eps": 1e-5,
            "weight_decay": 0
        },
    }

    # One step A2C
    a2c_scores = []
    for trial in range(trials):
        agent, steps = make_agent("SB3_ON", environment, directory, a2c_hypers,
                                  eval_freq=100000000,
                                  n_eval_episodes=1,
                                  device=args.device)
        for name, param in agent.algorithm.policy.named_parameters():
            if param.requires_grad:
                layer_param_sum = round(param.data.sum().item(), 4)
                print(f"{name}'s sum = {layer_param_sum}")
        #agent.algorithm.set_parameters(f"{directory}/{checkpoint}/checkpoint.zip")
        #attrs = vars(agent.algorithm)
        #for key, value in attrs.items():
        #    try:
        #        new_attrs = vars(value)
        #        print('\n'.join("%s: %s" % item for item in new_attrs.items()))
        #    except:
        #        print(key, ": ", value)

        agent.train(2048, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
        #agent = A2C('MlpPolicy', environment, seed=1+trial)
        #agent.load(f"{directory}/{checkpoint}/checkpoint.zip")
        #agent.learn(total_timesteps=2048)
        for name, param in agent.algorithm.policy.named_parameters():
            if param.requires_grad:
                layer_param_sum = round(param.data.sum().item(), 4)
                print(f"{name}'s sum = {layer_param_sum}")
        #evaluator = agent.evaluator()
        #a2c = evaluate(evaluator, episodes, 100000000)
        a2c = evaluate_agent(agent.algorithm, environment, num_episodes=episodes, seed=1+trial)
        print("A2C: ", a2c["episode_rewards"])
        a2c_scores.append(a2c["episode_rewards"])

    # One step PPO
    ppo_as_a2c_hypers["ALGO"] = "PPO"
    ppo_scores = []
    for trial in range(trials):
        agent, steps = make_agent("SB3_ON", environment, directory, ppo_as_a2c_hypers,
                                  eval_freq=100000000,
                                  n_eval_episodes=1,
                                  device=args.device)
        for name, param in agent.algorithm.policy.named_parameters():
            if param.requires_grad:
                layer_param_sum = round(param.data.sum().item(), 4)
                print(f"{name}'s sum = {layer_param_sum}")
        #agent.algorithm.set_parameters(f"{directory}/{checkpoint}/checkpoint.zip")
        attrs = vars(agent.algorithm)
        #print('\n'.join("%s: %s" % item for item in attrs.items()))
        #for key, value in attrs.items():
        #    try:
        #        new_attrs = vars(value)
        #        print('\n'.join("%s: %s" % item for item in new_attrs.items()))
        #    except:
        #        print(key, ": ", value)
        agent.train(2048, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
        #agent = PPO('MlpPolicy', environment, n_steps=5, ent_coef=0.0, learning_rate=7e-4, normalize_advantage=False, gae_lambda=1, n_epochs=1, batch_size=5, clip_range_vf=None, policy_kwargs=policy_kwargs, seed=1+trial)
        #agent.load(f"{directory}/{checkpoint}/checkpoint.zip")
        #agent.learn(total_timesteps=2048)
        for name, param in agent.algorithm.policy.named_parameters():
            if param.requires_grad:
                layer_param_sum = round(param.data.sum().item(), 4)
                print(f"{name}'s sum = {layer_param_sum}")
        #evaluator = agent.evaluator()
        #ppo = evaluate(evaluator, episodes, 100000000)
        ppo = evaluate_agent(agent.algorithm, environment, num_episodes=episodes, seed=1+trial)
        print("PPO: ", ppo["episode_rewards"])
        ppo_scores.append(ppo["episode_rewards"])

    # Calculate statistics
    print(ppo_scores)
    print(a2c_scores)
    print(baseline_reward)
    a2c_reward = sum(a2c_scores) / len(a2c_scores)
    ppo_reward = sum(ppo_scores) / len(ppo_scores)
    ppo_percent = math.copysign(1, baseline_reward) * ((ppo_reward / baseline_reward) - 1)
    a2c_percent = math.copysign(1, baseline_reward) * ((a2c_reward / baseline_reward) - 1)
    print(ppo_percent)
    print(a2c_percent)

    return ppo_percent, a2c_percent


def evaluate_checkpoint(inputs):
    env, name, number, episodes, trials, baseline = inputs
    return compare_a2c_ppo(env, name, number.strip(), episodes, trials, baseline)


# Cliff test
def test_checkpoints(checkpoint_filename, episodes, trials, baselines):
    checkpoints = []
    with open(checkpoint_filename, "r") as checkpoint_file:
        for i, line in enumerate(checkpoint_file):
            words = line.split(" ")
            assert len(words) == 3, f"Incorrectly formatted checkpoints file (Expected 3 words per line, found {len(words)}"
            env, name, number = words[0], words[1], words[2]
            checkpoints.append((env, name, number, episodes, trials, baselines[i]))

    print(checkpoints)
    pool = Pool(1)
    ppo_percents, a2c_percents = zip(*pool.map(evaluate_checkpoint, checkpoints))
    # Store results
    env_names = [cpt[0] for cpt in checkpoints]
    outfile.write(f"Env Names: {env_names}\n")
    outfile.write(f"Baseline: {baselines}\n")
    outfile.write(f"PPO: {ppo_percents}\n")
    outfile.write(f"A2C: {a2c_percents}\n")
    return ppo_percents, a2c_percents


def test_baselines(checkpoint_filename, episodes, trials):
    checkpoints = []
    with open(checkpoint_filename, "r") as checkpoint_file:
        for line in checkpoint_file:
            words = line.split(" ")
            assert len(words) == 3, f"Incorrectly formatted checkpoints file (Expected 3 words per line, found {len(words)}"
            env, name, number = words[0], words[1], words[2]
            checkpoints.append((env, name, number, episodes, trials))

    pool = Pool(1)
    baselines = pool.map(evaluate_baselines, checkpoints)
    return baselines


def evaluate_baselines(inputs):
    env, name, number, episodes, trials = inputs
    return compute_baselines(env, name, number.strip(), episodes, trials)


def compute_baselines(environment, agent_name, checkpoint, episodes, trials):
    directory = f"./runs/{agent_name}_checkpoints"
    pretraining = {
        "latest": f"{directory}/{checkpoint}/checkpoint.zip",
        "best": "./runs/{agent_name}/best/checkpoint.zip",
        "trained_steps": int(f"{checkpoint}"),
    }
    ppo_as_a2c_hypers = {
        "ALGO": "A2C",
        "seed": 1,
        "n_envs": 8,
        "n_steps": 5,
        "ent_coef": 0.0,
        "learning_rate": 7e-4,
        "normalize_advantage": False,
        "gae_lambda": 1,
        "n_epochs": 1,
        "batch_size": 5,
        "clip_range_vf": 10000000,
        "policy_kwargs": '''{
            "optimizer_class": "torch.optim.RMSprop",
            "optimizer_kwargs": {
                "alpha": 0.99,
                "eps": 1e-5,
                "weight_decay": 0
            },
        }''',
    }

    # Baseline
    baseline_scores = []
    for trial in range(trials):
        agent, steps = make_agent("SB3_ON", environment, directory, {"ALGO": "A2C", "seed": 1, "policy": "MlpPolicy"},
                                  eval_freq=100000000,
                                  n_eval_episodes=1,
                                  device=args.device)
        #agent.algorithm.set_parameters(f"{directory}/{checkpoint}/checkpoint.zip")
        #agent = PPO('MlpPolicy', environment)
        #agent.load(f"{directory}/{checkpoint}/checkpoint.zip")
        #agent.set_weights(weights)
        #evaluator = agent.evaluator()
        #baseline = evaluate(evaluator, episodes, 100000000)
        baseline = evaluate_agent(agent.algorithm, environment, num_episodes=episodes, seed=1+trial)
        print("Baseline: ", baseline["episode_rewards"])
        baseline_scores.append(baseline["episode_rewards"])
    return sum(baseline_scores) / len(baseline_scores)


parser = argparse.ArgumentParser(description='Train an agent and keep track of important information.')
parser.add_argument('--device', type=str, default="cuda", help="Device used for training ('cpu' or 'cuda')")
parser.add_argument('--cliff-file', type=str, help="File of cliff checkpoints to test")
parser.add_argument('--noncliff-file', type=str, help="File of non-cliff checkpoints to test")
parser.add_argument('--num-episodes', type=int, default=100, help="Number of evaluation episodes")
parser.add_argument('--num-trials', type=int, default=5, help="Number of evaluation trials")
args = parser.parse_args()


outfile = open("vpg_output.txt", "w")
cliff_baselines = test_baselines(args.cliff_file, args.num_episodes, args.num_trials)
noncliff_baselines = test_baselines(args.noncliff_file, args.num_episodes, args.num_trials)
outfile.write("Cliffs:\n")
cliff_ppo_percents, cliff_a2c_percents = test_checkpoints(args.cliff_file, args.num_episodes, args.num_trials, cliff_baselines)
outfile.write("Non-Cliffs:\n")
noncliff_ppo_percents, noncliff_a2c_percents = test_checkpoints(args.noncliff_file, args.num_episodes, args.num_trials, noncliff_baselines)
outfile.write("\nResults:\n")


def calculate_stats(data):
    return np.mean(data), np.std(data, ddof=1), len(data)


cliff_ppo_mean, cliff_ppo_std, cliff_ppo_n = calculate_stats(cliff_ppo_percents)
noncliff_ppo_mean, noncliff_ppo_std, noncliff_ppo_n = calculate_stats(noncliff_ppo_percents)
cliff_a2c_mean, cliff_a2c_std, cliff_a2c_n = calculate_stats(cliff_a2c_percents)
noncliff_a2c_mean, noncliff_a2c_std, noncliff_a2c_n = calculate_stats(noncliff_a2c_percents)
print(cliff_ppo_mean, cliff_ppo_std, cliff_ppo_n)
print(noncliff_ppo_mean, noncliff_ppo_std, noncliff_ppo_n)
print(cliff_a2c_mean, cliff_a2c_std, cliff_a2c_n)
print(noncliff_a2c_mean, noncliff_a2c_std, noncliff_a2c_n)

# Perform T-Test
# Evaluate the probability that the difference between the difference between the percent improvement for cliff vs. noncliff checkpoints for a2c and ppo is due to chance
ppo_t_stat = (cliff_ppo_mean - noncliff_ppo_mean) / math.sqrt(((cliff_ppo_std**2) / cliff_ppo_n) + ((noncliff_ppo_std**2) / noncliff_ppo_n))
a2c_t_stat = (cliff_a2c_mean - noncliff_a2c_mean) / math.sqrt(((cliff_a2c_std**2) / cliff_a2c_n) + ((noncliff_a2c_std**2) / noncliff_a2c_n))
print("PPO T test statistic: ", ppo_t_stat)
print("A2C T test statistic: ", a2c_t_stat)

print(ttest_ind(cliff_ppo_percents, noncliff_ppo_percents, equal_var=False))
print(ttest_ind(cliff_a2c_percents, noncliff_a2c_percents, equal_var=False))
outfile.write("PPO: " + str(ttest_ind(cliff_ppo_percents, noncliff_ppo_percents, equal_var=False)) + "\n")
outfile.write("A2C: " + str(ttest_ind(cliff_a2c_percents, noncliff_a2c_percents, equal_var=False)) + "\n")


print("PPO diff:", cliff_ppo_mean, noncliff_ppo_mean)
print("A2C diff:", cliff_a2c_mean, noncliff_a2c_mean)
outfile.write(f"PPO diff: {cliff_ppo_mean}, {noncliff_ppo_mean} \n")
outfile.write(f"A2C diff: {cliff_a2c_mean}, {noncliff_a2c_mean}, \n")
