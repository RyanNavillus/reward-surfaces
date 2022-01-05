import json
import os
import argparse
from stable_baselines3 import PPO
from reward_surfaces.agents.make_agent import make_agent
from reward_surfaces.algorithms import evaluate

parser = argparse.ArgumentParser(description='Train an agent and keep track of important information.')
parser.add_argument('--device', type=str, default="cuda", help="Device used for training ('cpu' or 'cuda')")
parser.add_argument('--cliff-file', type=str, help="File of cliff checkpoints to test")
parser.add_argument('--noncliff-file', type=str, help="File of non-cliff checkpoints to test")
args = parser.parse_args()


def compare_a2c_ppo(environment, agent_name, checkpoint):
    directory = f"./runs/{agent_name}_checkpoints"
    pretraining = {
        "latest": f"{directory}/{checkpoint}/checkpoint.zip",
        "best": "./runs/{agent_name}/best/checkpoint.zip",
        "trained_steps": int(f"{checkpoint}"),
    }

    # Baseline
    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "A2C"}'), eval_freq=100000000, n_eval_episodes=1, pretraining=pretraining, device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    #agent.set_weights(weights)
    evaluator = agent.evaluator()
    baseline = evaluate(evaluator, 10, 100000000)
    print("Baseline: ", baseline["episode_rewards"])

    # One step A2C
    agent.train(1, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
    a2c = evaluate(evaluator, 10, 100000000)
    print("A2C: ", a2c["episode_rewards"])

    # One step PPO
    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "PPO"}'), eval_freq=100000000, n_eval_episodes=1, pretraining=pretraining, device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    agent.train(1, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
    ppo = evaluate(evaluator, 10, 100000000)
    print("PPO: ", ppo["episode_rewards"])

    # Calculate statistics
    baseline_reward = baseline["episode_rewards"]
    a2c_reward = a2c["episode_rewards"]
    ppo_reward = ppo["episode_rewards"]
    ppo_percent = ppo_reward / baseline_reward
    a2c_percent = a2c_reward / baseline_reward
    return ppo_percent, a2c_percent

# Cliff test
def test_checkpoints(checkpoint_filename):
    checkpoints = []
    with open(checkpoint_filename, "r") as checkpoint_file:
        for line in checkpoint_file:
            words = line.split(" ")
            assert len(words) == 3, f"Incorrectly formatted checkpoints file (Expected 3 words per line, found {len(words)}"
            env, name, number = words[0], words[1], words[2]
            checkpoints.append((env, name, number))
 
    ppo_percents = []
    a2c_percents = []
    for checkpoint in checkpoints:
        env, name, number = checkpoint
        ppo_percent, a2c_percent = compare_a2c_ppo(env, name, number.strip())
        ppo_percents.append(ppo_percent)
        a2c_percents.append(a2c_percent)
    ppo_avg_percent = sum(ppo_percents) / len(ppo_percents)
    a2c_avg_percent = sum(a2c_percents) / len(a2c_percents)
    return ppo_avg_percent, a2c_avg_percent

cliff_ppo, cliff_a2c = test_checkpoints(args.cliff_file)
noncliff_ppo, noncliff_a2c = test_checkpoints(args.noncliff_file)
print(noncliff_ppo, cliff_ppo)
print(noncliff_a2c, cliff_a2c)


# agent, steps = make_agent("SB3_ON", "InvertedDoublePendulum-v2", "./runs/vpg_test", json.loads('{"ALGO": "PPO", "n_timesteps": 100000}'), eval_freq=10000, device="cuda")
os.makedirs("./runs/vpg_test", exist_ok=True)
# agent.train(steps, "./runs/vpg_test", save_freq=10000)
# old_weights = agent.get_weights()



# norms = []
# for d1, d2 in zip(old_weights, agent.get_weights()):
#     norms.append(abs(np.linalg.norm(d1) - np.linalg.norm(d2)))
# print(sum(norms))
