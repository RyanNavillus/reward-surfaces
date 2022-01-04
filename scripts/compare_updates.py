import json
import os
import argparse
from stable_baselines3 import PPO
from reward_surfaces.agents.make_agent import make_agent
from reward_surfaces.algorithms import evaluate

parser = argparse.ArgumentParser(description='Train an agent and keep track of important information.')
parser.add_argument('--device', type=str, default="cuda", help="Device used for training ('cpu' or 'cuda')")
parser.add_argument('--checkpoint', type=int, help="Specific checkpoint to test")
parser.add_argument('--name', type=str, help="Name of checkpoint")
parser.add_argument('--env', type=str, help="Environment name")
parser.add_argument('--checkpoint-file', type=str, help="File of checkpoint to test")
args = parser.parse_args()
assert not args.checkpoint or not args.checkpoint_file, "must choose either one checkpoint or checkpoint file"
if args.checkpoint:
    assert args.name and args.env, "Must specify an agent name when using a single checkpoint"


checkpoint_name = args.name
environment = args.env


def compare_a2c_ppo(environment, agent_name, checkpoint):
    directory = f"./runs/{agent_name}_checkpoints"
    pretraining = {
        "latest": f"{directory}/{checkpoint}/checkpoint.zip",
        "best": "./runs/{agent_name}/best/checkpoint.zip",
        "trained_steps": int(f"{checkpoint}"),
    }

    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "A2C"}'), eval_freq=4096, n_eval_episodes=100, pretraining=pretraining, device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    #agent.set_weights(weights)
    evaluator = agent.evaluator()
    print(evaluate(evaluator, 100, 100000000))
    agent.train(1, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)
    print(evaluate(evaluator, 100, 100000000))

    agent, steps = make_agent("SB3_ON", environment, directory, json.loads('{"ALGO": "PPO"}'), eval_freq=4096, n_eval_episodes=1, pretraining=pretraining, device=args.device)
    agent.load_weights(f"{directory}/{checkpoint}/checkpoint.zip")
    agent.train(1, f"./runs/vpg/{agent_name}/{checkpoint}", save_freq=10000)


if args.checkpoint:
    compare_a2c_ppo(args.env, args.name, args.checkpoint)
else:
    checkpoints = []
    with open(args.checkpoint_file, "r") as checkpoint_file:
        for line in checkpoint_file:
            words = line.split(" ")
            assert len(words) == 3, f"Incorrectly formatted checkpoints file (Expected 3 words per line, found {len(words)}"
            env, name, number = words[0], words[1], words[2]
            checkpoints.append((env, name, number))

    for checkpoint in checkpoints:
        env, name, number = checkpoint
        compare_a2c_ppo(env, name, number.strip())


# agent, steps = make_agent("SB3_ON", "InvertedDoublePendulum-v2", "./runs/vpg_test", json.loads('{"ALGO": "PPO", "n_timesteps": 100000}'), eval_freq=10000, device="cuda")
os.makedirs("./runs/vpg_test", exist_ok=True)
# agent.train(steps, "./runs/vpg_test", save_freq=10000)
# old_weights = agent.get_weights()



# norms = []
# for d1, d2 in zip(old_weights, agent.get_weights()):
#     norms.append(abs(np.linalg.norm(d1) - np.linalg.norm(d2)))
# print(sum(norms))
