import argparse
import sys
print(sys.path)
import reward_surfaces
from reward_surfaces.agents.make_agent import make_agent
import torch
import json
import os


def main():
    parser = argparse.ArgumentParser(description='run a particular evaluation job')
    parser.add_argument('save_dir', type=str)
    parser.add_argument('agent_name', type=str)
    parser.add_argument('env', type=str)
    parser.add_argument('device', type=str)
    parser.add_argument('hyperparameters', type=str)
    parser.add_argument('--save_freq', type=int, default=10000)

    args = parser.parse_args()

    torch.set_num_threads(1)

    #trainer = SB3HerPolicyTrainer(robo_env_fn,HER("MlpPolicy",robo_env_fn(),model_class=TD3,device="cpu",max_episode_length=100))
    agent, steps = make_agent(args.agent_name, args.env, args.device, json.loads(args.hyperparameters))

    os.makedirs(args.save_dir, exist_ok=False)

    hyperparams = json.loads(args.hyperparameters)
    run_info = {
        "agent_name": args.agent_name,
        "env": args.env,
        "hyperparameters": hyperparams,
    }
    run_info_fname = os.path.join(args.save_dir, "info.json")
    with open(run_info_fname, 'w') as file:
        file.write(json.dumps(run_info, indent=4))

    agent.train(steps, args.save_dir, save_freq=args.save_freq)


if __name__ == "__main__":
    main()
