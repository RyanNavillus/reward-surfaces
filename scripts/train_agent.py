import argparse
from reward_surfaces.agents.make_agent import make_agent
import torch
import json
import os
from glob import glob


def main():
    parser = argparse.ArgumentParser(description='Train an agent and keep track of important information.')
    parser.add_argument('save_dir', type=str, help="Directory where checkpoints will be saved")
    parser.add_argument('agent_name', type=str, help="One of 'rainbow', 'SB3_OFF', 'SB3_ON', or 'SB3_HER'")
    parser.add_argument('env', type=str, help="Environment name")
    parser.add_argument('device', type=str, help="Device used for training ('cpu' or 'cuda')")
    parser.add_argument('hyperparameters', type=str, help="Dictionary of hyperparameters for training. Should include the intended training algorithm (E.g. {'ALGO': 'PPO'})")
    parser.add_argument('--save_freq', type=int, default=10000, help="Training steps between each saved checkpoint.")
    parser.add_argument('--resume', action='store_true', help="Continue training from last checkpoint")

    args = parser.parse_args()
    assert args.agent_name in ['rainbow', 'SB3_OFF', 'SB3_ON', 'SB3_HER'], "Name must be one of 'rainbow', 'SB3_OFF', 'SB3_ON', or 'SB3_HER'"

    torch.set_num_threads(1)

    zip_path = ""
    timesteps = 0
    pretraining = None
    if args.resume:
        subdirs = glob(args.save_dir+"/*/")
        for i, subdir in enumerate(subdirs):
            parts = subdir.split("/")
            subdirs[i] = ""
            for part in parts:
                if part.isdigit():
                    subdirs[i] = int(part)
        subdirs = sorted(list(filter(lambda a: a != "", subdirs)))
        latest_checkpoint = subdirs.pop()
        timesteps = int(latest_checkpoint)
        zip_path = args.save_dir + "/" + latest_checkpoint + "/checkpoint.zip"
        best_path = args.save_dir + "/best/checkpoint.zip"
        pretraining = {
            "latest": zip_path,
            "best": best_path,
            "trained_steps": timesteps,
        }
        print(zip_path)

    # trainer = SB3HerPolicyTrainer(robo_env_fn,HER("MlpPolicy",robo_env_fn(),model_class=TD3,device="cpu",max_episode_length=100))
    print(args.resume)
    agent, steps = make_agent(args.agent_name, args.env, args.device, args.save_dir, json.loads(args.hyperparameters),
                              pretraining=pretraining)

    os.makedirs(args.save_dir, exist_ok=True)

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
