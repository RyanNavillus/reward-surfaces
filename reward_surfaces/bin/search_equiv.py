import argparse
import json
import os
import torch
from reward_surfaces.utils.surface_utils import readz
from pathlib import Path
from reward_surfaces.algorithms import search


bigint = 1000000000000
def main():
    parser = argparse.ArgumentParser(description='run a particular evaluation job')
    parser.add_argument('params', type=str)
    parser.add_argument('dir', type=str, help="direction to search for threshold")
    parser.add_argument('outputfile', type=str)
    parser.add_argument('--num-steps', type=int, default=bigint)
    parser.add_argument('--num-episodes', type=int, default=bigint)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--key', type=str, default="episode_rewards")
    parser.add_argument('--tolerance', type=float, required=True)

    args = parser.parse_args()

    info = json.load(open(Path(args.params).parent.parent / "info.json"))
    params = [v.cpu().detach().numpy() for v in torch.load(args.params,map_location=torch.device('cpu')).values()]
    dir = readz(args.dir)

    up_bound = search(info, params, dir, args.num_steps, args.num_episodes, args.key, args.device, args.tolerance)

    out_dict = {
        "offset": up_bound,
    }
    with open(args.outputfile, 'w') as file:
        file.write(json.dumps(out_dict))

if __name__ == "__main__":
    main()
