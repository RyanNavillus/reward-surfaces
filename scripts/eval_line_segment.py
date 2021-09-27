import argparse
import json
import os
from pathlib import Path, PurePosixPath

bigint = 1000000000000


def main():
    parser = argparse.ArgumentParser(description='run a particular evaluation job')
    parser.add_argument('train_dir', type=str)
    parser.add_argument('grads_dir', type=str, help="direction to search for threshold")
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--num-steps', type=int, default=bigint)
    parser.add_argument('--num-episodes', type=int, default=bigint)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--length', type=int, default=5)
    parser.add_argument('--max-magnitude', type=float, default=0.1)
    parser.add_argument('--scale-dir', action="store_true")
    parser.add_argument('--random-dir-seed', type=int, help="if set, specified the seed for the random direction")

    args = parser.parse_args()

    assert args.num_steps != bigint or args.num_episodes != bigint, "one of num_steps or num_episodes must be specified"

    out_path = Path(args.output_dir)
    train_path = Path(args.train_dir)
    grad_path = PurePosixPath(args.grads_dir)
    os.mkdir(out_path)
    os.mkdir(out_path/"results")

    checkpoints = sorted([checkpoint for checkpoint in os.listdir(train_path) if (os.path.isdir(train_path/checkpoint)
                                                                                  and checkpoint.isdigit())])
    commands = []
    for checkpoint in checkpoints:
        params = train_path/checkpoint/"parameters.th"
        grad = grad_path/checkpoint/"grad.npz"
        if not os.path.isfile(grad):
            continue
        out_fname = str(out_path/"results"/checkpoint)
        command = f"python3 -m reward_surfaces.bin.eval_line {params} {grad} {out_fname} --num-steps={args.num_steps} --num-episodes={args.num_episodes} --device {args.device} --length {args.length} --max-magnitude {args.max_magnitude}"
        commands.append(command)

    info = json.load(open(train_path/"info.json"))
    info['num_steps'] = args.num_steps
    info['num_episodes'] = args.num_episodes
    info['draw_length'] = args.length
    info['draw_max_magnitude'] = args.max_magnitude
    info['grad_loc'] = args.grads_dir
    info['params_loc'] = args.train_dir
    info['scale_dir'] = args.scale_dir
    info['random_dir_seed'] = args.random_dir_seed

    json.dump(info, open(out_path/"info.json", 'w'))

    with open(out_path/"jobs.sh", 'w') as file:
        file.write("\n".join(commands))
    # run_job_list_list(commands)
    # job_results_to_csv(out_path)

if __name__ == "__main__":
    main()
