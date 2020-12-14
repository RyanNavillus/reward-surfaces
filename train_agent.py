import argparse
from agents.make_agent import make_agent
import torch
import json
import os


def filter_normalize(param):
    # TODO: verify with loss landscapes code
    ndims = len(param.shape)
    if ndims == 1 or ndims == 0:
        # don't do any random direction for scalars
        return np.zeros_like(param)
    elif ndims == 2:
        dir = np.random.normal(size=param.shape)
        dir /= np.sqrt(np.sum(np.square(dir),axis=0,keepdims=True))
        dir *= np.sqrt(np.sum(np.square(param),axis=0,keepdims=True))
        return dir
    elif ndims == 4:
        dir = np.random.normal(size=param.shape)
        dir /= np.sqrt(np.sum(np.square(dir),axis=(0,1,2),keepdims=True))
        dir *= np.sqrt(np.sum(np.square(param),axis=(0,1,2),keepdims=True))
        return dir
    else:
        assert False, "only 1, 2, 4 dimentional filters allowed, got {}".format(param.shape)


def fischer_normalization(aget):
    


def main():
    parser = argparse.ArgumentParser(description='run a particular evaluation job')
    parser.add_argument('save_dir', type=str)
    parser.add_argument('num_steps', type=int)
    parser.add_argument('agent_name', type=str)
    parser.add_argument('env', type=str)
    parser.add_argument('device', type=str)
    parser.add_argument('hyperparameters', type=str)
    parser.add_argument('--save_freq', type=int, default=10000)

    args = parser.parse_args()

    #trainer = SB3HerPolicyTrainer(robo_env_fn,HER("MlpPolicy",robo_env_fn(),model_class=TD3,device="cpu",max_episode_length=100))
    agent = make_agent(args.agent_name, args.env, args.device, json.loads(args.hyperparameters))

    os.makedirs(args.save_dir,exist_ok=False)

    agent.train(args.num_steps, args.save_dir, save_freq=args.save_freq)

    run_info = {
        "agent_name": args.agent_name,
        "env": args.env,
        "hyperparameters": json.loads(args.hyperparameters),
    }
    run_info_fname = os.path.join(args.save_dir, "info.json")
    with open(run_info_fname, 'w') as file:
        file.write(json.dumps(run_info,indent=4))


if __name__ == "__main__":
    main()
