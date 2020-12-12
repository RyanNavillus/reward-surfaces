import argparse
from agents.make_agent import make_agent
import torch
import json

def main():
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--agent_name', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--hyperparameters', type=str, required=True)
    parser.add_argument('--eval-seperate', action="store_true")

    args = parser.parse_args()

    agent = make_agent(args.agent_name, args.env, args.device, json.loads(args.hyperparameters))
    agent.load_weights(args.checkpoint_path)

    weights = torch.load(args.params_path)
    weights = [w.cpu().numpy() for n,w in weights.items()]
    agent.set_weights(weights)

    eval_agent = None
    if args.eval_seperate:
        eval_agent = make_agent(args.agent_name, args.env, args.device, json.loads(args.hyperparameters))
        eval_agent.load_weights(args.checkpoint_path)

    vals = agent.evaluate(10, 1000, eval_agent)
    print(vals)
    # valstr = ",".join([str(val) for val in vals])
    with open(args.output_path,'w') as file:
        file.write(json.dumps(vals))


if __name__ == "__main__":
    main()
