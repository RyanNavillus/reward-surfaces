from Rainbow.agent import Agent
from Rainbow.env import Env
from Rainbow.main import main, get_parser
import torch
from evaluate import evaluate

class RainbowEvaluator:
    def __init__(self, env, agent, gamma, eval_trainer):
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.eval_trainer = eval_trainer
        self.state = env.reset()
        self.done = True

    def _next_state(self):
        action, value = self.agent.act_and_eval(self.state)  # Choose an action ε-greedily
        self.state, reward, done = self.env.step(action)  # Step
        if done:
            self.state = self.env.reset()
        return reward, done, value


class RainbowTrainer:
    def __init__(self, atari_env_name, learning_starts=int(20e3), device='cpu'):
        self.atari_env_name = atari_env_name
        self.learning_starts = learning_starts
        self.device = device
        parser = get_parser()
        args = parser.parse_args(self.args_list())
        if torch.cuda.is_available() and not args.disable_cuda:
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')
        self.env = Env(args)
        self.agent = Agent(args, self.env)

    def args_list(self):
        args = [
            f"--game={self.atari_env_name}",
            f"--learn-start={self.learning_starts}",
        ]
        if self.device != 'cuda':
            args.append('--disable-cuda')
        return args

    def train(self, num_steps, save_dir, save_freq=1000):
        parser = get_parser()
        args = self.args_list() + [
            f"--T-max={num_steps}",
            f"--results-dir={save_dir}",
            f"--checkpoint-interval={save_freq}",
        ]
        args = parser.parse_args(args)
        saved_files = main(args)
        print(saved_files)
        return saved_files

    def load_weights(self, checkpoint):
        state_dict = torch.load(checkpoint)
        self.agent.online_net.load_state_dict(state_dict)

    def save_weights(self, checkpoint):
        self.agent.save_path(checkpoint)

    def get_weights(self):
        np_arrs = [param.cpu().detach().numpy() for param in self.agent.online_net.parameters()]
        return np_arrs

    def set_weights(self, np_arrs):
        for np_arr, param in zip(np_arrs, self.agent.online_net.parameters()):
            param.data = torch.tensor(np_arr)

    def evaluate(self, num_episodes, num_steps, eval_trainer=None):
        evaluator = RainbowEvaluator(self.env, self.agent, self.agent.discount, eval_trainer)
        return evaluate(evaluator, num_episodes, num_steps)
