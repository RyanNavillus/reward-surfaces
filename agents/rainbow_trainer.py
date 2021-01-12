from .Rainbow.agent import Agent
from .Rainbow.env import Env
from .Rainbow.main import main, get_parser
from .Rainbow.memory import ReplayMemory
import torch
import numpy as np
from .evaluate import evaluate
from .evaluate_est_hesh import calculate_est_hesh_eigenvalues

class RainbowEvaluator:
    def __init__(self, env, agent, gamma, eval_trainer):
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.eval_trainer = eval_trainer
        self.state = env.reset()
        self.done = True
        self._step_num = 0

    def _next_state(self):
        action, value = self.agent.act_and_eval(self.state)  # Choose an action ε-greedily
        self.action = action
        if self.eval_trainer is not None:
            value = self.agent.evaluate_q(self.state)
        self.state, reward, done = self.env.step(action)  # Step
        if self._step_num % 1024 == 0:
            self.agent.reset_noise()
        if done:
            self.state = self.env.reset()
        return reward, done, value


class RainbowHeshEvaluator:
    def __init__(self, agent, env, args):
        self.agent = agent
        self.env = env
        self.args = args
        self.batch_size = args.batch_size
        self.device = args.device

    def parameters(self):
        return list(self.agent.online_net.parameters())

    def setup_buffer(self, num_samples):
        evaluator = RainbowEvaluator(self.env, self.agent, 1.0, None)
        self.buffer = ReplayMemory(self.args, num_samples)
        self.num_batches = num_samples//self.batch_size
        self.num_samples = num_samples
        for i in range(num_samples):
            rew, done, value = evaluator._next_state()
            state = evaluator.state
            action = evaluator.action
            self.buffer.append(state, action, rew, done)

    def calulate_grad_from_buffer(self, i):
        # print(i)
        # print(self.batch_size)
        indexs = np.arange(self.batch_size) + i*self.batch_size
        sample = self.buffer.get_samples_from_idxs(indexs)
        weights = torch.ones(self.batch_size, device=self.device)
        indexes = np.zeros(self.batch_size, dtype=np.int32)
        grad = self.agent.calulate_grad_from_buffer((indexes,)+sample+(weights,))
        return grad

    def calculate_hesh_vec_prod(self, vec, num_samples):
        assert num_samples == self.num_samples
        self.agent.online_net.zero_grad()
        for i in range(self.num_batches):
            grad = self.calulate_grad_from_buffer(i)
            loss = 0
            for g, p in zip(grad, vec):
                loss += (g*p).sum()

            loss.backward()

    def cleanup_buffer(self):
        self.buffer = None


class RainbowTrainer:
    def __init__(self, atari_env_name, learning_starts=int(20e3), device='cpu'):
        self.atari_env_name = atari_env_name
        self.learning_starts = learning_starts
        self.device = device
        parser = get_parser()
        args = parser.parse_args(self.args_list())
        if torch.cuda.is_available() and device != 'cpu' and not args.disable_cuda:
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')
        self.env = Env(args)
        self.agent = Agent(args, self.env)
        self.args = args

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
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.agent.online_net.load_state_dict(state_dict)

    def save_weights(self, checkpoint):
        self.agent.save_path(checkpoint)

    def get_weights(self):
        np_arrs = [param.cpu().detach().numpy() for param in self.agent.online_net.parameters()]
        return np_arrs

    def set_weights(self, np_arrs):
        for np_arr, param in zip(np_arrs, self.agent.online_net.parameters()):
            param.data = torch.tensor(np_arr,device=self.device)

    def evaluate(self, num_episodes, num_steps, eval_trainer=None):
        self.agent.train()
        self.env.eval()
        evaluator = RainbowEvaluator(self.env, self.agent, self.agent.discount, eval_trainer)
        return evaluate(evaluator, num_episodes, num_steps)

    def calculate_eigenvalues(self, num_steps, tol=1e-2):
        hesh_eval = RainbowHeshEvaluator(self.agent, self.env, self.args)
        return calculate_est_hesh_eigenvalues(hesh_eval,num_steps,tol)
