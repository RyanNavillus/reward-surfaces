from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
import numpy as np
import torch
import torch as th
from torch.nn import functional as F
from gym import spaces
from scipy.sparse.linalg import LinearOperator, eigsh
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

class DoNothingCallback(BaseCallback):
    def __init__(self, model):
        super().__init__()
        self.init_callback(model)
    def _on_step(self):
        return True

def clip_norm_(values, max_norm):
    mag = 0
    length = 0
    for g in values:
        mag += (g*g).sum()
        length += np.prod(g.shape)
    norm = mag / np.sqrt(length)
    if norm > max_norm:
        clip_v = max_norm / norm
        for g in values:
            g.data *= clip_v

def gradtensor_to_npvec(net, include_bn=True):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])


def npvec_to_tensorlist(vec, params, device):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).to(device).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


class ExtA2C(A2C):
    def calulate_grad_from_buffer(self, rollout_data):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        # TODO: avoid second computation of everything because of the gradient
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        # Normalize advantage (not present in the original implementation)
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        params = self.policy.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        # TODO: check this--grad clipping!!!
        clip_norm_(grad_f, self.max_grad_norm)

        return grad_f


    def calculate_hesh_vec_prod(self, vec, num_samples):
        '''
        stores hessian vector dot product in params.grad
        '''
        self.policy.zero_grad()
        max_batch_size = 2048
        # start = 0
        for rollout_data in self.rollout_buffer.get(batch_size=max_batch_size):
            #batch_size = min(num_samples-start, max_batch_size)
            grad = self.calulate_grad_from_buffer(rollout_data)
            loss = 0
            for g, p in zip(grad, vec):
                loss += (g*p).sum()
            #accumulates grad inside sb3_algo.policy.parameters().grad
            loss.backward()

    def calculate_hesh_eigenvalues(self, num_samples, tol):
        rollout_steps = num_samples//self.n_envs
        old_buffer = self.rollout_buffer
        self.rollout_buffer = RolloutBuffer(
            num_samples,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        cb = DoNothingCallback(self)
        self.collect_rollouts(self.env, cb, self.rollout_buffer, n_rollout_steps=rollout_steps)

        self.dot_prod_calcs = 0

        def hess_vec_prod(vec):
            self.dot_prod_calcs += 1
            vec = npvec_to_tensorlist(vec, self.policy.parameters(), self.device)
            self.calculate_hesh_vec_prod(vec, num_samples)
            return gradtensor_to_npvec(self.policy)


        N = sum(np.prod(param.shape) for param in self.policy.parameters())
        A = LinearOperator((N, N), matvec=hess_vec_prod)
        eigvals, eigvecs = eigsh(A, k=1, tol=tol)
        maxeig = eigvals[0]
        print(f"max eignvalue = {maxeig}")
        print(eigvecs[0])
        # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
        # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
        shift = maxeig*.51
        def shifted_hess_vec_prod(vec):
            return hess_vec_prod(vec) - shift*vec

        A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
        eigvals, eigvecs = eigsh(A, k=1, tol=tol)
        eigvals = eigvals + shift
        mineig = eigvals[0]
        print(f"min eignvalue = {mineig}")

        assert maxeig >= 0 or mineig < 0, "something weird is going on but this case is handled by the loss landscapes paper, so duplicating that here"
        # cleanup
        self.rollout_buffer = old_buffer

        print("number of evaluations required: ", self.dot_prod_calcs)

        return maxeig, mineig
