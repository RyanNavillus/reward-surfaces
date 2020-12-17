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
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer
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

def gradtensor_to_npvec(params, include_bn=True):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in params if filter(p)])


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

class HeshCalcOnlineMixin:
    def eval_log_prob(self, obs, act):
        '''
        returns action, logprob
        logprob should be differentialble to allow for backprop
        '''
        val, log_prob, entropy = self.evaluate_actions(obs, act)
        return log_prob

    def calculate_hesh_vec_prod(self, vec, num_samples):
        '''
        stores hessian vector dot product in params.grad
        '''
        self.policy.zero_grad()
        max_batch_size = 2048
        # start = 0
        for rollout_data in self.generate_samples(batch_size=max_batch_size, max_samples=num_samples):
            #batch_size = min(num_samples-start, max_batch_size)
            grad = self.calulate_grad_from_buffer(rollout_data)
            loss = 0
            for g, p in zip(grad, vec):
                loss += (g*p).sum()
            #accumulates grad inside sb3_algo.parameters().grad
            loss.backward()

    def generate_samples(self, batch_size, max_samples):
        return self.rollout_buffer.get(batch_size=batch_size)

    def setup_buffer(self, num_samples):
        rollout_steps = num_samples//self.n_envs
        self._old_buffer = self.rollout_buffer
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

    def cleanup_buffer(self):
        self.rollout_buffer = self._old_buffer

    def calculate_hesh_eigenvalues(self, num_samples, tol):
        self.setup_buffer(num_samples)

        self.dot_prod_calcs = 0

        def hess_vec_prod(vec):
            self.dot_prod_calcs += 1
            vec = npvec_to_tensorlist(vec, self.parameters(), self.device)
            self.calculate_hesh_vec_prod(vec, num_samples)
            return gradtensor_to_npvec(self.parameters())


        N = sum(np.prod(param.shape) for param in self.parameters())
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

        print("number of evaluations required: ", self.dot_prod_calcs)

        self.cleanup_buffer()

        return maxeig, mineig

class HeshCalcOfflineMixin(HeshCalcOnlineMixin):
    def eval_log_prob(self, obs, act):
        raise NotImplmenetedError("eval_log_prob not implemented for all sb3 algorithms including TD3 and DDPG (in theory maybe td3 can work???)")

    def generate_samples(self, max_samples, batch_size):
        for i in range(0, max_samples+batch_size-1, batch_size):
            yield self.replay_buffer.sample(batch_size=batch_size)

    def cleanup_buffer(self):
        self.replay_buffer = self._old_buffer

    def setup_buffer(self, num_samples):
        assert self.n_envs == 1, "I don't think multiple envs works for offline policies, but you can check and make suitable updates"
        self._old_buffer = self.replay_buffer
        callback = DoNothingCallback(self)
        self.replay_buffer = ReplayBuffer(
            num_samples,
            self.observation_space,
            self.action_space,
            self.device,
        )
        self.collect_rollouts(
            self.env,
            n_episodes=num_samples,
            n_steps=num_samples,
            action_noise=self.action_noise,
            callback=callback,
            learning_starts=0,
            replay_buffer=self.replay_buffer,
            log_interval=10,
        )


class ExtA2C(A2C, HeshCalcOnlineMixin):
    def parameters(self):
        return list(self.policy.parameters())

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
        params = self.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        # TODO: check this--grad clipping!!!
        clip_norm_(grad_f, self.max_grad_norm)

        return grad_f


class ExtPPO(PPO, HeshCalcOnlineMixin):
    def parameters(self):
        return list(self.policy.parameters())

    def calulate_grad_from_buffer(self, rollout_data):
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # Re-sample the noise matrix because the log_std has changed
        # TODO: investigate why there is no issue with the gradient
        # if that line is commented (as in SAC)
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        params = self.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        # TODO: check this--grad clipping!!!
        clip_norm_(grad_f, self.max_grad_norm)

        return grad_f


class ExtSAC(SAC, HeshCalcOfflineMixin):
    def parameters(self):
        # print(self.policy.critic.state_dict().keys())
        return list(self.policy.actor.parameters()) + list(self.policy.critic.parameters())

    def eval_log_prob(self, obs, act):
        mean_actions, log_std, kwargs = self.policy.critic.get_action_dist_params(obs)
        return self.policy.critic.action_dist.actions_from_params()

    def calulate_grad_from_buffer(self, replay_data):
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        else:
            ent_coef = self.ent_coef_tensor

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the target Q value: min over all critics targets
            targets = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            target_q, _ = th.min(targets, dim=1, keepdim=True)
            # add entropy term
            target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            q_backup = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

        # Get current Q estimates for each critic network
        # using action from the replay buffer
        current_q_estimates = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

        loss = actor_loss + critic_loss

        params = self.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        return grad_f
