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
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from ..utils import clip_norm_


class DoNothingCallback(BaseCallback):
    def __init__(self, model):
        super().__init__()
        self.init_callback(model)
    def _on_step(self):
        return True


class HeshCalcOnlineMixin:
    def eval_log_prob(self, obs, act):
        '''
        returns action, logprob
        logprob should be differentialble to allow for backprop
        '''
        val, log_prob, entropy = self.policy.evaluate_actions(obs, act)
        return log_prob

    def calc_loss_and_grad(self, num_samples):
        max_batch_size = 1024
        tot_loss = 0
        tot_size = 0
        tot_grad = [torch.zeros_like(p) for p in self.parameters()]
        for rollout_data in self.generate_samples(batch_size=max_batch_size, max_samples=num_samples):
            #batch_size = min(num_samples-start, max_batch_size)
            batch_loss, batch_grad = self.calulate_grad_from_buffer(rollout_data)
            tot_loss += batch_loss.item()
            for t, g in zip(tot_grad, batch_grad):
                t.data += g.detach()
            tot_size += len(rollout_data.observations)
        print("tot eval size:",tot_size)
        tot_loss /= tot_size
        for g in tot_grad:
            g.data /= tot_size
        return tot_loss, tot_grad


    def calculate_hesh_vec_prod(self, vec, num_samples):
        '''
        stores hessian vector dot product in params.grad
        '''
        self.policy.zero_grad()
        max_batch_size = 2048
        # start = 0
        for rollout_data in self.generate_samples(batch_size=max_batch_size, max_samples=num_samples):
            #batch_size = min(num_samples-start, max_batch_size)
            _, grad = self.calulate_grad_from_buffer(rollout_data)
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
            (num_samples+self.n_envs-1)//self.n_envs,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.env.reset()
        cb = DoNothingCallback(self)
        self.collect_rollouts(self.env, cb, self.rollout_buffer, n_rollout_steps=rollout_steps)
        evaluation_data = list(zip(self.rollout_buffer.rewards, self.rollout_buffer.dones, self.rollout_buffer.values))

    def cleanup_buffer(self):
        self.rollout_buffer = self._old_buffer


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
        self.env.reset()
        train_freq = TrainFreq(num_samples, TrainFrequencyUnit("step"))
        self.collect_rollouts(
            self.env,
            train_freq=train_freq,
            action_noise=self.action_noise,
            callback=callback,
            learning_starts=0,
            replay_buffer=self.replay_buffer,
            log_interval=10,
        )
        evaluation_data = list(zip(self.replay_buffer.rewards, self.replay_buffer.dones, np.zeros_like(self.replay_buffer.rewards)))

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

        loss = policy_loss + self.ent_coef * entropy_loss + 0*self.vf_coef * value_loss
        params = self.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        # TODO: check this--grad clipping!!!
        clip_norm_(grad_f, self.max_grad_norm)

        return loss, grad_f


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

        loss = policy_loss + self.ent_coef * entropy_loss + 0*self.vf_coef * value_loss
        params = self.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        # TODO: check this--grad clipping!!!
        clip_norm_(grad_f, self.max_grad_norm)

        return loss, grad_f


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

        loss = actor_loss + 0*critic_loss

        params = self.parameters()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        # Optimization step

        return loss,grad_f
