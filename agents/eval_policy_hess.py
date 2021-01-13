import time
from .evaluate import mean, generate_data
from .evaluate_est_hesh import npvec_to_tensorlist
import numpy as np
import torch
import torch as th
from torch.nn import functional as F
from gym import spaces
from scipy.sparse.linalg import LinearOperator, eigsh


def gen_advantage_est_episode(rews, vals, decay, gae_lambda=1.):
    last_value = (1/decay)*(vals[-1]-rews[-1])# estiamte of next value

    advantages = [0]*len(rews)
    last_gae_lam = 0
    buf_size = len(rews)
    for step in reversed(range(buf_size)):
        if step == buf_size - 1:
            next_non_terminal = 0.
            next_values = last_value
        else:
            next_non_terminal = 1.
            next_values = vals[step + 1]
        delta = rews[step] + decay * next_values * next_non_terminal - vals[step]
        last_gae_lam = delta + decay * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    return advantages


def gen_advantage_est(rewards, values, decay, gae_lambda=1.):
    return [gen_advantage_est_episode(rew,val,decay,gae_lambda) for rew,val in zip(rewards,values)]


# def split_data(datas):
#     episode_datas = []
#     ep_data = []
#     for rew,done,value in datas:
#         ep_data.append((rew,value))
#         if done:
#             episode_datas.append(ep_data)
#             ep_data = []
#     return episode_datas


def mean_baseline_est(rewards):
    baseline = mean([sum(rew) for rew in rewards])
    return [sum(rew)-baseline for rew in rewards]


def decayed_baselined_values(rewards, decay):
    values = []
    for rews in rewards:
        vals = [0]*len(rews)
        vals[-1] = rews[-1]
        for i in reversed(range(len(vals)-1)):
            vals[i] = rews[i] + vals[i+1]*decay
        values.append(vals)

    baseline_val = mean([mean(vals) for vals in values])

    return [[val-baseline_val for val in vals] for vals in values]


def gather_policy_hess_data(evaluator, num_episodes, num_steps, gamma, returns_method='baselined_vals', gae_lambda=1.0):
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_value_ests = []

    ep_rews = []
    ep_values = []
    ep_states = []
    ep_actions = []
    tot_steps = 0
    start_t = time.time()
    done = False
    while not done or (len(episode_rewards) < num_episodes and tot_steps < num_steps):
        rew, done, value, state, act = evaluator._next_state_act()#, deterministic=True)
        ep_states.append(state)
        ep_actions.append(act)
        ep_rews.append(rew)
        ep_values.append(value)
        tot_steps += 1
        if done:
            episode_states.append(ep_states)
            episode_actions.append(ep_actions)
            episode_rewards.append(ep_rews)
            episode_value_ests.append((ep_values))

            ep_rews = []
            ep_values = []
            ep_states = []
            ep_actions = []
            end_t = time.time()
            print("done!", (end_t - start_t)/len(episode_rewards))

    # returns = mean_baseline_est(episode_rewards)
    # if returns_method == 'baselined_vals':
    #returns = decayed_baselined_values(episode_rewards, gamma)
    # elif returns_method == 'gen_advantage':
    returns = gen_advantage_est(episode_rewards, episode_value_ests, gamma, gae_lambda)
    # else:
    #     raise ValueError("bad value for `returns_method`")


    single_dim_grad = None
    #
    # all_states = sum(episode_states,[])
    # all_returns = sum(returns,[])
    # all_actions = sum(episode_actions,[])
    # print(len(all_returns))
    # print(len(sum(episode_rewards,[])))
    # print(len(sum(episode_states,[])))
    # exit(0)

    return episode_states, returns, episode_actions

def get_used_params(algorithm, states, actions):
    params = algorithm.parameters()
    out = torch.sum(algorithm.eval_log_prob(states,actions))
    grads = torch.autograd.grad(out, inputs=params, create_graph=True, allow_unused=True)
    new_params = [p for g,p in zip(grads,params) if g is not None]
    # grads = torch.autograd.grad(out, inputs=new_params, create_graph=True, allow_unused=True)
    # print(grads)
    return new_params

def accumulate(accumulator, data):
    assert len(accumulator) == len(data)
    for a,d in zip(accumulator,data):
        a.data += d

def compute_vec_hesh_prod(algorithm, params, all_states, all_returns, all_actions, vec, batch_size = 1):
    device = params[0].device
    accum = [p*0 for p in params]
    # print(len(all_states))
    # print(len(all_returns))
    assert len(all_states) == len(all_actions)
    assert len(all_states) == len(all_returns)
    for eps in range(len(all_states)):
        # print("vec prod computed")
        grad_accum = [p*0 for p in params]
        grad_mul_ret_accum = [p*0 for p in params]
        hesh_prod_accum = [p*0 for p in params]
        eps_states = all_states[eps]
        eps_returns = all_returns[eps]
        eps_act = all_actions[eps]
        assert len(eps_act) == len(eps_states)
        assert len(eps_act) == len(eps_returns)
        for idx in range(0, len(eps_act), batch_size):
            eps_batch_size = min(batch_size, len(eps_act) - idx)
            batch_states = torch.squeeze(torch.tensor(eps_states[idx:idx + eps_batch_size],device=device),dim=1)
            batch_actions = torch.squeeze(torch.tensor(eps_act[idx:idx + eps_batch_size],device=device),dim=1)
            batch_returns = torch.tensor(eps_returns[idx:idx + eps_batch_size],device=device).float()

            logprob = torch.sum(algorithm.eval_log_prob(batch_states,batch_actions))
            grads = torch.autograd.grad(outputs=logprob, inputs=tuple(params), create_graph=True)

            logprob_mul_return = torch.dot(algorithm.eval_log_prob(batch_states,batch_actions), batch_returns)
            grad_mul_ret = torch.autograd.grad(outputs=logprob_mul_return, inputs=tuple(params), create_graph=True)
            assert len(vec) == len(grad_mul_ret)
            assert len(vec) == len(grads)
            g_mr_dot_v = sum([torch.dot(g_mr.view(-1),v.view(-1)) for g_mr,v in zip(grad_mul_ret, vec)], torch.zeros(1,device=device))

            hesh_prods = torch.autograd.grad(g_mr_dot_v, inputs=params, create_graph=True)
            assert len(hesh_prods) == len(vec)
            accumulate(grad_mul_ret_accum,grad_mul_ret)
            accumulate(grad_accum,grads)
            accumulate(hesh_prod_accum,hesh_prods)

        grad_vec_prod = sum([torch.dot(g_acc.view(-1),v.view(-1)) for g_acc,v in zip(grad_accum, vec)], torch.zeros(1,device=device))
        t1s = [g_mr_acc * grad_vec_prod for g_mr_acc in grad_mul_ret_accum]
        t2s = hesh_prod_accum
        assert len(accum) == len(t1s) == len(t2s)
        for acc,t1,t2 in zip(accum, t1s, t2s):
            acc.data += (t1 + t2)

    return accum


def gradtensor_to_npvec(params, include_bn=True):
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.data.cpu().numpy().ravel() for p in params if filter(p)])


def calculate_true_hesh_eigenvalues(algorithm, all_states, all_returns, all_actions, tol, device):
    algorithm.dot_prod_calcs = 0
    device = algorithm.parameters()[0].device

    params = get_used_params(algorithm, torch.tensor(all_states[0][0:2],device=device),torch.tensor(all_actions[0][0:2],device=device))

    def hess_vec_prod(vec):
        algorithm.dot_prod_calcs += 1
        vec = npvec_to_tensorlist(vec, params, device)
        accum = compute_vec_hesh_prod(algorithm, params, all_states, all_returns, all_actions, vec)
        return gradtensor_to_npvec(accum)


    N = sum(np.prod(param.shape) for param in params)
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

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig
    else:
        assert maxeig >= 0 or mineig < 0, "something weird is going on that loss landscapes paper does not handle"

    print("number of evaluations required: ", algorithm.dot_prod_calcs)

    return float(maxeig), float(mineig)


#
