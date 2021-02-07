import os
import json
import pathlib
from .eval_policy_hess import gather_policy_hess_data, calculate_true_hesh_eigenvalues, compute_policy_gradient
from .evaluate import evaluate
import numpy as np


def save_results(agent, info, out_dir, results, job_name):
    if info['est_hesh']:
        print(f"estimating hesh with {info['num_steps']} steps")
        assert info['num_episodes'] > 100000000, "hesh calculation only takes in steps, not episodes"
        results = agent.calculate_eigenvalues(info['num_steps'])

    if info['calc_hesh'] or info['calc_grad']:
        print(f"computing rollout with {info['num_steps']} steps, {info['num_episodes']} episodes")
        evaluator = agent.evaluator()
        action_evalutor = agent.action_evalutor()
        all_states, all_returns, all_actions = gather_policy_hess_data(evaluator, info['num_episodes'], info['num_steps'], action_evalutor.gamma, "UNUSED", gae_lambda=1.0)
        vec_folder = out_dir/f"results/{job_name}"
        os.makedirs(vec_folder,exist_ok=True)

    if info['calc_grad']:
        policy_grad, policy_mag = compute_policy_gradient(action_evalutor, all_states, all_returns, all_actions, action_evalutor.device)
        np.savez(vec_folder / f"grad.npz", *policy_grad)
        np.savez(vec_folder / f"grad_mag.npz", *policy_grad)

    if info['calc_hesh']:
        print(f"estimating hesh")
        maxeig, mineig, maxeigvec, mineigvec = calculate_true_hesh_eigenvalues(action_evalutor, all_states, all_returns, all_actions, tol=0.01, device=action_evalutor.device)
        results['mineig'] = mineig
        results['maxeig'] = maxeig
        results['ratio'] = mineig / max(-0.001*mineig,maxeig)
        vec_folder = out_dir/f"results/{job_name}"
        np.savez(vec_folder/"maxeigvec.npz", *maxeigvec)
        np.savez(vec_folder/"mineigvec.npz", *mineigvec)

    if not info['calc_hesh'] and not info['est_hesh'] and not info['calc_grad']:
        evaluator = agent.evaluator()
        eval_results = evaluate(evaluator, info['num_episodes'], info['num_steps'])
        results.update(eval_results)

    json.dump(results,open(out_dir/f"results/{job_name}.json",'w'))

if __name__ == "__main__":
    from .make_agent import make_agent
    agent = make_agent("SB3_ON","Pendulum-v0","cpu",{'ALGO':'PPO'})
    info = { }
    info['num_episodes'] = 10
    info['num_steps'] = 1000
    info['est_hesh'] = False
    info['calc_hesh'] = False
    folder = "compute_res_test"
    os.makedirs(folder+"/results")
    save_results(agent, info, pathlib.Path(folder), {}, "test_eval")
    info['calc_hesh'] = True
    save_results(agent, info, pathlib.Path(folder), {}, "test_calc_hesh")
