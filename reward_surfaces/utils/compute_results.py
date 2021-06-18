import os
import json
import numpy as np
from reward_surfaces.algorithms import (evaluate,
                                        gather_policy_hess_data,
                                        calculate_true_hesh_eigenvalues,
                                        compute_policy_gradient,
                                        calculate_policy_ests)


def save_results(agent, info, out_dir, results, job_name):
    if info['est_hesh']:
        print(f"estimating hesh with {info['num_steps']} steps")
        assert info['num_episodes'] > 100000000, "hesh calculation only takes in steps, not episodes"
        results = agent.calculate_eigenvalues(info['num_steps'])

    if info.get('est_grad', False):
        print(f"estimating est grad with {info['num_steps']} steps")
        assert info['num_episodes'] > 100000000, "calculation only takes in steps, not episodes"
        action_evalutor = agent.action_evalutor()
        loss, grad = calculate_policy_ests(action_evalutor, info['num_steps'])
        vec_folder = out_dir/f"results/{job_name}"
        os.makedirs(vec_folder, exist_ok=True)
        np.savez(vec_folder / "est_grad.npz", *grad)
        results['est_loss'] = loss

    if info['calc_hesh'] or info['calc_grad']:
        print(f"computing rollout with {info['num_steps']} steps, {info['num_episodes']} episodes")
        evaluator = agent.evaluator()
        action_evalutor = agent.action_evalutor()
        all_states, all_returns, all_actions = gather_policy_hess_data(evaluator,
                                                                       info['num_episodes'],
                                                                       info['num_steps'],
                                                                       action_evalutor.gamma,
                                                                       "UNUSED",
                                                                       gae_lambda=1.0)
        vec_folder = out_dir/f"results/{job_name}"
        os.makedirs(vec_folder, exist_ok=True)

    if info['calc_grad']:
        policy_grad, _ = compute_policy_gradient(action_evalutor,
                                                 all_states,
                                                 all_returns,
                                                 all_actions,
                                                 action_evalutor.device)
        np.savez(vec_folder / "grad.npz", *policy_grad)
        np.savez(vec_folder / "grad_mag.npz", *policy_grad)

    if info['calc_hesh']:
        print("estimating hesh")
        maxeig, mineig, maxeigvec, mineigvec = calculate_true_hesh_eigenvalues(action_evalutor,
                                                                               all_states,
                                                                               all_returns,
                                                                               all_actions,
                                                                               tol=0.01,
                                                                               device=action_evalutor.device)
        results['mineig'] = mineig
        results['maxeig'] = maxeig
        results['ratio'] = mineig / max(-0.001*mineig, maxeig)
        vec_folder = out_dir/f"results/{job_name}"
        np.savez(vec_folder/"maxeigvec.npz", *maxeigvec)
        np.savez(vec_folder/"mineigvec.npz", *mineigvec)

    if not info['calc_hesh'] and not info['est_hesh']:
        evaluator = agent.evaluator()
        eval_results = evaluate(evaluator, info['num_episodes'], info['num_steps'])
        results.update(eval_results)

    print("dumping results")
    json.dump(results, open(out_dir/f"results/{job_name}.json", 'w'))
