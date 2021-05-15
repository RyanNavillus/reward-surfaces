### Install

```
pip install git+https://github.com/benblack769/reward-surfaces.git
```


### Code structure:

reward_surfaces
* agents
  * Rainbow
  * SB3
  * Make_agent.py
* algorithms
  * compute_results.py
  * evalulte_policy_hesh.py
  * evalute_est_hesh.py
* runners
  * multiproc.py
  * kabuki.py
  * slurm.py
* plotting
  * plot_grid.py
  * plot_traj.py
* utils
  * vector.py

scripts:
* train_agent.py
* run_jobs.py
* plot_traj.py
* generate_plane_jobs.py
* generate_eval_jobs.py
* job_results_to_csv.py
* plot_grid.py

### Tests

to run tests, you may have to install pybulletgym

```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```
