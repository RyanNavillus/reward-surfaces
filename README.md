## Install

```
pip install git+https://github.com/RyanNavillus/reward-surfaces.git
```

## Usage

#### Training
To train an agent to play CartPole with PPO on GPU, saving checkpoints every 10,000 steps:
```
python3 scripts/train_agent.py "./runs/cartpole_checkpoints" SB3_ON CartPole-v1 cuda '{"ALGO": "PPO"}' --save_freq=10000
```


#### Evaluating
To evaluate the gradient direction:
```
python3 scripts/generate_eval_jobs.py --batch-grad --num-steps=1000000 "./runs/cartpole_checkpoints" "./runs/eval_grad/cartpole/"
python3 scripts/run_jobs_multiproc.py --num-cpus=8 "./runs/eval_grad/cartpole/jobs.sh"
```


#### Plotting
To plot a reward surface:
```
python3 scripts/generate_plane_jobs.py --grid-size=31 --magnitude=1.0 --num-steps=200000 "./runs/cartpole_checkpoints/best/" "./runs/cartpole_surface/"
python3 scripts/run_jobs_multiproc.py --num-cpus=8 "./runs/cartpole_surface/jobs.sh"
python3 scripts/job_results_to_csv.py "./runs/cartpole_surface/"
python3 scripts/plot_plane.py "./runs/cartpole_surface/results.csv" --outname="./runs/cartpole" --env_name="CartPole-v1"
```

 * You can use the `--keyname` option for `plot_plane.py` to select a different value to plot. For example: `--keyname="episode_stderr_rewards"` will plot the standard error of each point.
    * To see all available keys try `python3 scripts/print_keys.py`
 * You can use the `--type` option for `plot_plane.py` to select a different plot style. For example: `--type="heat"` will plot a heatmap instead of a reward surface.

To plot a gradient line search:
```
python scripts/eval_line_segment.py ./runs/cartpole_checkpoints/ ./runs/eval_grad/cartpole/results ./runs/eval_line/cartpole/ --num-episodes=100 --length=20 --max-magnitude=0.4 --scale-dir
python scripts/run_jobs_multiproc.py --num-cpus=8 ./runs/eval_line/cartpole/jobs.sh
python scripts/job_results_to_csv.py ./runs/eval_line/cartpole/
python scripts/plot_eval_line_segement.py ./runs/eval_line/cartpole/results.csv --outname="cartpole__lines.png"
```
 * This requires the gradient direction at each checkpoint to be evaluated beforehand.

For all scripts, try the `-h/--help` flag to see more options.


## Tests

to run tests, you may have to install pybulletgym

```
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```
