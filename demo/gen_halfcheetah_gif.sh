python scripts/train_agent.py ./half_cheetah_checkpoints 1000000 SB3_ON HalfCheetahPyBulletEnv-v0 cpu '{"ALGO": "PPO", "num_envs": 1, "n_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "learning_rate": 0.0003, "batch_size": 64}' --save_freq=10000
python scripts/generate_gif_jobs.py --device=cpu --num-episodes=200 --seed=42 --magnitude=0.8 --grid-size=31 ./half_cheetah_checkpoints ./half_cheetah_gif_results
# consider replacing with scripts/run_jobs_slurm if you are running on a cluster.
python scripts/run_jobs_multiproc.py ./half_cheetah_gif_results/all_jobs.sh
python scripts/plot_gif.py ./half_cheetah_gif_results --outname=half_cheetah_ppo_training.gif
# resulting gif will be stored in half_cheetah_ppo_training.gif
