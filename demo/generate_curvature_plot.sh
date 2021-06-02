python scripts/train_agent.py ./hopper_checkpoints 1000000 SB3_ON HopperPyBulletEnv-v0 cpu '{"ALGO": "PPO", "num_envs": 1, "n_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "learning_rate": 0.0003, "batch_size": 64}' --save_freq=10000
python scripts/generate_eval_jobs.py --calc-hesh --num-steps=1000000 ./hopper_checkpoints/ ./hopper_eig_vecs/
python scripts/generate_plane_jobs.py --dir1=./hopper_eig_vecs/results/0040000/mineigvec.npz --dir2=./hopper_eig_vecs/results/0040000/mineigvec.npz --grid-size=31 --magnitude=1.0 --num-steps=200000   ./hopper_checkpoints/ ./hopper_eig_vecs_plane/
python scripts/run_jobs_multiproc.py ./hopper_eig_vecs_plane/jobs.sh
python scripts/job_results_to_csv.py ./hopper_eig_vecs_plane/
python scripts/plot_plane.py ./hopper_eig_vecs_plane/results.csv --outname=curvature_plot.png
