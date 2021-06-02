mkdir train eval_line
python scripts/train_agent.py train/inv_double_pendulum 1000000 SB3_OFF InvertedDoublePendulumPyBulletEnv-v0 cuda ' {"ALGO": "SAC", "policy_kwargs": {"net_arch": [256, 256]}}' --save_freq=10000
python scripts/eval_line_segment.py train/inv_double_pendulum/ None eval_line/inv_double_pendulum/ --num-episodes=50 --random-dir-seed=42 --device=cpu --length=20 --max-magnitude=0.4 --scale-dir
python scripts/run_jobs_multiproc.py eval_line/inv_double_pendulum/jobs.sh
python scripts/job_results_to_csv.py eval_line/inv_double_pendulum/
python scripts/plot_eval_line_segement.py eval_line/inv_double_pendulum/results.csv inv_double_pendulum_lines.png
