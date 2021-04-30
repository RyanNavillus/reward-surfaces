import subprocess
for i in range(100):
    p_str = str(i*10000).zfill(7)
    wait = "--wait"
    cmd = f"python scripts/run_jobs_slurm.py generated_dirs/gif_hopper/{p_str}/jobs.sh --max-job-time=60 {wait}"
    print(cmd)
    subprocess.run(cmd.split())
    subprocess.run("rm slurm-*",shell=True)
