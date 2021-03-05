import subprocess
import argparse
import tempfile

slurm_template = '''#!/bin/bash

#SBATCH --job-name=surfaces
#SBATCH --array=1-{NUM_JOBS}

#SBATCH --time={MAX_MINUTES}

# default qos should do it
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS_PER_JOB}

# 500MB memory per core
# this is a hard limit
#SBATCH --mem-per-cpu={MEMORY_MAX}MB

# you may not place bash commands before the last SBATCH directive
source ~/miniconda3/bin/deactivate
source ~/miniconda3/bin/deactivate
CMD=$(sed -n ${{SLURM_ARRAY_TASK_ID}}p ${{1}} )
source $HOME/init.sh
eval $CMD
'''

def run_job_list(jobs_fname, max_job_time=30, max_memory=800, max_num_jobs=300, wait=True, cpus_per_job=1):
    job_list = open(jobs_fname).readlines()
    num_jobs = len(job_list)
    slurm_batch_script = slurm_template.format(
        NUM_JOBS=num_jobs,
        MAX_MINUTES=max_job_time,
        MEMORY_MAX=max_memory,
        CPUS_PER_JOB=cpus_per_job,
    )
    with tempfile.NamedTemporaryFile(suffix=".sh", dir=".") as file:
        file.write(slurm_batch_script.encode("utf-8"))
        file.flush()
        args = ["sbatch"]
        if wait:
            args += ["--wait"]
        args += [file.name, jobs_fname]
        print("running: " + " ".join(args))
        subprocess.check_call(args)
