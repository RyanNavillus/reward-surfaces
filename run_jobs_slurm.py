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

# 500MB memory per core
# this is a hard limit
#SBATCH --mem-per-cpu={MEMORY_MAX}MB

# you may not place bash commands before the last SBATCH directive

CMD=$(sed -n ${{SLURM_ARRAY_TASK_ID}}p ${{1}} )
source $HOME/init.sh
eval $CMD
'''

def run_job_list(jobs_fname, max_job_time=30, max_memory=800, wait=True):
    job_list = open(jobs_fname).readlines()
    num_jobs = len(job_list)
    slurm_batch_script = slurm_template.format(
        NUM_JOBS=num_jobs,
        MAX_MINUTES=max_job_time,
        MEMORY_MAX=max_memory,
    )
    with tempfile.NamedTemporaryFile(suffix=".sh", dir=".") as file:
        file.write(slurm_batch_script)
        file.flush()
        args = ["sbatch"]
        if wait:
            args += ["--wait"]
        args += [file.name, jobs_fname]
        print("running: " + " ".join(args))
        subprocess.check_call(args)



def main():
    parser = argparse.ArgumentParser(description='generate jobs for plane')
    parser.add_argument('jobs_fname', type=str)
    parser.add_argument('--max-job-time', type=int, default=30, help="maximum time for single job in minutes")
    parser.add_argument('--max-memory', type=int, default=800, help="maximum memory of program in megabytes")
    parser.add_argument('--wait', action="store_true", default=False, help="waits for job to finish before completing script")

    args = parser.parse_args()
    run_job_list(args.jobs_fname, max_job_time=args.max_job_time, max_memory=args.max_memory, wait=args.wait)


if __name__ == "__main__":
    main()
