import subprocess
import argparse
from reward_surfaces.runners.run_jobs_slurm import run_job_list


def main():
    parser = argparse.ArgumentParser(description='run every command in batch script as seperate process on a cluster using SLURM')
    parser.add_argument('jobs_fname', type=str)
    parser.add_argument('--max-job-time', type=int, default=30, help="maximum time for single job in minutes")
    parser.add_argument('--max-memory', type=int, default=800, help="maximum memory of program in megabytes")
    parser.add_argument('--wait', action="store_true", default=False, help="waits for job to finish before completing script")

    args = parser.parse_args()
    run_job_list(args.jobs_fname, max_job_time=args.max_job_time, max_memory=args.max_memory, wait=args.wait)


if __name__ == "__main__":
    main()
