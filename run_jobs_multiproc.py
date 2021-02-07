import subprocess
import sys
import time
import multiprocessing
import argparse

def run_job_list(jobs_fname, num_cpus=None):
    job_list = open(jobs_fname).readlines()
    job_idx = 0
    num_procs = multiprocessing.cpu_count() if num_cpus is None else num_cpus
    proc_list = [None]*num_procs
    while job_idx != len(job_list) or any(proc_list):
        for i,proc in enumerate(proc_list):
            if proc is not None and proc.poll() is not None:
                proc_list[i] = None
            if proc_list[i] is None and job_idx < len(job_list):
                job = job_list[job_idx]
                try:
                    proc_list[i] = subprocess.Popen(job.strip(), shell=True)
                    print("started: ",job)
                except IndexError:
                    print("job did not start:", job)
                    pass

                job_idx += 1
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='run ever job in batch script on a seperate process using python multiprocessing')
    parser.add_argument('jobs_fname', type=str)
    parser.add_argument('--num-cpus', type=int, default=None, help="Maximum number of processes to run in parallel. Defaults to number of cpus on machine")

    args = parser.parse_args()

    run_job_list(args.jobs_fname, args.num_cpus)


if __name__ == "__main__":
    main()
