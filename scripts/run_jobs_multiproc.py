import argparse
from reward_surfaces.runners.run_jobs_multiproc import run_job_list

def main():
    parser = argparse.ArgumentParser(description='run ever job in batch script on a seperate process using python multiprocessing')
    parser.add_argument('jobs_fname', type=str)
    parser.add_argument('--num-cpus', type=int, default=None, help="Maximum number of processes to run in parallel. Defaults to number of cpus on machine")

    args = parser.parse_args()

    run_job_list(args.jobs_fname, args.num_cpus)


if __name__ == "__main__":
    main()
