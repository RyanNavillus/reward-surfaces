import subprocess
import sys
import time
import multiprocessing
import tempfile
import os

def modify_job(job, is_eval):
    new_job = "workon main_env && " + job
    if is_eval:
        _, _, _, train_dir, checkpoint, out_dir = job.split()
        

def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('jobs_fname', type=str)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('--machines', nargs='*', help='machine id', required=True)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--is-eval', action="store_true", help="is evaluation job")

    args = parser.parse_args()

    job_list = open(args.jobs_fname).readlines()
    results_dir = args.results_dir
    device = args.device
    machine_names = args.machines

    new_job_list = ["workon main_env && "+jobl for jobl in job_list]

    if args.is_eval:
        new_job_list = [jobl for jobl in new_job_list]

    with tempfile.NamedTemporaryFile(suffix=".sh") as shfile:
        shfile.write("".join(new_job_list).encode("utf-8"))
        shfile.flush()
        if device == "cuda":
            requirements_args = " --no-reserve-gpu --gpu-memory-required=1000 --gpu-utilization=0.8 --memory-required=12000 --num-cpus=1 "
        else:
            requirements_args = " --no-gpu-required --memory-required=1000 --num-cpus=1 "

        exec_command = f"execute_batch --copy-forward agents '*.py' {fname} --copy-backwards {results_dir} --machines {' '.join(machine_names)} {requirements_args} {shfile.name}"
        print(exec_command)
        subprocess.run(exec_command,shell=True)

if __name__ == "__main__":
    main()
