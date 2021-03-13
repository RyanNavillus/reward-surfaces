import subprocess
import sys
import time
import multiprocessing
import tempfile
import os
import argparse

def modify_job(job, is_eval):
    job = job.strip()
    new_job = "workon main_env && " + job
    if is_eval:
        _, _, _, train_dir, checkpoint, out_dir, _ = job.split()
        new_job = f'execute_remote --copy-forward {os.path.join(train_dir,checkpoint)} {out_dir} --copy-backwards {os.path.join(out_dir,"results")} --verbose "{new_job}" '
    return  new_job + "\n"

def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('jobs_fname', type=str)
    # parser.add_argument('results_dir', type=str)
    parser.add_argument('--machines', nargs='*', help='machine id', required=True)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--is-eval', action="store_true", help="is evaluation job")
    parser.add_argument('--dry-run', action="store_true", help="kabuki dry run")

    args = parser.parse_args()

    job_list = open(args.jobs_fname).readlines()
    jobs_folder = os.path.dirname(args.jobs_fname)
    # results_dir = args.results_dir
    device = args.device
    machine_names = args.machines

    new_job_list = [modify_job(jobl, args.is_eval) for jobl in job_list]
    print(new_job_list[0])
    with tempfile.NamedTemporaryFile(suffix=".sh") as shfile:
        shfile.write("".join(new_job_list).encode("utf-8"))
        shfile.flush()
        if device == "cuda":
            requirements_args = " --no-reserve-gpu --gpu-memory-required=1000 --gpu-utilization=0.8 --memory-required=12000 --num-cpus=1 "
        else:
            requirements_args = " --no-gpu-required --memory-required=2000 --num-cpus=1 "

        dry_run_args = "" if not args.dry_run else "--dry-run"
        exec_command = f"execute_batch --copy-forward {jobs_folder} --copy-backwards . --machines {' '.join(machine_names)} {requirements_args} {shfile.name} {'--kabuki-commands' if args.is_eval else ''} {dry_run_args} "
        print(exec_command)
        subprocess.run(exec_command,shell=True)

if __name__ == "__main__":
    main()
