import subprocess
import time
import multiprocessing
from tqdm import tqdm


def run_job_list_list(job_list, num_cpus=None, disable_warnings=False):
    job_idx = 0
    job_iterator = iter(tqdm(range(len(job_list))))
    num_procs = multiprocessing.cpu_count() - 1 if num_cpus is None else num_cpus
    proc_list = [None]*num_procs
    while job_idx != len(job_list) or any(proc_list):
        for i, proc in enumerate(proc_list):
            if proc is not None and proc.poll() is not None:
                proc_list[i] = None
            if proc_list[i] is None and job_idx < len(job_list):
                job = job_list[job_idx]
                try:
                    if disable_warnings:
                        proc_list[i] = subprocess.Popen(job.strip(),
                                                        shell=True,
                                                        stdout=subprocess.DEVNULL,
                                                        stderr=subprocess.DEVNULL)
                    else:
                        proc_list[i] = subprocess.Popen(job.strip(), shell=True)
                    # print("started: ",job)
                except IndexError:
                    print("job did not start:", job)

                job_idx += 1
                next(job_iterator)
        time.sleep(0.1)


def run_job_list(jobs_fname, num_cpus=None, disable_warnings=False):
    job_list = open(jobs_fname).readlines()
    run_job_list_list(job_list, num_cpus=num_cpus, disable_warnings=disable_warnings)
