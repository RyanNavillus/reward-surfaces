import subprocess
import sys
import time
import multiprocessing
import argparse
from tqdm.notebook import tqdm


def run_job_list(jobs_fname, num_cpus=None,disable_warnings=False):
    job_list = open(jobs_fname).readlines()
    job_idx = 0
    job_iterator = iter(tqdm(range(len(job_list))))
    num_procs = multiprocessing.cpu_count() if num_cpus is None else num_cpus
    proc_list = [None]*num_procs
    while job_idx != len(job_list) or any(proc_list):
        for i,proc in enumerate(proc_list):
            if proc is not None and proc.poll() is not None:
                proc_list[i] = None
            if proc_list[i] is None and job_idx < len(job_list):
                job = job_list[job_idx]
                try:
                    if disable_warnings:
                        proc_list[i] = subprocess.Popen(job.strip(), shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    else:
                        proc_list[i] = subprocess.Popen(job.strip(), shell=True)
                    # print("started: ",job)
                except IndexError:
                    print("job did not start:", job)
                    pass

                job_idx += 1
                next(job_iterator)
        time.sleep(0.1)
