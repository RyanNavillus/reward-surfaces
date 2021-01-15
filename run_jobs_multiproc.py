import subprocess
import sys
import time
import multiprocessing
import argparse

def main():
    assert len(sys.argv) == 2, "needs one argument, the file with jobs"
    job_list = open(sys.argv[1]).readlines()
    job_idx = 0
    num_procs = multiprocessing.cpu_count()
    proc_list = [None]*num_procs
    while job_idx != len(job_list) or any(proc_list):
        for i,proc in enumerate(proc_list):
            if proc is not None and proc.poll() is not None:
                proc_list[i] = None
            if proc_list[i] is None:
                job = job_list[job_idx]
                try:
                    proc_list[i] = subprocess.Popen(job.strip(), shell=True)
                    print("started: ",job)
                except IndexError:
                    print("job did not start:", job)
                    pass

                job_idx += 1
        time.sleep(0.2)


if __name__ == "__main__":
    main()
