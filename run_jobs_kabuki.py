import subprocess
import sys
import time
import multiprocessing
import tempfile
import os

def main():
    assert len(sys.argv) >= 4, "needs three arguments, the folder with jobs, device, followed by machine names (possible multiple)"
    fname = (sys.argv[1])
    job_list = open(os.path.join(fname,"jobs.sh")).readlines()
    results_dir = os.path.join(fname,"results")
    device = sys.argv[2]
    machine_names = sys.argv[3:]
    new_job_list = ["workon main_env && "+jobl for jobl in job_list]

    with tempfile.NamedTemporaryFile(suffix=".sh") as shfile:
        shfile.write("".join(new_job_list).encode("utf-8"))
        shfile.flush()
        if device == "cuda":
            requirements_args = " --no-reserve-gpu --gpu-memory-required=1000 --gpu-utilization=0.25 --memory-required=2000 --num-cpus=1 "
        else:
            requirements_args = " --no-gpu-required --memory-required=1000 --num-cpus=1 "

        exec_command = f"execute_batch --copy-forward agents '*.py' {fname} --copy-backwards {results_dir} --machines {' '.join(machine_names)} {requirements_args} {shfile.name}"
        print(exec_command)
        subprocess.run(exec_command,shell=True)

if __name__ == "__main__":
    main()
