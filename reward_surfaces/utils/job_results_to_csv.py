import os
import json
import argparse
from pathlib import Path

def job_results_to_csv(job_dir):
    job_dir = Path(job_dir)
    results_dir = job_dir / 'results'

    results_fnames = list(os.listdir(results_dir))

    keys = list(json.load(open(results_dir / results_fnames[0])).keys())
    num_ids = len(results_fnames[0][:-5].split(","))

    header = keys + [f"dim{i}" for i in range(num_ids)]
    #keys += [f"_id{i}" for i in range(num_ids)]
    csv_rows = [",".join(header)]
    for fname in results_fnames:
        entries = json.load(open(results_dir / fname))
        row_data = [str(entries[k]) for k in keys]
        row_data += fname[:-5].split(",")

        csv_rows.append(",".join(row_data))

    with open(job_dir / "results.csv", 'w') as file:
        file.write("\n".join(csv_rows) + "\n")
