import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='concatenate job results into csv')
    parser.add_argument('job_dir', type=str, help="directory with info.json and results/ as a subdirectory")

    args = parser.parse_args()

    results_dir = Path(args.job_dir) / 'results'

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

    print("\n".join(csv_rows))

if __name__ == "__main__":
    main()
