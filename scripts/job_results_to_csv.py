from reward_surfaces.utils import job_results_to_csv
import argparse


def main():
    parser = argparse.ArgumentParser(description='concatenate job results into csv')
    parser.add_argument('job_dir', type=str, help="directory with info.json and results/ as a subdirectory")

    args = parser.parse_args()

    job_results_to_csv(args.job_dir)


if __name__ == "__main__":
    main()
