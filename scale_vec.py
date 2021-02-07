import numpy as np
import argparse
from surface_utils import scale_vec

def main():
    parser = argparse.ArgumentParser(description='plot csv data via matplotlib')
    parser.add_argument('in_vec', type=str, help="source .npz file")
    parser.add_argument('out_vec', type=str, help="dest .npz file")
    parser.add_argument('scale_val', type=float)

    args = parser.parse_args()
    vec = readz(args.in_vec)
    vec = [v*args.scale_val for v in vec]
    np.savez(args.out_vec, *vec)


if __name__ == "__main__":
    main()
