
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--path', help='input txt path')
    parser.add_argument('--output', help='output name of txt')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ratio', default=0.25)
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    np.random.seed(args.seed)

    lines = open(args.path).readlines()

    keep = int(args.ratio * len(lines))

    idx = np.random.permutation(range(len(lines)))[:keep]

    with open(args.output, 'w') as f:
        for i in idx:
            f.writelines(lines[i])

if __name__ == '__main__':
    main()
