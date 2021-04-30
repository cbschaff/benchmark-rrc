#!/usr/bin/env python3
"""Simply script to quickly plot data from a log file."""

import argparse
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("columns", type=str, nargs="+")
    parser.add_argument("--base", type=str, default="0")
    args = parser.parse_args()

    data = pandas.read_csv(
        args.filename, delim_whitespace=True, header=0, low_memory=False
    )

    # convert to int if indices are given instead of names
    base = int(args.base) if args.base.isdigit() else args.base
    columns = [int(c) if c.isdigit() else c for c in args.columns]

    plt.figure(figsize=(12, 8))
    data.plot(x=base, y=columns, ax=plt.gca())
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
