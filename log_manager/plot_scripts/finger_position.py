#!/usr/bin/env python3
"""Simply script to quickly plot data from a log file."""

import argparse
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=str)
    parser.add_argument("--base", type=str, default="0")
    args = parser.parse_args()

    data = pandas.read_csv(
        args.filename, delim_whitespace=True, header=0, low_memory=False
    )

    # convert to int if indices are given instead of names
    base = int(args.base) if args.base.isdigit() else args.base
    columns = ['observation_position', 'desired_action_position']
    outdir = os.path.dirname(args.filename)

    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(18, 18))
    for i, ax in enumerate(axes):
        inds = range(3*i, 3*(i+1))
        cols = [c + f'_{ind}' for c in columns for ind in inds]
        data.plot(x=base, y=cols, ax=ax)
        ax.set_title(f"Finger {i}")
        ax.set_ylabel("Radians")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(outdir, "plot_finger_positions.pdf"))


if __name__ == "__main__":
    main()
