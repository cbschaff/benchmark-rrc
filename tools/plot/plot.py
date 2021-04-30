#!/usr/bin/env python3
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--csv', default='eval_scores.csv', help='specify a path to save a generated csv (default: eval_scores.csv)')
    parser.add_argument('--plot', default='plot.pdf', help='specify a path to save a generated plot (default: plot.pdf)')
    parser.add_argument('--exp', default='align', help='specify the type of experiments. Either align or mix-match (default: align)')
    args = parser.parse_args()

    if args.exp == 'align':
        import exp_align_obj
        generate_csv = exp_align_obj.generate_csv
        generate_plot = exp_align_obj.generate_plot
    elif args.exp == 'mix-match':
        import exp_approach_goal
        generate_csv = exp_approach_goal.generate_csv
        generate_plot = exp_approach_goal.generate_plot

    print('generating a csv file from log files...')
    generate_csv(args.log_dir, csv_file=args.csv)

    print('generating plot from the csv file...')
    generate_plot(args.csv, plot_file=args.plot)
