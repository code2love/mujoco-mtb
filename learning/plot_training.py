import argparse
from pathlib import Path

from .learner import DEFAULT_LOG_DIR, DEFAULT_RESULTS_DIR, PPOLearner


def parse_args():
	parser = argparse.ArgumentParser(
		description="Plot episode rewards and lengths for a saved PPO training run."
	)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=DEFAULT_RESULTS_DIR,
		help="Root directory containing timestamped result folders",
	)
	parser.add_argument(
		"--result",
		help="Timestamp, run folder, or model.zip path (default: most recent run)",
	)
	parser.add_argument(
		"--log-dir",
		type=Path,
		default=DEFAULT_LOG_DIR,
		help="Root directory containing per-run monitor logs",
	)
	parser.add_argument(
		"--window",
		type=int,
		default=100,
		help="Rolling-average window in completed episodes (default: 100)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Optional path to save the plot instead of only displaying it",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Do not open an interactive plot window",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	learner = PPOLearner(
		results_dir=args.results_dir,
		log_dir=args.log_dir,
	)
	try:
		learner.load(args.result)
		learner.plot_training(
			window=args.window,
			output=args.output,
			show=not args.no_show,
		)
	finally:
		learner.close()


if __name__ == "__main__":
	main()
