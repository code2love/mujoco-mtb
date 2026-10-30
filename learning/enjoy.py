import argparse
from pathlib import Path

from .learner import DEFAULT_RESULTS_DIR, PPOLearner


def parse_args():
	parser = argparse.ArgumentParser(
		description="Render an episode using a saved deterministic PPO policy."
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
	parser.add_argument("--device", default="cpu")
	return parser.parse_args()


def main():
	args = parse_args()
	learner = PPOLearner(results_dir=args.results_dir)
	try:
		learner.load(args.result, device=args.device)
		learner.enjoy()
	finally:
		learner.close()


if __name__ == "__main__":
	main()
