import argparse
from pathlib import Path

from .learner import DEFAULT_LOG_DIR, DEFAULT_RESULTS_DIR, PPOLearner


def parse_args():
	parser = argparse.ArgumentParser(
		description="Train and save a PPO policy for the MTB environment."
	)
	parser.add_argument("--total-timesteps", type=int, default=4_000_000)
	parser.add_argument("--n-steps", type=int, default=512)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--n-envs", type=int, default=8)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument(
		"--task", 
		default="manual",
		help="Learning task")
	parser.add_argument(
		"--log-dir",
		type=Path,
		default=DEFAULT_LOG_DIR,
		help="Root directory for per-run monitor logs",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_RESULTS_DIR,
		help="Root directory for timestamped model result folders",
	)
	parser.add_argument(
		"--render",
		action="store_true",
		help="Render the environment during training (requires --n-envs 1)",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Resume the last training run",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	env_kwargs = {"task": args.task}
	if args.render:
		env_kwargs["render_mode"] = "human"
	learner = PPOLearner(
		env_kwargs=env_kwargs,
		results_dir=args.output,
		log_dir=args.log_dir,
	)
	try:
		if args.resume:
			learner.load()
		learner.learn(
			total_timesteps=args.total_timesteps,
			n_steps=args.n_steps,
			batch_size=args.batch_size,
			n_envs=args.n_envs,
			seed=args.seed,
			device=args.device,
			learning_rate=args.learning_rate,
		)
		learner.save()
	finally:
		learner.close()


if __name__ == "__main__":
	main()
