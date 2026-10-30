from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
	SubprocVecEnv,
	VecEnv,
	VecMonitor,
	VecNormalize,
)

from .environment import MountainBikeEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "learning" / "models" / "ppo_mtb"
DEFAULT_LOG_DIR = PROJECT_ROOT / "learning" / "logs" / "ppo_mtb"
HIDDEN_DIMS_PI = [128, 128, 128]
HIDDEN_DIMS_VF = [512, 512, 512]
NORMALIZATION_SETTINGS = {
	"norm_obs": True,
	"norm_reward": True,
	"clip_obs": 1e6,
	"clip_reward": 1e6,
}


def _json_default(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


class PPOLearner:

	def __init__(
		self,
		env_kwargs: dict[str, Any] | None = None,
		results_dir: str | Path = DEFAULT_RESULTS_DIR,
		log_dir: str | Path = DEFAULT_LOG_DIR,
	):
		self.env_kwargs = dict(env_kwargs or {})
		self.results_dir = Path(results_dir).expanduser().resolve()
		self.log_dir = Path(log_dir).expanduser().resolve()
		self.model: PPO | None = None
		self.env: VecNormalize | None = None
		self.run_dir: Path | None = None
		self.run_id: str | None = None
		self.training_parameters: dict[str, Any] = {}
		self._training_env: VecEnv | None = None
		self._has_episode_monitor = False

	def _create_run_directory(self) -> Path:
		self.results_dir.mkdir(parents=True, exist_ok=True)
		base_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		run_id = base_id
		suffix = 1
		while True:
			run_dir = self.results_dir / run_id
			try:
				run_dir.mkdir()
				break
			except FileExistsError:
				run_id = f"{base_id}_{suffix}"
				suffix += 1

		self.run_id = run_id
		self.run_dir = run_dir
		return run_dir

	def _make_vec_env(
		self,
		n_envs: int,
		seed: int,
		monitor_dir: Path | None = None,
	) -> VecNormalize:
		if self.env is not None:
			self.env.close()
			self.env = None

		vec_env = make_vec_env(
			MountainBikeEnv,
			n_envs=n_envs,
			seed=seed,
			monitor_dir=str(monitor_dir) if monitor_dir is not None else None,
			env_kwargs=self.env_kwargs,
			vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
		)
		self.env = VecNormalize(vec_env, **NORMALIZATION_SETTINGS)
		self._training_env = self.env
		self._has_episode_monitor = monitor_dir is not None
		return self.env

	def learn(
		self,
		total_timesteps: int = 2_000_000,
		n_steps: int = 512,
		batch_size: int = 64,
		n_envs: int = 4, 
		seed: int = 0,
		device: str = "cpu",
		learning_rate: float = 3e-4,
	) -> PPO:
		"""Train PPO using the configured environment and training parameters."""
		if total_timesteps <= 0:
			raise ValueError("total_timesteps must be positive")
		if n_steps <= 0 or batch_size <= 0:
			raise ValueError("n_steps and batch_size must be positive")
		if n_envs <= 0:
			raise ValueError("n_envs must be positive")
		if n_envs > 1 and self.env_kwargs.get("render_mode") == "human":
			raise ValueError("human rendering requires n_envs=1")
		if (
			self.model is not None
			and self.env is not None
			and self.env.num_envs != n_envs
		):
			raise ValueError(
				f"The loaded model uses {self.env.num_envs} environments; "
				f"continue training with n_envs={self.env.num_envs}"
			)

		self.training_parameters = {
			"total_timesteps": total_timesteps,
			"n_steps": n_steps,
			"batch_size": batch_size,
			"n_envs": n_envs,
			"seed": seed,
			"device": device,
			"learning_rate": learning_rate,
		}
		run_dir = self._create_run_directory()
		self._write_run_config(run_dir)
		monitor_dir = self.log_dir / run_dir.name
		monitor_dir.mkdir(parents=True, exist_ok=True)
		if self.model is None or self.env is None:
			self._make_vec_env(
				n_envs=n_envs,
				seed=seed,
				monitor_dir=monitor_dir,
			)
			policy_kwargs = {
				"net_arch": {"pi": HIDDEN_DIMS_PI, "vf": HIDDEN_DIMS_VF},
				"activation_fn": nn.ELU,
			}
			self.model = PPO(
				"MlpPolicy",
				self.env,
				policy_kwargs=policy_kwargs,
				learning_rate=lambda x: x * learning_rate,
				n_steps=n_steps,
				batch_size=batch_size,
				seed=seed,
				device=device,
				verbose=1,
				tensorboard_log=None,
			)
		else:
			self.env.training = True
			if not self._has_episode_monitor:
				self._training_env = VecMonitor(
					self.env,
					filename=str(monitor_dir / "monitor.csv"),
				)
				self._has_episode_monitor = True
			self.model.set_env(self._training_env)

		model = self.model
		if model is None:
			raise RuntimeError("PPO model initialization failed")
		checkpoint_callback = CheckpointCallback(
			save_freq=int(max(1, total_timesteps / n_envs / 10)),
			save_path=str(run_dir),
			save_vecnormalize=True,
		)
		model.learn(
			total_timesteps=total_timesteps,
			callback=checkpoint_callback, 
			progress_bar=False,
		)
		return model

	def _write_run_config(self, run_dir: Path) -> None:
		manifest = {
			"run_id": run_dir.name,
			"env_kwargs": self.env_kwargs,
			"training_parameters": self.training_parameters,
			"normalization_settings": NORMALIZATION_SETTINGS,
			"policy": {
				"net_arch": {"vf": HIDDEN_DIMS_VF, "pi": HIDDEN_DIMS_PI},
				"activation_fn": "ELU",
			},
		}
		with (run_dir / "config.json").open("w", encoding="utf-8") as config_file:
			json.dump(manifest, config_file, indent=2, default=_json_default)
			config_file.write("\n")

	def save(self) -> Path:
		"""Save model, normalization state, and environment/training configuration."""
		if self.model is None or self.env is None:
			raise RuntimeError("No trained or loaded model is available to save")

		run_dir = self.run_dir or self._create_run_directory()
		model_path = run_dir / "model"
		normalization_path = run_dir / "vecnormalize.pkl"
		self.model.save(str(model_path))
		self.env.save(str(normalization_path))
		self._write_run_config(run_dir)
		print(f"Saved PPO model to {model_path}.zip")
		print(f"Saved normalization state to {normalization_path}")
		print(f"Saved run configuration to {run_dir / 'config.json'}")
		return run_dir

	def plot_training(
		self,
		window: int = 100,
		output: str | Path | None = None,
		show: bool = True,
	):
		"""Plot episode rewards and lengths from this run's monitor logs."""
		if window <= 0:
			raise ValueError("window must be positive")
		if self.run_dir is None:
			raise RuntimeError("No current run; train or load a result before plotting")

		monitor_dir = self.log_dir / self.run_dir.name
		log_files = sorted(monitor_dir.glob("*.monitor.csv"))
		if not log_files:
			raise FileNotFoundError(f"No monitor CSV files found in {monitor_dir}")

		episodes = []
		for path in log_files:
			episodes.extend(self._read_monitor_file(path))
		if not episodes:
			raise ValueError(f"No completed episodes found in {monitor_dir}")

		episodes.sort(key=lambda episode: episode["completed_at"])
		rewards = np.asarray([episode["reward"] for episode in episodes])
		lengths = np.asarray([episode["length"] for episode in episodes])
		episode_numbers = np.arange(1, len(episodes) + 1)
		reward_average = pd.Series(rewards).rolling(
			window=window,
			min_periods=1,
		).mean()

		figure, (reward_axis, length_axis) = plt.subplots(
			2,
			1,
			sharex=True,
			figsize=(11, 8),
		)
		reward_axis.plot(episode_numbers, rewards, alpha=0.3, label="Episode reward")
		reward_axis.plot(
			episode_numbers,
			reward_average,
			linewidth=2,
			label=f"{window}-episode rolling mean",
		)
		reward_axis.set_ylabel("Reward")
		reward_axis.set_title("PPO training progress")
		reward_axis.grid(True, alpha=0.3)
		reward_axis.legend()

		length_axis.plot(episode_numbers, lengths, color="tab:orange")
		length_axis.set_xlabel("Completed episode (all monitor files, ordered by time)")
		length_axis.set_ylabel("Episode length [steps]")
		length_axis.grid(True, alpha=0.3)
		figure.tight_layout()

		print(f"Loaded {len(episodes)} episodes from {len(log_files)} monitor files")
		if output is not None:
			output_path = Path(output).expanduser().resolve()
			output_path.parent.mkdir(parents=True, exist_ok=True)
			figure.savefig(output_path, dpi=150)
			print(f"Saved plot to {output_path}")
		if show:
			plt.show()
		return figure, (reward_axis, length_axis)

	@staticmethod
	def _read_monitor_file(path: Path) -> list[dict[str, float | int | str]]:
		with path.open(encoding="utf-8", newline="") as monitor_file:
			metadata_line = monitor_file.readline()
			if not metadata_line.startswith("#"):
				raise ValueError(f"{path} is missing Stable-Baselines3 monitor metadata")
			try:
				metadata = json.loads(metadata_line[1:])
			except json.JSONDecodeError as error:
				raise ValueError(f"{path} has invalid monitor metadata") from error

			reader = csv.DictReader(monitor_file)
			required_columns = {"r", "l", "t"}
			if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
				raise ValueError(
					f"{path} must contain monitor columns {sorted(required_columns)}"
				)

			episodes: list[dict[str, float | int | str]] = []
			for row in reader:
				try:
					episodes.append({
						"reward": float(row["r"]),
						"length": int(row["l"]),
						"completed_at": float(metadata["t_start"]) + float(row["t"]),
						"source": path.name,
					})
				except (KeyError, TypeError, ValueError) as error:
					raise ValueError(f"{path} contains an invalid episode row") from error
		return episodes

	def _resolve_run_directory(self, result: str | Path | None) -> Path:
		def is_saved_run(path: Path) -> bool:
			has_final_model = (
				(path / "model.zip").is_file()
				and (path / "vecnormalize.pkl").is_file()
			)
			has_checkpoint = bool(self._checkpoint_pairs(path))
			return (path / "config.json").is_file() and (
				has_final_model or has_checkpoint
			)

		if result is None:
			if not self.results_dir.is_dir():
				raise FileNotFoundError(f"No saved results in {self.results_dir}")
			runs = sorted(
				(path for path in self.results_dir.iterdir() if is_saved_run(path)),
				key=lambda path: path.name,
			)
			if not runs:
				raise FileNotFoundError(
					f"No runs with a complete model and normalization state in {self.results_dir}"
				)
			return runs[-1]

		selected = Path(result).expanduser()
		candidates = [selected]
		if not selected.is_absolute():
			candidates.append(self.results_dir / selected)
		if selected.suffix == ".zip":
			candidates.extend(
				path.with_suffix("") for path in tuple(candidates)
			)
		for candidate in candidates:
			if candidate.is_file() and candidate.name == "model.zip":
				candidate = candidate.parent
			if is_saved_run(candidate):
				return candidate.resolve()

		raise FileNotFoundError(
			f"Saved run {result!s} was not found or has no complete model and "
			"normalization state"
		)

	@staticmethod
	def _checkpoint_pairs(run_dir: Path) -> list[tuple[int, Path, Path]]:
		pairs = []
		for model_path in run_dir.glob("rl_model_*_steps.zip"):
			match = re.fullmatch(r"rl_model_(\d+)_steps\.zip", model_path.name)
			if match is None:
				continue
			step = int(match.group(1))
			normalization_path = run_dir / f"rl_model_vecnormalize_{step}_steps.pkl"
			if normalization_path.is_file():
				pairs.append((step, model_path, normalization_path))
		return sorted(pairs, key=lambda pair: pair[0])

	def load(
		self,
		result: str | Path | None = None,
		device: str = "cpu",
	) -> PPO:
		"""Load a run by folder, timestamp, or model path; default to the newest run."""
		run_dir = self._resolve_run_directory(result)
		model_path = run_dir / "model.zip"
		normalization_path = run_dir / "vecnormalize.pkl"
		if not model_path.is_file() or not normalization_path.is_file():
			checkpoints = self._checkpoint_pairs(run_dir)
			if not checkpoints:
				raise FileNotFoundError(
					f"No checkpoint with matching VecNormalize statistics found in {run_dir}"
				)
			step, model_path, normalization_path = checkpoints[-1]
			print(
				f"Final model not found; loading intermediate checkpoint at "
				f"{step:,} timesteps from {model_path}"
			)

		with (run_dir / "config.json").open(encoding="utf-8") as config_file:
			manifest = json.load(config_file)

		if "env_kwargs" not in manifest or "training_parameters" not in manifest:
			raise ValueError(f"{run_dir / 'config.json'} is missing required run settings")
		self.env_kwargs = dict(manifest["env_kwargs"])
		self.training_parameters = dict(manifest["training_parameters"])
		self.run_dir = run_dir
		self.run_id = run_dir.name

		n_envs = int(self.training_parameters.get("n_envs", 1))
		seed = int(self.training_parameters.get("seed", 0))
		if n_envs <= 0:
			raise ValueError(f"Invalid n_envs value in {run_dir / 'config.json'}")
		if n_envs > 1 and self.env_kwargs.get("render_mode") == "human":
			raise ValueError(
				f"Saved run {run_dir} requests human rendering with multiple environments"
			)

		if self.env is not None:
			self.env.close()
		vec_env = make_vec_env(
			MountainBikeEnv,
			n_envs=n_envs,
			seed=seed,
			env_kwargs=self.env_kwargs,
			vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
		)
		self.env = VecNormalize.load(str(normalization_path), vec_env)
		self.env.training = False
		self._training_env = self.env
		self._has_episode_monitor = False
		model = PPO.load(
			str(model_path),
			env=self.env,
			device=device,
		)
		self.model = model
		return model

	def enjoy(self, result: str | Path | None = None) -> None:
		"""Render one deterministic episode with frozen observation normalization."""
		if result is not None or self.model is None:
			self.load(result)
		if self.model is None or self.env is None:
			raise RuntimeError("Load or train a model before calling enjoy")

		env_kwargs = dict(self.env_kwargs)
		env_kwargs["render_mode"] = "human"
		env = MountainBikeEnv(**env_kwargs)
		was_training = self.env.training
		self.env.training = False
		try:
			observation, _ = env.reset()
			total_reward = 0.0
			terminated = False
			truncated = False
			steps = 0
			while not (terminated or truncated):
				# Use saved observation statistics without updating them during evaluation.
				normalized_observation = self.env.normalize_obs(observation)
				action, _ = self.model.predict(
					normalized_observation,
					deterministic=True,
				)
				observation, reward, terminated, truncated, _ = env.step(action)
				total_reward += reward
				steps += 1
			print(
				f"Episode ended after {steps} steps; total reward={total_reward:.2f}; "
				f"terminated={terminated}; truncated={truncated}"
			)
		finally:
			env.close()
			self.env.training = was_training

	def close(self) -> None:
		"""Close any owned vector environment."""
		if self.env is not None:
			self.env.close()
			self.env = None
			self._training_env = None
			self._has_episode_monitor = False
