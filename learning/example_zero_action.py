import matplotlib.pyplot as plt
import numpy as np

from learning import MountainBikeEnv


def main():
	env = MountainBikeEnv(render_mode="human", task="obstacle")
	rewards = []

	try:
		observation, info = env.reset()
		action = np.zeros(env.action_space.shape, dtype=np.float32)
		terminated = False
		truncated = False

		while not (terminated or truncated):
			observation, reward, terminated, truncated, info = env.step(action)
			rewards.append(reward) 

		print(
			f"Episode ended at x={info['x']:.2f} m after "
			f"{info['episode_time']:.2f} s; total reward={sum(rewards):.2f}"
		)
	finally:
		env.close()

	times = np.arange(1, len(rewards) + 1) * env.environment_timestep
	plt.plot(times, rewards)
	plt.xlabel("Episode time [s]")
	plt.ylabel("Reward per environment step")
	plt.title("Zero-action MTB episode")
	plt.grid(True)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()