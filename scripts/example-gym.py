import tqdm
import gym
import argparse
from stable_baselines3 import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envname", type=str, default="CartPole-v0")
    parser.add_argument("--num-steps", default=10_000, type=int, help="number of training timesteps")
    args = parser.parse_args()

    env = gym.make(args.envname)

    model = PPO("MlpPolicy", env, verbose=1)
    with tqdm.tqdm(total=args.num_steps) as pbar:
        def callback(local_vars, global_vars):
            pbar.update(model.num_timesteps - pbar.n)
        model.learn(total_timesteps=args.num_steps, callback=callback, reset_num_timesteps=False)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    env.close()
