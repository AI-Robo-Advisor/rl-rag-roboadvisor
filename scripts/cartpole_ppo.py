import gymnasium as gym
from stable_baselines3 import PPO 

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

test_env = gym.make("CartPole-v1", render_mode="human")
obs, info = test_env.reset()
score = 0
episode = 1
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)

    score += reward

    if terminated or truncated:
        print("Episode:{} Score:{}".format(episode, score))
        obs, _ = test_env.reset()
        score = 0
        episode += 1

env.close()
test_env.close()