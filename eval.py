

from env_warehouse_with_obstacles import JetBotEnv
from stable_baselines3 import PPO
import datetime as dt

policy_path = "./cnn_policy/jetbot_policy_warehouse_2M_with_obstacles.zip"

my_env = JetBotEnv(headless=False)
model = PPO.load(policy_path)


for _ in range(10):
    obs, _ = my_env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation=obs, deterministic=False)
        obs, reward, done, truncated, info = my_env.step(action)
        my_env.render()
        # Get the current epoch time in seconds
        epoch_time = dt.datetime.now().timestamp()
        #capture_viewport_to_file(vp_api, f"/root/Documents/output_{epoch_time}.png")


my_env.close()
