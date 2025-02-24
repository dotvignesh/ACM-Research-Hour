import gymnasium as gym
import panda_gym
import numpy as np
import json
import time 

env = gym.make("PandaReach-v3", render_mode="human")
observation, info = env.reset()

data = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "dones": []
}

for _ in range(10000):
    current_position = observation["observation"][0:3]
    desired_position = observation["desired_goal"][0:3]
    action = 5.0 * (desired_position - current_position)
    
    data["observations"].append(np.concatenate((current_position, desired_position)).tolist())
    data["actions"].append(action.tolist())
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    data["rewards"].append(reward)
    data["dones"].append(terminated or truncated)

    time.sleep(0.1)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# Save data to a JSON file
# with open("expert_demonstrations.json", "w") as f:
#     json.dump(data, f)