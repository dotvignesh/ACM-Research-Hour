import gymnasium as gym
import panda_gym
import numpy as np
import torch
from collections import deque
import time

class TransformerPolicy(torch.nn.Module):
    def __init__(self, input_dim=6, action_dim=3, hidden_dim=128, num_layers=3):
        super().__init__()
        self.embedding = torch.nn.Linear(input_dim, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, action_dim),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return 5.0 * self.action_head(x)

def evaluate_model(model, num_episodes=10, context_length=5):
    env = gym.make("PandaReach-v3", render_mode="human")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    total_rewards = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        obs_buffer = deque(maxlen=context_length)
        episode_reward = 0
        
        while True:
            # Prepare current observation
            current_pos = observation["observation"][0:3]
            desired_pos = observation["desired_goal"][0:3]
            current_obs = np.concatenate((current_pos, desired_pos))
            
            # Update observation buffer
            obs_buffer.append(current_obs)
            
            # If buffer not full, pad with zeros
            if len(obs_buffer) < context_length:
                padded_obs = np.zeros((context_length, 6))
                padded_obs[-len(obs_buffer):] = np.array(list(obs_buffer))
            else:
                padded_obs = np.array(list(obs_buffer))
            
            # Convert to tensor and get model prediction
            obs_tensor = torch.FloatTensor(padded_obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()[0]
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Render and add small delay for visualization
            time.sleep(0.1)
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} Reward: {episode_reward:.2f}")
    
    env.close()
    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    # Load the trained model
    model = TransformerPolicy()
    model.load_state_dict(torch.load("transformer_policy_mps.pth"))
    
    # Run evaluation
    evaluate_model(model, num_episodes=10)

