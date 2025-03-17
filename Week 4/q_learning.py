import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict
from rlang.grounding.utils.primitives import VectorState
import rlang
from rlang.agents.RLangPolicyAgentClass import RLangPolicyAgent
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import os  # To force quit Pygame if needed


class RLangQLearningAgent:
    def __init__(self, env,knowledge=None, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay=0.0001):
        self.env = env
        self.alpha = alpha  
        self.gamma = gamma 
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay 
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0))
        self.knowledge = knowledge

    def weighted_reward(self, r_func, state_dict, action):
        return sum(r_func(state=VectorState(k), action=action) * v for k, v in state_dict.items())
    
    def weighted_value(self, q_func, state_dict, actions):
        return sum(max(q_func[k][a] for a in actions) * v for k, v in state_dict.items())
    
    def initialize_q_table_with_knowledge(self, states, actions):
        if self.knowledge:
            reward_function = self.knowledge.reward_function
            transition_function = self.knowledge.transition_function
            
            if reward_function:
                for s in states:
                    taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                    for i, a in enumerate(actions):
                        self.q_table[s][a] = reward_function(state=VectorState([taxi_row, taxi_col, passenger_location, destination]), action=i)
            
            if transition_function and reward_function:
                for s in states:
                    taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                    for i, a in enumerate(actions):
                        s_primei = transition_function(state=VectorState([taxi_row, taxi_col, passenger_location, destination]), action=i)
                        if s_primei:
                            r_prime = self.weighted_reward(reward_function, s_primei, action=i)
                            v_s_prime = self.weighted_value(self.q_table, s_primei, actions)
                            self.q_table[s][a] += self.alpha * (r_prime + self.gamma * v_s_prime)
    
    def train(self, episodes):
        states = range(self.env.observation_space.n)
        actions = range(self.env.action_space.n)
        
        if self.knowledge:
            self.initialize_q_table_with_knowledge(states, actions)
        
        rewards_per_episode = np.zeros(episodes)
        rng = np.random.default_rng()
        
        for i in tqdm(range(episodes)):
            state = self.env.reset()[0]
            terminated, truncated, rewards = False, False, 0
            
            while not (terminated or truncated):
                if rng.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = max(self.q_table[state], key=self.q_table[state].get, default=self.env.action_space.sample())
                
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards += reward
                
                max_q = max(self.q_table[new_state].values(), default=0)
                self.q_table[state][action] += self.alpha * (reward + self.gamma * max_q - self.q_table[state][action])
                state = new_state
                
            self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
            self.alpha = 0.0001 if self.epsilon == 0 else self.alpha
            rewards_per_episode[i] = rewards
            
      
        self.env.close()
        

        return rewards_per_episode
    
    def test(self, episodes=10, render=True):
        self.env = gym.make('Taxi-v3', render_mode='human')
    
        pygame.display.set_mode((500, 500))

        rewards_per_episode = np.zeros(episodes)
        for i in range(episodes):
            state = self.env.reset()[0]
        
            terminated, truncated, rewards = False, False, 0
            
            while not (terminated or truncated):
                action = max(self.q_table[state], key=self.q_table[state].get, default=self.env.action_space.sample())
                state, reward, terminated, truncated, _ = self.env.step(action)
                rewards += reward
                
            rewards_per_episode[i] = rewards
        
        if render:
            self.env.close()
            self.env = gym.make('Taxi-v3')
            self.env.reset()
        
        self.env.close()
        pygame.quit()
        print(f"Average reward over {episodes} test episodes: {np.mean(rewards_per_episode)}")
        return np.mean(rewards_per_episode)
    

    def plot_training_rewards(self,rewards, window_size=100,save_path="training_rewards.png"):
        episodes = np.arange(len(rewards))
    
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label="Rewards per Episode", alpha=0.3)
        plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f"Moving Average (window={window_size})", color='red')
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards Over Episodes")
        plt.legend()
        plt.grid()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        print(f"Plot saved as {save_path}")

        plt.close()  

if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    np.set_printoptions(threshold=np.inf) 

    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
    rewards_with_policy = agent_with_policy.train(episodes=15000)
    print(f"Training complete. Average reward: {agent_with_policy.test(10)}")
    agent = RLangQLearningAgent(env)
    rewards = agent.train(episodes=15000)
    print(f"Training complete. Average reward: {agent.test(10)}")
    agent.plot_training_rewards(rewards_with_policy,save_path="./plots/q_learning_training_rewards_knowledge.png")
    agent.plot_training_rewards(rewards,save_path="./plots/q_learning_training_rewards.png")

