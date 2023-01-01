import gymnasium as gym
from agent import Agent

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def launch():
    env = gym.make('CartPole-v1', render_mode = 'human')
    # env = Environment()

    agent = Agent(environment=env, memory_capacity=10000)
    agent.train()

if __name__ == '__main__':
    launch()