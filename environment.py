from itertools import count
import gymnasium as gym
from matplotlib import pyplot as plt
import torch
import torch.optim as optim

from device import standard as device
from memory import ReplayMemory
from model import DQN
from model.optimize import optimize_model
from utils.functions import plot_durations, select_action
from utils.hyperparameters import BATCH_SIZE, GAMMA, LR, TAU

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def launch():
    env = gym.make('CartPole-v1', render_mode = 'human')
    n_actions = env.action_space.n
    state, info = env.reset()
    n_obs = len(state)

    policy_net = DQN(n_observations=n_obs, n_actions=n_actions).to(device)
    target_net = DQN(n_observations=n_obs, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    print(f'Using device: {device}')
    if device == 'cpu':
        num_episodes = 50
    else:
        num_episodes = 600
    
    episode_durations = []
    
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state=state, policy_net=policy_net, env=env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to next state
            state = next_state

            # Perform one step of the optimization on the policy network
            optimize_model(optimizer=optimizer,
                           memory=memory,
                           policy_net=policy_net,
                           target_net=target_net,
                           GAMMA=GAMMA,
                           BATCH_SIZE=BATCH_SIZE)
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = TAU * policy_net_state_dict[key] + target_net_state_dict[key] * (1-TAU)
            if done:
                episode_durations.append(t+1)
                plot_durations(episode_durations=episode_durations, show_result = False)
                break
            env.render()
    print('Complete')
    plot_durations(episode_durations=episode_durations, show_result=True)
    plt.ioff()
    plt.show()
    print('Close the plot to exit')

if __name__ == '__main__':
    launch()