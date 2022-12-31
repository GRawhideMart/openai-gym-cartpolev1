import gymnasium as gym
import torch.optim as optim

from device import standard as device
from memory import ReplayMemory
from model import DQN
from utils.hyperparameters import LR

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

    for _ in range(1000):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            state, info = env.reset()
        env.render()

    env.close()

if __name__ == '__main__':
    launch()