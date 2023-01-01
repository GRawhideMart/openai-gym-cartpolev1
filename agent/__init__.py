from itertools import count
import math
import random
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from memory import ReplayMemory

from model import DQN
from device import standard as device
from model.optimize import PolicyOptimizer
from utils.hyperparameters import BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START, GAMMA, LR, TAU

class Agent:
    """
    An agent that learns to interact with the environment using a policy network. It stores transitions in a replay memory
    and optimizes the policy network using this memory. It also has a target network used for the optimization process.
    """    

    def __init__(self, environment, memory_capacity):
        """
        Initializes the agent, policy network, target network, replay memory and the optimizer.
        :param environment: The environment the agent interacts with.
        :param memory_capacity: The capacity of the replay memory.
        """
        self.device = device
        self.environment = environment
        self.state, self._ = self.environment.reset()
        self.n_states = len(self.state)
        self.n_actions = environment.action_space.n
        self.policy_network = DQN(n_observations=self.n_states, n_actions=self.n_actions).to(self.device)
        self.target_network = DQN(n_observations=self.n_states, n_actions=self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.policy_optimizer = PolicyOptimizer(optimizer=self.optimizer,
                                       memory=self.memory,
                                       policy_network=self.policy_network,
                                       target_network=self.target_network,
                                       BATCH_SIZE=BATCH_SIZE,
                                       GAMMA=GAMMA)
        self.steps_done = 0
        self.episode_durations = []

    def train(self):
        """
        Trains the agent on the environment.
        """
        plt.ion()
        num_episodes = 600 if self.device == 'cuda' else 50
        for i_episode in range(num_episodes):
            state = self._compute_state()
            for t in count():
                action, observation, reward, terminated, truncated = self._play_step(state=state)
                done = terminated or truncated
                next_state = self._compute_next_state(terminated, observation)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                # Move to next state
                state = next_state

                # Perform one step of the optimization on the policy network
                self._optimize()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                self._soft_update()

                if done:
                    self.episode_durations.append(t+1)
                    self._plot_durations(show_result=False)
                    break
                self.environment.render()
        self._train_end_routine()

    def _optimize(self):
        """
        (Inherited from PolicyOptimizer)
        Optimizes the policy network using a batch of transitions from the memory object.
        
        Returns:
        None
        
        """
        self.policy_optimizer.optimize()

    def _soft_update(self):
        """Soft update of the target network's weights.
        θ′ ← τ θ + (1 − τ) θ′

        Parameters
        ----------
        self : object
            Agent object

        Returns
        -------
        None
        """
        policy_net_state_dict = self.policy_network.state_dict()
        target_net_state_dict = self.target_network.state_dict()
        for key in policy_net_state_dict:
                target_net_state_dict[key] = TAU * policy_net_state_dict[key] + target_net_state_dict[key] * (1-TAU)
        self.target_network.load_state_dict(target_net_state_dict)

    def _select_action(self, state, policy_network, environment):
        """
        Selects an action from the given policy network using an epsilon-greedy strategy.

        Args:
            state (torch.Tensor): The current state of the environment.
            policy_network (nn.Module): The neural network policy to use for action selection.
            environment (gym.Env): The gym environment being used.

        Returns:
            torch.Tensor: The selected action.
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[environment.action_space.sample()]], device=device, dtype=torch.long)

    def _compute_state(self):
        """Computes the state of the environment.
    
        Returns:
            state (torch.tensor): The state of the environment.
        """
        state, info = self.environment.reset()
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _play_step(self, state):
        """
        Play a single step of the game, by selecting an action, taking it, and returning the resulting observations, rewards, and termination status.

        Parameters:
        - state (torch.tensor): current state of the game

        Returns:
        - action (torch.tensor): action taken
        - observation (tuple): new state, reward, termination status, and other information
        - reward (torch.tensor): reward received
        - terminated (bool): whether the game terminated
        - truncated (bool): whether the game terminated due to reaching the maximum number of steps
        """
        action = self._select_action(state=state, policy_network=self.policy_network, environment=self.environment)
        observation, reward, terminated, truncated, _ = self.environment.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated
        return action, observation, reward, terminated, truncated
    
    def _compute_next_state(self, terminated, observation):
        """
        Compute the next state based on whether the last environment step terminated. If the last environment
        step terminated, return None. Otherwise, return the next state as a tensor.

        Parameters:
        terminated (bool): whether the last environment step terminated
        observation (list): the observation of the environment at the last step

        Returns:
        torch.Tensor or None: the next state as a tensor if the last environment step did not terminate, else None
        """
        if terminated:
            return None
        else:
            return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    def _plot_durations(self, show_result=False):
        """Plots the durations of episodes.
    
        Args:
            show_result (bool): If set to True, the result of training will be shown. Otherwise, training progress will be shown.
        """
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
            plt.pause(0.001)  # pause a bit so that plots are updated

    def _train_end_routine(self):
        """
        This method is called after the training loop has completed. It closes the plot window and displays the final plot.
        """
        print('Complete')
        self._plot_durations(show_result=True)
        plt.ioff()
        plt.show()
        print('Close the plot to exit')