import torch
import torch.nn as nn
from device import standard as device
from utils.hyperparameters import TAU

class PolicyOptimizer:
    """
    Class responsible for optimizing the policy network using a batch of transitions from a replay memory.
    
    Parameters:
    optimizer (torch.optim.Optimizer): the optimizer used to update the policy network
    qlearner (DQLearner): a DQLearner object used to calculate the Q values, expected Q values, and V values for a batch of transitions
    memory (ReplayMemory): a replay memory object containing a collection of transitions
    policy_network (torch.nn.Module): the policy network to be optimized
    target_network (torch.nn.Module): a target network used to compute expected state-action values
    BATCH_SIZE (int): the batch size to use for optimization
    GAMMA (float): the discount factor
    """
    def __init__(self, optimizer, qlearner, memory,policy_network, target_network, BATCH_SIZE, GAMMA):
        self.optimizer = optimizer
        self.qlearner = qlearner
        self.memory = memory
        self.policy_network = policy_network
        self.target_network = target_network
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.device = device
        self.criterion = nn.SmoothL1Loss()

    def optimize(self):
        """
        Optimizes the policy network using a batch of transitions from the memory object.
        
        Returns:
        None
        
        """
        if self._not_enough_samples():
            return
        state_action_values, _, expected_state_action_values = self.qlearner.calculate()
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
    
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        self._soft_update()
    
    def _not_enough_samples(self):
        """
        Returns True if there are not enough samples in the memory to form a batch, False otherwise.
        """
        return len(self.memory) < self.BATCH_SIZE

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