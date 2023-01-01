import torch
import torch.nn as nn
from memory import Transition
from device import standard as device

class PolicyOptimizer:
    """
    Class responsible for optimizing the policy network using a batch of transitions from a replay memory.
    
    Parameters:
    optimizer (torch.optim.Optimizer): the optimizer used to update the policy network
    memory (ReplayMemory): a replay memory object containing a collection of transitions
    policy_network (torch.nn.Module): the policy network to be optimized
    target_network (torch.nn.Module): a target network used to compute expected state-action values
    BATCH_SIZE (int): the batch size to use for optimization
    GAMMA (float): the discount factor
    """
    def __init__(self, optimizer, memory,policy_network, target_network, BATCH_SIZE, GAMMA):
        self.optimizer = optimizer
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
        if self._enough_samples():
            return
        batch = self._sample_transitions()
        state_b, action_b, non_final_next_states_b, reward_b = self._single_batches(batch)
        state_action_values = self._q_values(states=state_b, actions=action_b)
        next_state_values = self._v_values(batch=batch, non_final_next_states=non_final_next_states_b)
        expected_state_action_values = self._expected_q_values(reward_b, next_state_values)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
    
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()
    
    ### PRIVATE METHODS
    def _enough_samples(self):
        """
        Returns True if there are not enough samples in the memory to form a batch, False otherwise.
        """
        return len(self.memory) < self.BATCH_SIZE

    def _sample_transitions(self):
        """
        Samples a batch of transitions from the agent's memory.
        
        Returns:
        Transition: a batch of transitions
        """
        transitions = self.memory.sample(self.BATCH_SIZE)
        return Transition(*zip(*transitions))

    def _non_final_mask(self, batch):
        """
        Computes a mask indicating which states in a batch of transitions are non-final.
        
        Parameters:
        batch (Transition): a batch of transitions
        
        Returns:
        torch.Tensor: a boolean tensor representing the mask
        """
        return torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool,
            device=self.device
        )

    def _single_batches(self, batch):
        """
        Extracts and concatenates the corresponding elements of a batch of transitions.
        
        Parameters:
        batch (Transition): a batch of transitions
        
        Returns:
        Tuple[torch.Tensor]: a tuple containing the concatenated state, action, next state, and reward tensors
        """
        return (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat([s for s in batch.next_state if s is not None]),
            torch.cat(batch.reward)
        )

    def _q_values(self, states, actions):
        """
        Computes the Q values for a batch of states and actions.
        
        Parameters:
        states (torch.Tensor): a tensor of states
        actions (torch.Tensor): a tensor of actions
        
        Returns:
        torch.Tensor: a tensor of Q values
        """
        return self.policy_network(states).gather(1, actions)

    def _expected_q_values(self, rewards, next_state_values):
        """
        Computes the expected Q values for a batch of rewards and next state values.
        
        Parameters:
        rewards (torch.Tensor): a tensor of rewards
        next_state_values (torch.Tensor): a tensor of next state values
        
        Returns:
        torch.Tensor: a tensor of expected Q values
        """
        return rewards + (self.GAMMA * next_state_values)

    def _v_values(self, batch, non_final_next_states):
        """
        Computes the V values for a batch of non-final next states.
        
        Parameters:
        batch (Transition): a batch of transitions
        non_final_next_states (torch.Tensor): a tensor of non-final next states
        
        Returns:
        torch.Tensor: a tensor of V values
        """
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        non_final_mask = self._non_final_mask(batch=batch)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        
        return next_state_values